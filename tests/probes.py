"""Probes over a request's tokens, with seeded random weights.

Three published sequence-level probes and a token-wise dictionary encoder, in
the form `Coalescer.register` takes them. A span probe is called as
``probe(rows, sizes=[n_1, ..., n_K])``: ``rows`` is K requests' token rows
concatenated, and item ``i`` of the result is computed from request ``i``'s
``n_i`` rows only. The weights are random, so the outputs are not meaningful
scores.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def seeded(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """``module`` with every parameter redrawn from a generator seeded ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return module


def owners(sizes: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """For each of ``sum(sizes)`` rows: which span it is in, and how many rows
    of its span come after it."""
    counts = torch.tensor(sizes, device=device)
    span = torch.repeat_interleave(torch.arange(len(sizes), device=device), counts)
    after = counts.cumsum(0)[span] - 1 - torch.arange(int(counts.sum()), device=device)
    return span, after


class AttentionPoolProbe(torch.nn.Module):
    """Attention pooling and max pooling over a request's tokens, then a linear
    readout (Kramar et al. 2026; McKenzie et al. 2025). Each head's softmax runs
    over the request's own tokens."""

    def __init__(self, width: int, heads: int = 8, values: int = 64, classes: int = 2, seed: int = 0) -> None:
        super().__init__()
        self.heads = heads
        self.query = torch.nn.Linear(width, heads, bias=False)
        self.value = torch.nn.Linear(width, values, bias=False)
        self.readout = torch.nn.Linear(heads * values + values, classes)
        seeded(self, seed)

    def forward(self, rows: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        rows = rows.float()
        scores = pad_sequence(self.query(rows).split(sizes), batch_first=True)  # [K, L, heads]
        values = pad_sequence(self.value(rows).split(sizes), batch_first=True)  # [K, L, values]
        lengths = torch.tensor(sizes, device=rows.device)
        absent = torch.arange(scores.shape[1], device=rows.device) >= lengths[:, None]  # [K, L]
        weights = torch.softmax((scores / self.heads**0.5).masked_fill(absent[..., None], float("-inf")), dim=1)
        pooled = weights.transpose(1, 2) @ values  # [K, heads, values]
        peak = values.masked_fill(absent[..., None], float("-inf")).amax(dim=1)  # [K, values]
        return self.readout(torch.cat([pooled.flatten(1), peak], dim=-1))  # [K, classes]


class AggregateProbe(torch.nn.Module):
    """An exponentially weighted average, the mean and the mean of the last
    tokens of a request, then a linear readout (Cunningham et al. 2026). A
    token's weight depends on its distance from the end of its own request."""

    def __init__(self, width: int, decay: float = 0.9, suffix: int = 4, classes: int = 2, seed: int = 0) -> None:
        super().__init__()
        self.decay, self.suffix = decay, suffix
        self.readout = torch.nn.Linear(3 * width, classes)
        seeded(self, seed)

    def forward(self, rows: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        rows = rows.float()
        span, after = owners(sizes, rows.device)
        features = []
        for weight in (self.decay ** after.float(), torch.ones_like(after).float(), (after < self.suffix).float()):
            total = torch.zeros(len(sizes), device=rows.device).index_add_(0, span, weight)
            summed = torch.zeros(len(sizes), rows.shape[1], device=rows.device).index_add_(0, span, weight[:, None] * rows)
            features.append(summed / total[:, None])
        return self.readout(torch.cat(features, dim=-1))  # [K, classes]


class StructuralProbe(torch.nn.Module):
    """Squared distances between every pair of a request's tokens after a
    linear projection (Hewitt and Manning 2019): ``[n_i, n_i]`` per request."""

    def __init__(self, width: int, rank: int = 64, seed: int = 0) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(width, rank, bias=False)
        seeded(self, seed)

    def forward(self, rows: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        projected = pad_sequence(self.projection(rows.float()).split(sizes), batch_first=True)  # [K, L, rank]
        distances = torch.cdist(projected, projected).square()  # [K, L, L]
        return [distances[index, :size, :size] for index, size in enumerate(sizes)]


class Dictionary(torch.nn.Module):
    """A sparse-autoencoder encoder: token-wise, of a class outside every
    runtime's token-wise list."""

    def __init__(self, width: int, features: int = 512, seed: int = 0) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(width))
        self.encoder = torch.nn.Parameter(torch.zeros(features, width))
        seeded(self, seed)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.relu((rows.float() - self.bias) @ self.encoder.T)
