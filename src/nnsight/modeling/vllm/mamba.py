"""Recurrent kernel sites on vLLM's Mamba2 layers (Mamba2, Nemotron-H, Jamba, Granite hybrid, Falcon-H1, Zamba2).

vLLM's ``MambaMixer2`` runs its state-space recurrence in ``conv_ssm_forward``,
once per step for every scheduled request, and keeps each request's state in a
slot of its state cache. Inside, a request goes through one of two kernels:

* **Prefill**: ``mamba_chunk_scan_combined_varlen`` over the prefilling requests
  packed into one strip of tokens. vLLM cuts the strip into chunks at every
  multiple of the model's chunk size and at every request boundary, and the
  scan's ``_state_passing_fwd`` returns the state at the end of every chunk.
* **Decode**: ``selective_state_update`` advances the request's cache slot in
  place by one token.

`serve` is registered on ``conv_ssm_forward`` and records both. The `Scan`
reader serves the sites of the transformers reader
([`Recurrence`][nnsight.modeling.kernels.recurrent.Recurrence]) in vLLM's
layout, without a batch axis:

* ``chunk_states``: ``[chunks + 1, heads, head_dim, state]``. On a prefill
  step, the state entering each of the request's chunks as the kernel cut them
  in this step, then its final state. On a decode step, the state before the
  token and the state after it.
* ``states``: ``[tokens, heads, head_dim, state]``, the state after each token
  of the step. A token that ends a chunk is the kernel's own state; one inside
  a chunk is advanced from the chunk's entering state in fp32 with the layer's
  recurrence ``S_t = exp(dt_t A) S_{t-1} + (dt_t x_t) B_t^T``.

Because the cut follows the strip, a request that is not first in a step's
prefill strip has a shorter first chunk. ``states`` does not depend on the cut.

Under tensor parallelism each rank's layer holds ``num_heads // tp_size`` heads
and their states, so ``heads`` is `SPLIT`.

Not served: the ReplaySSM decode path and speculative decoding.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ...intervention import sites
from ...intervention.kernel import Attr, Located, Spec, unshimmed
from ...intervention.sites import SPLIT, Batch, Provenance, Sites, Unavailable
from ...intervention.source import arguments
from ..kernels.recurrent import Recurrence

NAME = "MambaMixer2.conv_ssm_forward"


class _Capture(threading.local):
    """Thread-local list collecting one record per kernel call during `serve` (``None`` outside a served call)."""

    calls: list | None = None


_capture = _Capture()
_installed = False
_lock = threading.Lock()


def _install() -> None:
    """Wrap the layer's kernels once so that, inside a served call, each records what the reader needs."""
    global _installed
    with _lock:
        if _installed:
            return
        from vllm.model_executor.layers.mamba import mamba_mixer2 as layer
        from vllm.model_executor.layers.mamba.ops import ssd_combined

        scan, passing, update = layer.mamba_chunk_scan_combined_varlen, ssd_combined._state_passing_fwd, layer.selective_state_update

        def recording_scan(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            if calls is None:
                return scan(*args, **kwargs)
            calls.append(call := {"kind": "scan", **arguments(scan, args, kwargs)})
            output = scan(*args, **kwargs)
            call["closed"] = True
            return output

        def recording_passing(*args: Any, **kwargs: Any) -> Any:
            result = passing(*args, **kwargs)
            calls = _capture.calls
            if calls and calls[-1]["kind"] == "scan" and "closed" not in calls[-1]:
                calls[-1]["ends"] = result
            return result

        def recording_update(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            if calls is None:
                return update(*args, **kwargs)
            bound = arguments(update, args, kwargs)
            cache, slots = bound["state"], bound["state_batch_indices"]
            # One token per request and one slot per request: the non-speculative layout.
            if slots is None or slots.reshape(slots.shape[0], -1).shape[1] != 1 or slots.shape[0] != bound["x"].shape[0]:
                calls.append({"kind": "unsupported"})
                return update(*args, **kwargs)
            slots = slots.reshape(-1).long()
            before = cache[slots].clone()
            output = update(*args, **kwargs)
            calls.append({"kind": "update", "before": before, "after": cache[slots].clone()})
            return output

        layer.mamba_chunk_scan_combined_varlen = recording_scan
        ssd_combined._state_passing_fwd = recording_passing
        layer.selective_state_update = recording_update
        _installed = True


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run ``conv_ssm_forward`` with its kernels recording, and hand back a `Scan` over what they did."""
    from vllm.forward_context import get_forward_context

    _install()
    outer, _capture.calls = _capture.calls, (calls := [])
    try:
        output = fn(*args, **kwargs)
    finally:
        _capture.calls = outer
    layer = args[0]
    metadata = get_forward_context().attn_metadata
    metadata = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
    if metadata is None:
        return output, Unavailable(f"{NAME} ran without attention metadata; its kernel sites are not served on this call")
    if layer.use_replayssm or layer.num_spec > 0:
        return output, Unavailable(f"{NAME}: kernel sites are not served on the ReplaySSM decode path or under speculative decoding")
    if any(call["kind"] == "unsupported" for call in calls):
        return output, Unavailable(f"{NAME}: the decode kernel advanced more than one token per request; its kernel sites are not served on this call")
    return output, Scan({call["kind"]: call for call in calls}, metadata.num_decode_tokens)


def _step(call: dict[str, Any], tokens: slice) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """The layer's fp32 recurrence over one request's rows of a scan call: ``S_t = exp(dt_t A) S_{t-1} + (dt_t x_t) B_t^T``."""
    x, groups = call["x"][tokens].float(), call["B"][tokens].float()
    dt = F.softplus(call["dt"][tokens].float() + call["dt_bias"].float()) if call["dt_softplus"] else call["dt"][tokens].float()
    dt = dt.clamp(*call["dt_limit"])
    decay = torch.exp(dt * call["A"].float())
    # A group's input projection is shared by its heads.
    inputs = groups.repeat_interleave(x.shape[1] // groups.shape[1], dim=1)

    def step(state: torch.Tensor, t: int) -> torch.Tensor:
        return state * decay[t, :, None, None] + (dt[t, :, None] * x[t]).unsqueeze(-1) * inputs[t].unsqueeze(-2)

    return step


class Scan(Sites):
    """Site reader for one ``conv_ssm_forward`` call: the step's decoding requests first, then its prefilling ones."""

    names = Recurrence.names
    axes = Recurrence.axes
    #: Each rank's layer holds one block of the heads, and so does its state.
    layout = {"tp": {"heads": SPLIT}}
    narrowed = True

    def __init__(self, calls: dict[str, dict[str, Any]], decoding: int) -> None:
        self.calls = calls
        #: Token rows of the step's decoding requests, one each; the prefilling requests' rows follow.
        self.decoding = decoding
        self.batch: Batch | None = None

    def local(self, axis: str) -> int:
        call = next(iter(self.calls.values()))
        return (call["before"] if call["kind"] == "update" else call["x"]).shape[1]

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        self.batch = batch
        return super().read(site, spec, batch)

    def _request(self) -> tuple[list[int], torch.Tensor, Callable | None]:
        """The requesting worker's chunk starts (plus its length) in its own tokens, the state entering each chunk then the final one, and its per-token step."""
        if self.batch is None:
            if len(self.calls) != 1:
                raise ValueError(f"{NAME} ran prefilling and decoding requests in one call and the trace has no batcher to tell them apart")
            start, size = 0, None
        else:
            start, size = self.batch.start, self.batch.size
        if start < self.decoding or (self.batch is None and "update" in self.calls):
            call = self.calls["update"]
            if size not in (None, 1) or (size is None and call["before"].shape[0] != 1):
                raise ValueError(f"kernel sites: the worker's rows [{start}, {start + (size or 0)}) are not one decoding request's row in this call")
            return [0, 1], torch.stack([call["before"][start], call["after"][start]]), None
        call = self.calls["scan"]
        starts, cuts, first = call["cu_seqlens"].tolist(), call["cu_chunk_seqlens"].tolist(), start - self.decoding
        if size is None:
            if len(starts) != 2:
                raise ValueError(f"{NAME} prefilled {len(starts) - 1} requests in one call and the trace has no batcher to tell them apart")
            size = starts[1]
        if first not in starts[:-1] or starts[starts.index(first) + 1] != first + size:
            raise ValueError(f"kernel sites: the worker's rows [{start}, {start + size}) are not one request's rows in this call (prefill boundaries {starts}, after {self.decoding} decoding rows)")
        request = starts.index(first)
        chunks = [index for index, cut in enumerate(cuts[:-1]) if first <= cut < first + size]
        ends = call["ends"][chunks[0] : chunks[-1] + 1].unflatten(-1, call["x"].shape[-1:] + (-1,))
        initial = call["initial_states"]
        entering = torch.zeros_like(ends[0]) if initial is None else initial[request].to(ends.dtype)
        bounds = [cuts[index] - first for index in chunks] + [size]
        return bounds, torch.cat([entering.unsqueeze(0), ends]), _step(call, slice(first, first + size))

    def chunk_states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        _, states, _ = self._request()
        device = states.device
        value = sites.select(sites.select(states, 0, sites.indices(spec.get("chunks"), states.shape[0], device)), 1, sites.indices(spec.get("heads"), states.shape[1], device))
        note = "the state entering each chunk the kernel cut in this step, then the final state"
        return sites.squeeze(value, spec, {"chunks": 0, "heads": 1}), Provenance("exact export", NAME, note)

    def states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        bounds, states, step = self._request()
        heads = sites.indices(spec.get("heads"), states.shape[1], states.device)
        wanted = sites.offsets(spec.get("tokens"), bounds[-1])
        sites.warn_if_large((len(wanted), *sites.select(states[0], 0, heads).shape), "states", "tokens= or heads=")
        found: dict[int, torch.Tensor] = {}
        for chunk, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            inside = sorted({token for token in wanted if begin <= token < end})
            if not inside:
                continue
            if inside == [end - 1]:
                found[end - 1] = states[chunk + 1].float()  # the kernel's own state at the end of the chunk
                continue
            state = states[chunk].float()
            for token in range(begin, inside[-1] + 1):
                state = step(state, token)
                if token in inside:
                    found[token] = state
            if end - 1 in found:
                found[end - 1] = states[chunk + 1].float()
        value = sites.select(torch.stack([found[token] for token in wanted]), 1, heads)
        note = "a token that ends a chunk is the kernel's state; one inside a chunk is advanced from the chunk's entering state in fp32"
        return sites.squeeze(value, spec, {"tokens": 0, "heads": 1}), Provenance("derived", NAME, note)


def conv_ssm_forward(context: None) -> Located:
    """Locate ``MambaMixer2.conv_ssm_forward``, which the layer's custom op reads off the layer on every call."""
    try:
        from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    except ImportError:  # a vLLM without this layer
        return
    yield unshimmed(MambaMixer2.conv_ssm_forward), (Attr(MambaMixer2, "conv_ssm_forward"),)


CATALOG: list[Spec] = [
    Spec(NAME, "recurrent", conv_ssm_forward, serve),
]
