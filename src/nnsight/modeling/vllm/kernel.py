"""Attention kernel sites on vLLM's paged FlashAttention backend.

vLLM's `Attention` layer calls ``flash_attn_varlen_func`` once per step over a
packed batch: the query rows of every scheduled request are concatenated
(``cu_seqlens_q`` gives each request's span), and keys and values are read from
the paged KV cache through ``block_table``. The kernel never writes the
probability matrix, but it returns the per-row logsumexp when called with
``return_softmax_lse=True``, in the same launch that fills ``out``. `serve` asks
for it, and `Paged` reconstructs ``exp(scores - lse)`` in fp32 for the requesting
worker's rows, gathering that request's keys from its cache pages.

Values follow vLLM's layout ([`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher]):
no batch axis, and the query axis holds the rows this step computed for the
request. A prefill step serves every prompt token as a query row, a decode step
serves one, and the key axis always spans every token of the request so far.

* ``lse``: ``[queries, heads]``, the kernel's own logsumexp (``"exact export"``).
* ``scores``: ``[queries, heads, keys]``, ``q . k * scale`` in fp32, ``-inf``
  above the causal diagonal (``"exact reconstruction"``).
* ``probs``: ``[queries, heads, keys]``, ``exp(scores - lse)``.

Under tensor parallelism each rank's call holds one block of the heads. The
reader declares ``heads`` as `SPLIT`, and `Sites.read` assembles the ranks'
blocks, so ``heads=`` indexes the model's heads on any number of ranks.

Sites take ``query=``, ``heads=`` and ``key=`` with the indexing rules of
intervention/sites.py. Calls the reconstruction does not model (a sliding
window, ALiBi, a logit soft cap, attention sinks, a custom mask, a key cache in
another dtype than the queries such as fp8, a call without a block table, or
one of the several calls vLLM merges under cascade or context-parallel
attention) are served as `Unavailable`.

`CATALOG` has one row, registered in the worker process (`register_kernels`), on the eager
engine: under ``taps`` a replayed step runs no module controller, so the shim
finds no running module and the site is not served.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ...intervention import sites
from ...intervention.attention import _exact_fp32
from ...intervention.kernel import Attr, Located, Spec, register_catalog, unshimmed
from ...intervention.sites import SPLIT, Batch, Provenance, Ranks, Sites, Unavailable
from ...intervention.source import arguments

NAME = "flash_attn_varlen_func"


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run ``flash_attn_varlen_func`` with its logsumexp returned and hand back what the caller asked for and a `Paged` reader."""
    bound = arguments(fn, args, kwargs)
    asked = bound["return_softmax_lse"]
    output, lse = fn(**{**bound, "return_softmax_lse": True})
    result = (output, lse) if asked else output
    # vLLM asks for the logsumexp itself only to merge several kernel calls into one attention (cascade and
    # context-parallel paths), where one call covers part of each request's keys.
    reason = "the call is one of several that vLLM merges into one attention" if asked else _unsupported(bound)
    if reason is not None:
        return result, Unavailable(f"{NAME}: {reason}; its kernel sites are not served on this call")
    return result, Paged(bound, lse)


def _unsupported(bound: dict[str, Any]) -> str | None:
    """Why the reconstruction does not model this call, or ``None``."""
    if bound["block_table"] is None or bound["cu_seqlens_q"] is None or bound["seqused_k"] is None:
        return "the call does not read keys from the paged cache"
    if bound["k"].dtype != bound["q"].dtype:
        return f"the key cache is stored as {bound['k'].dtype} and the queries as {bound['q'].dtype}"
    if bound["s_aux"] is not None:
        return "the call adds attention sinks"
    if bound["mask_mod"] is not None or bound["dynamic_causal"] is not None:
        return "the call uses a custom attention mask"
    if bound["alibi_slopes"] is not None:
        return "the call adds ALiBi biases"
    if bound["softcap"]:
        return f"the call applies a logit soft cap of {bound['softcap']}"
    window = bound["window_size"]
    if window is not None and tuple(window) != (-1, -1):
        return f"the call uses a sliding window of {tuple(window)}"
    return None


class Paged(Sites):
    """Site reader for one ``flash_attn_varlen_func`` call over the paged KV cache.

    Holds the call's arguments by reference. They are the engine's live buffers,
    valid while the worker reads during the handoff, and every site returns a
    freshly computed tensor.
    """

    names = ("lse", "scores", "probs")
    axes = {"lse": ("query", "heads"), "scores": ("query", "heads", "key"), "probs": ("query", "heads", "key")}
    #: vLLM shards the query projection by head: rank ``r`` of ``n`` holds heads ``[r * heads / n, (r + 1) * heads / n)``.
    layout = {"tp": {"heads": SPLIT}}
    narrowed = True

    def __init__(self, bound: dict[str, Any], lse: torch.Tensor) -> None:
        self.q: torch.Tensor = bound["q"].detach()
        self.cache: torch.Tensor = bound["k"].detach()
        self.block_table: torch.Tensor = bound["block_table"]
        self.spans: list[int] = bound["cu_seqlens_q"].tolist()
        self.used: list[int] = bound["seqused_k"].tolist()
        self.scale: float = bound["softmax_scale"] if bound["softmax_scale"] is not None else self.q.shape[-1] ** -0.5
        self.causal = bool(bound["causal"])
        #: ``[heads, total query rows]`` fp32, as the kernel returns it.
        self.exported: torch.Tensor = lse.detach()
        self.batch: Batch | None = None
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        world = get_tensor_model_parallel_world_size()
        if world > 1:
            self.ranks = Ranks("tp", world, get_tensor_model_parallel_rank(), tensor_model_parallel_all_reduce)

    def local(self, axis: str) -> int:
        return self.q.shape[1]

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        self.batch = batch
        return super().read(site, spec, batch)

    def _request(self) -> tuple[int, int, int]:
        """The requesting worker's request index, first query row, and row count in this call."""
        if self.batch is None:
            if len(self.used) != 1:
                raise ValueError(f"{NAME} ran {len(self.used)} requests in one call and the trace has no batcher to tell them apart")
            return 0, 0, self.spans[1]
        start, size = self.batch.start, self.batch.size
        if start not in self.spans[:-1] or self.spans[self.spans.index(start) + 1] != start + size:
            raise ValueError(
                f"kernel sites: the worker's rows [{start}, {start + size}) are not one request's query rows in this call (request boundaries {self.spans})"
            )
        return self.spans.index(start), start, size

    def _axes(self, spec: dict[str, Any]) -> tuple:
        index, start, size = self._request()
        device = self.q.device
        keys = self.used[index]
        heads = sites.positions(sites.indices(spec.get("heads"), self.q.shape[1], device), self.q.shape[1], device)
        rows = sites.positions(sites.indices(spec.get("query"), size, device), size, device)
        cols = sites.positions(sites.indices(spec.get("key"), keys, device), keys, device)
        return index, start, size, keys, heads, rows, cols

    def _lse(self, start: int, heads: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return self.exported.index_select(0, heads).index_select(1, rows + start).T.float()

    def _scores(self, index: int, start: int, size: int, keys: int, heads: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        q = self.q.index_select(0, rows + start).index_select(1, heads).float()  # [rows, heads, dim]
        # Key ``c`` of the request lives in page ``block_table[index, c // page]`` at offset ``c % page``.
        page = self.cache.shape[1]
        pages = self.block_table[index].index_select(0, cols // page).long()
        kv_heads = heads // (self.q.shape[1] // self.cache.shape[2])
        k = self.cache[pages, cols % page].index_select(1, kv_heads).float()  # [cols, heads, dim]
        with _exact_fp32():
            scores = torch.einsum("rhd,chd->rhc", q, k) * self.scale
        if self.causal:
            # The call's query rows are the request's last ``size`` tokens: row ``r`` is token ``keys - size + r``.
            hidden = cols[None, :] > (rows[:, None] + (keys - size))
            scores = scores.masked_fill(hidden[:, None, :], float("-inf"))
        return scores

    def lse(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        _, start, _, _, heads, rows, _ = self._axes(spec)
        provenance = Provenance("exact export", NAME, "the kernel's own per-row logsumexp, fp32")
        return sites.squeeze(self._lse(start, heads, rows), spec, {"query": 0, "heads": 1}), provenance

    def scores(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        index, start, size, keys, heads, rows, cols = self._axes(spec)
        sites.warn_if_large((len(rows), len(heads), len(cols)), "scores", "query=, heads= or key=")
        scores = self._scores(index, start, size, keys, heads, rows, cols)
        provenance = Provenance("exact reconstruction", NAME, "q . k * scale in fp32 from the kernel's query rows and the request's cached keys")
        return sites.squeeze(scores, spec, {"query": 0, "heads": 1, "key": 2}), provenance

    def probs(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        index, start, size, keys, heads, rows, cols = self._axes(spec)
        sites.warn_if_large((len(rows), len(heads), len(cols)), "probs", "query=, heads= or key=")
        probs = torch.exp(self._scores(index, start, size, keys, heads, rows, cols) - self._lse(start, heads, rows).unsqueeze(-1))
        provenance = Provenance("exact reconstruction", NAME, "exp(scores - lse) in fp32, the kernel backward's own formula")
        return sites.squeeze(probs, spec, {"query": 0, "heads": 1, "key": 2}), provenance


def flash_attention(context: None) -> Located:
    """Locate ``flash_attn_varlen_func`` as the FlashAttention backend's module global.

    The backend imports the function only when vLLM's flash-attn build is
    available, so an absent global means there is no kernel to serve.
    """
    from vllm.v1.attention.backends import flash_attn

    fn = vars(flash_attn).get(NAME)
    if fn is not None:
        yield unshimmed(fn), (Attr(flash_attn, NAME),)


#: Every kernel served on a vLLM model.
CATALOG: list[Spec] = [
    Spec(NAME, "attention", flash_attention, serve),
]


def register_kernels() -> None:
    """Register `CATALOG`; called in the worker process, which is where the kernel runs."""
    register_catalog(CATALOG)
