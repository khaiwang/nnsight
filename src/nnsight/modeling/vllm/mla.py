"""Attention kernel sites on vLLM's multi-head latent attention (MLA; DeepSeek-V2/V3 and related models).

An MLA layer (``MLAAttention``) stores one compressed key-value row per token
and runs each step's tokens through one of two kernels inside ``forward_impl``:

* **Prefill**: the keys and values of the request's new tokens are expanded per
  head, and ``flash_attn_varlen_func`` attends over them with a causal mask. The
  kernel computes each row's logsumexp and vLLM does not ask for it; the
  recorder asks for it in the same launch, which leaves the output unchanged.
* **Decode**: the query is projected into the compressed space and the Triton
  decode kernel attends over the request's compressed rows in the paged cache.
  It returns its logsumexp in the query's dtype.

`serve` is registered on ``MLAAttention.forward_impl`` and records both calls.
The `Latent` reader serves the sites of the paged FlashAttention reader
([`Paged`][nnsight.modeling.vllm.kernel.Paged]) with the same layout: no batch
axis, ``[query, heads, key]``.

* ``scores``: ``q . k * scale`` in fp32 from the kernel's own query and key rows.
* ``lse``: the kernel's logsumexp.
* ``probs``: on a prefill step ``exp(scores - lse)``, the kernel's own formula.
  On a decode step the kernel's logsumexp has the precision of the query dtype
  (bf16), so the probabilities are the fp32 softmax of the scores.

Not served: a prefill that continues cached or earlier-chunk context, which vLLM
computes as several kernel calls merged by logsumexp; the other MLA backends
(FlashMLA, FlashInfer, CUTLASS, sparse); and a quantized cache.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch

from ...intervention import sites
from ...intervention.kernel import Attr, Located, Spec, unshimmed
from ...intervention.sites import SPLIT, Batch, Provenance, Sites, Unavailable
from ...intervention.source import arguments

NAME = "MLAAttention.forward_impl"


class _Capture(threading.local):
    """Thread-local list collecting one record per kernel call during `serve` (``None`` outside a served call)."""

    calls: list | None = None


_capture = _Capture()
_installed = False
_lock = threading.Lock()


def _install() -> None:
    """Wrap the prefill and decode kernel calls once so that, inside a served call, each records what the reader needs."""
    global _installed
    with _lock:
        if _installed:
            return
        from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend
        from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

        prefill, decode = FlashAttnPrefillBackend.run_prefill_new_tokens, TritonMLAImpl.forward_mqa

        def recording_prefill(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            if calls is None:
                return prefill(*args, **kwargs)
            bound = arguments(prefill, args, kwargs)
            asked = bound["return_softmax_lse"]
            output, lse = prefill(**{**bound, "return_softmax_lse": True})
            backend = bound["self"]
            calls.append({"kind": "prefill", "q": bound["q"], "k": bound["k"], "lse": lse, "starts": backend._prefill_metadata.query_start_loc.tolist(), "scale": backend.scale})
            return (output, lse) if asked else output

        def recording_decode(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            output, lse = decode(*args, **kwargs)
            if calls is not None:
                bound = arguments(decode, args, kwargs)
                impl, q, metadata = bound["self"], bound["q"], bound["attn_metadata"].decode
                calls.append({
                    "kind": "decode",
                    "q": torch.cat(q, dim=-1) if type(q) is tuple else q,
                    "cache": bound["kv_c_and_k_pe_cache"],
                    "block_table": metadata.block_table,
                    "lengths": metadata.seq_lens.tolist(),
                    "lse": lse,
                    "scale": impl.scale,
                })
            return output, lse

        FlashAttnPrefillBackend.run_prefill_new_tokens = recording_prefill
        TritonMLAImpl.forward_mqa = recording_decode
        _installed = True


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run ``forward_impl`` with its kernel calls recording, and hand back a `Latent` reader over them."""
    from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache

    _install()
    outer, _capture.calls = _capture.calls, (calls := [])
    try:
        output = fn(*args, **kwargs)
    finally:
        _capture.calls = outer
    bound = arguments(fn, args, kwargs)
    layer, metadata = bound["self"], bound["attn_metadata"]
    if metadata is None:
        return output, Unavailable(f"{NAME} ran without attention metadata; its kernel sites are not served on this call")
    if type(layer.impl) is not TritonMLAImpl or not isinstance(layer.prefill_backend, FlashAttnPrefillBackend):
        return output, Unavailable(
            f"{NAME}: kernel sites cover the Triton MLA decode kernel with the FlashAttention prefill kernel; "
            f"this layer runs {type(layer.impl).__name__} with {type(layer.prefill_backend).__name__}"
        )
    if is_quantized_kv_cache(layer.kv_cache_dtype):
        return output, Unavailable(f"{NAME}: kernel sites are not served on a quantized cache ({layer.kv_cache_dtype})")
    if metadata.prefill is not None and metadata.prefill.chunked_context is not None:
        return output, Unavailable(f"{NAME}: this prefill continues cached context, which vLLM computes as several kernel calls merged into one attention; its kernel sites are not served")
    return output, Latent(calls, metadata.num_decode_tokens)


class Latent(Sites):
    """Site reader for one ``MLAAttention.forward_impl`` call: the step's decoding requests first, then its prefilling ones."""

    names = ("lse", "scores", "probs")
    axes = {"lse": ("query", "heads"), "scores": ("query", "heads", "key"), "probs": ("query", "heads", "key")}
    #: vLLM shards the query projection by head, as on the other attention layers.
    layout = {"tp": {"heads": SPLIT}}
    narrowed = True

    def __init__(self, calls: list[dict[str, Any]], decoding: int) -> None:
        self.calls = {call["kind"]: call for call in calls}
        #: Token rows of the step's decoding requests, one each; the prefilling requests' rows follow.
        self.decoding = decoding
        self.batch: Batch | None = None

    def local(self, axis: str) -> int:
        return next(iter(self.calls.values()))["q"].shape[1]

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        self.batch = batch
        return super().read(site, spec, batch)

    def _request(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """The requesting worker's query rows ``[n, heads, dim]``, key rows, logsumexp ``[n, heads]`` and whether the rows are a prefill."""
        if self.batch is None:
            if len(self.calls) != 1:
                raise ValueError(f"{NAME} ran prefilling and decoding requests in one call and the trace has no batcher to tell them apart")
            start = 0
            size = None
        else:
            start, size = self.batch.start, self.batch.size
        if start < self.decoding or (self.batch is None and "decode" in self.calls):
            call = self.calls["decode"]
            if size not in (None, 1) or (size is None and len(call["lengths"]) != 1):
                raise ValueError(f"kernel sites: the worker's rows [{start}, {start + (size or 0)}) are not one decoding request's row in this call")
            cache, table = call["cache"], call["block_table"][start]
            length = call["lengths"][start]
            # The request's compressed rows, in order, from the pages its block table names.
            pages = table[: -(-length // cache.shape[1])].long()
            keys = cache[pages].flatten(0, 1)[:length]
            return call["q"][start : start + 1], keys, call["lse"][start : start + 1], False
        call = self.calls["prefill"]
        starts, first = call["starts"], start - self.decoding
        if size is None:
            if len(starts) != 2:
                raise ValueError(f"{NAME} prefilled {len(starts) - 1} requests in one call and the trace has no batcher to tell them apart")
            size = starts[1]
        if first not in starts[:-1] or starts[starts.index(first) + 1] != first + size:
            raise ValueError(f"kernel sites: the worker's rows [{start}, {start + size}) are not one request's query rows in this call (prefill boundaries {starts}, after {self.decoding} decoding rows)")
        rows = slice(first, first + size)
        # FlashAttention returns its logsumexp as [heads, rows].
        return call["q"][rows], call["k"][rows], call["lse"][:, rows].T, True

    def _scale(self, prefill: bool) -> float:
        return self.calls["prefill" if prefill else "decode"]["scale"]

    def _picked(self, spec: dict[str, Any], q: torch.Tensor, keys: torch.Tensor) -> tuple:
        device = q.device
        return (
            sites.indices(spec.get("query"), q.shape[0], device),
            sites.indices(spec.get("heads"), q.shape[1], device),
            sites.indices(spec.get("key"), keys.shape[0], device),
        )

    def _scores(self, q: torch.Tensor, keys: torch.Tensor, prefill: bool, rows: Any, heads: Any, cols: Any) -> torch.Tensor:
        """``q . k * scale`` in fp32 for the selected rows, heads and keys; a prefill masks the keys after each query."""
        q = sites.select(sites.select(q, 0, rows), 1, heads).float()
        keys = sites.select(keys, 0, cols).float()
        if not prefill:
            return torch.einsum("qhd,kd->qhk", q, keys) * self._scale(prefill)
        scores = torch.einsum("qhd,khd->qhk", q, sites.select(keys, 1, heads)) * self._scale(prefill)
        size, device = self.calls["prefill"]["q"].shape[0], q.device
        later = sites.positions(cols, size, device)[: keys.shape[0]].unsqueeze(0) > sites.positions(rows, size, device)[: q.shape[0]].unsqueeze(1)
        return scores.masked_fill(later.unsqueeze(1), float("-inf"))

    def lse(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        q, keys, lse, prefill = self._request()
        rows, heads, _ = self._picked(spec, q, keys)
        value = sites.select(sites.select(lse, 0, rows), 1, heads).float()
        return sites.squeeze(value, spec, {"query": 0, "heads": 1}), Provenance("exact export", NAME, "the kernel's logsumexp" + ("" if prefill else ", in the query dtype"))

    def scores(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        q, keys, _, prefill = self._request()
        rows, heads, cols = self._picked(spec, q, keys)
        sites.warn_if_large((q.shape[0], q.shape[1], keys.shape[0]), "scores", "query=, heads= or key=")
        value = self._scores(q, keys, prefill, rows, heads, cols)
        return sites.squeeze(value, spec, {"query": 0, "heads": 1, "key": 2}), Provenance("exact reconstruction", NAME, "q . k * scale in fp32 from the kernel's query and key rows")

    def probs(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        q, keys, lse, prefill = self._request()
        rows, heads, cols = self._picked(spec, q, keys)
        sites.warn_if_large((q.shape[0], q.shape[1], keys.shape[0]), "probs", "query=, heads= or key=")
        if prefill:
            lse = sites.select(sites.select(lse, 0, rows), 1, heads).float()
            value = torch.exp(self._scores(q, keys, prefill, rows, heads, cols) - lse.unsqueeze(-1))
            provenance = Provenance("exact reconstruction", NAME, "exp(scores - lse) in fp32, the kernel backward's own formula")
        else:
            value = sites.select(torch.softmax(self._scores(q, keys, prefill, rows, heads, slice(None)), -1), 2, cols)
            provenance = Provenance("replay", NAME, "softmax of the fp32 scores over the request's keys; the kernel's logsumexp is in the query dtype")
        return sites.squeeze(value, spec, {"query": 0, "heads": 1, "key": 2}), provenance


def latent_attention(context: None) -> Located:
    """Locate ``MLAAttention.forward_impl``, which the layer's custom op reads off the layer on every call."""
    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    except ImportError:  # a vLLM without this layer
        return
    yield unshimmed(MLAAttention.forward_impl), (Attr(MLAAttention, "forward_impl"),)


CATALOG: list[Spec] = [
    Spec(NAME, "attention", latent_attention, serve),
]
