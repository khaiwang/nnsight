"""Attention probabilities, scores, and logsumexp for transformers models running FlashAttention 2.

Under ``attn_implementation="flash_attention_2"`` every attention module calls
``_flash_attention_forward`` (transformers/modeling_flash_attention_utils.py)
with queries and keys laid out as ``[batch, length, heads, dim]`` plus the
padding mask. For a padded batch that function removes the padding, joins the
rows into one packed strip of tokens, calls the FlashAttention kernel on the
strip, and pads the result back.

Two locations matter. The kernel call is the only place the kernel's logsumexp
exists. ``_flash_attention_forward`` is the last place the batch rows and the
padding mask exist. `serve_flash` is therefore registered on
``_flash_attention_forward`` and reads the rows there, and it takes two things
from the kernel call made underneath: the logsumexp the kernel returns under
``return_attn_probs=True``, and the keyword arguments the kernel received
(causal flag, scale, packed sequence boundaries, and any option that changes
the attention). The `Flash` reader places the packed logsumexp back at the real
token positions, so the three sites have the ``[batch, heads, queries, keys]``
layout of the SDPA sites and batch rows split between invokes as usual.

``_flash_attention_forward`` fetches the kernel functions through
``lazy_import_flash_attention`` on every call, resolved as a module global.
`_install_recorder` wraps that global once per process. Outside a served call
the wrapper returns transformers' functions unchanged.
"""

from __future__ import annotations

import functools
import inspect
import sys
import threading
from types import ModuleType
from typing import Any, Callable

import torch

from ...intervention import sites
from ...intervention.attention import Attention
from ...intervention.sites import Sites, Unavailable
from ...intervention.source import arguments

#: Kernel keyword arguments the reader reads.
_READ = frozenset({"softmax_scale", "causal", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"})
#: Kernel keyword arguments with the values under which the kernel computes ``softmax(q . k * scale + causal mask)``.
_NEUTRAL = {"dropout_p": (0, 0.0), "window_size": ((-1, -1),), "softcap": (0, 0.0), "deterministic": (True, False)}


class _Capture(threading.local):
    """Thread-local state whose ``results`` list collects ``(logsumexp, keyword arguments)`` of FlashAttention 2 kernel calls during `serve_flash` (``None`` outside a served call)."""

    results: list | None = None


_capture = _Capture()
_installed: set[str] = set()
_lock = threading.Lock()


@functools.lru_cache(maxsize=None)
def _returns_lse(fn: Callable) -> bool:
    """Return whether ``fn`` accepts ``return_attn_probs``, the flag under which FlashAttention 2 returns its logsumexp."""
    return "return_attn_probs" in inspect.signature(fn).parameters


def _recording(fn: Callable | None, captured: list) -> Callable | None:
    """Wrap a FlashAttention 2 function so it records its logsumexp and keyword arguments in ``captured`` and returns the attention output alone."""
    if fn is None or not _returns_lse(fn):
        return fn

    def call(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **{**kwargs, "return_attn_probs": True})
        if not (isinstance(result, tuple) and len(result) == 3):
            return result
        captured.append((result[1].detach(), kwargs))
        return result[0]

    return call


def _install_recorder(utils: ModuleType) -> None:
    """Wrap ``utils.lazy_import_flash_attention`` once so that, inside a served call running FlashAttention 2, it returns recording kernel functions."""
    with _lock:
        if utils.__name__ in _installed:
            return
        inner = utils.lazy_import_flash_attention
        fa2 = {"flash_attention_2", utils.FLASH_ATTN_KERNEL_FALLBACK["flash_attention_2"]}

        def loading(*args: Any, **kwargs: Any) -> Any:
            functions, process = inner(*args, **kwargs)
            captured = _capture.results
            if captured is None or utils._loaded_implementation not in fa2:
                return functions, process
            dense, packed, *rest = functions
            return (_recording(dense, captured), _recording(packed, captured), *rest), process

        utils.lazy_import_flash_attention = loading
        _installed.add(utils.__name__)


def serve_flash(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Serve transformers' ``_flash_attention_forward`` from its padded inputs and the logsumexp of the FlashAttention 2 call it makes."""
    utils = sys.modules[fn.__module__]
    _install_recorder(utils)
    outer, _capture.results = _capture.results, (captured := [])
    try:
        output = fn(*args, **kwargs)
    finally:
        _capture.results = outer
    bound = arguments(fn, args, kwargs)
    if len(captured) != 1:
        return output, Unavailable(
            f"kernel sites cover FlashAttention 2's dense and packed calls; this call ran {utils._loaded_implementation!r} "
            f"and made {len(captured)} such calls that return a logsumexp"
        )
    lse, passed = captured[0]
    for name, value in passed.items():
        if name in _READ or value is None or (name in _NEUTRAL and value in _NEUTRAL[name]):
            continue
        return output, Unavailable(f"kernel sites do not cover FlashAttention 2 called with {name}")
    if bound["use_top_left_mask"]:
        return output, Unavailable("kernel sites do not cover FlashAttention 2 called with use_top_left_mask")
    if bound["attention_mask"] is None and "cu_seqlens_q" in passed:
        query, key = bound["query_states"], bound["key_states"]
        if query.shape[0] != 1:
            return output, Unavailable("kernel sites do not cover sequences packed across several batch rows")
        if passed["cu_seqlens_q"][-1] != query.shape[1] or passed["cu_seqlens_k"][-1] != key.shape[1]:
            return output, Unavailable("kernel sites do not cover a packed call whose sequences leave part of the row unused")
    return output, Flash(bound, lse, passed)


def _spans(boundaries: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each of ``length`` packed tokens, return its sequence index, its offset inside that sequence, and that sequence's length."""
    boundaries = boundaries.long()
    tokens = torch.arange(length, device=boundaries.device)
    index = torch.bucketize(tokens, boundaries[1:], right=True)
    return index, tokens - boundaries[index], (boundaries[1:] - boundaries[:-1])[index]


class Flash(Attention):
    """Site reader for a ``_flash_attention_forward`` call that ran FlashAttention 2.

    ``bound`` holds the call's arguments, ``lse`` the kernel's logsumexp, and
    ``passed`` the keyword arguments the kernel received. Three call layouts are
    read:

    * Dense (no padding): the kernel saw ``[batch, length]`` rows and ``lse`` is
      ``[batch, heads, queries]``.
    * Padded batch: transformers packed the real tokens of every row. The reader
      masks padding keys and padding query rows, and places the packed ``lse``
      (``[heads, real query tokens]``) back by the query mask transformers
      unpadded with. Padding query rows have ``lse`` of ``-inf`` and ``probs`` of 0.
    * Several sequences packed into one row (by ``position_ids`` or the caller's
      own boundaries): the reader masks attention across sequences using the
      boundaries the kernel received.

    The key axis covers the keys the kernel received. With a static cache
    transformers trims the keys to the mask length before the kernel call, so
    the axis is shorter than the cache.
    """

    def __init__(self, bound: dict[str, Any], lse: torch.Tensor, passed: dict[str, Any]) -> None:
        query, key, mask = bound["query_states"], bound["key_states"], bound["attention_mask"]
        if bound["target_dtype"] and query.dtype == torch.float32:
            # The cast `fa_peft_integration_check` applies before the kernel call.
            query, key = query.to(bound["target_dtype"]), key.to(bound["target_dtype"])
        rows, queries = query.shape[:2]
        self.packed = "cu_seqlens_q" in passed
        self.causal = passed["causal"]
        self.real_queries: torch.Tensor | None = None
        allowed = None
        if mask is not None:
            key = key[:, : mask.shape[-1]]
            real_keys = mask.bool()
            # The query mask `_upad_input` derives: the key mask, one unmasked token per row on a decode step, or the mask's last columns.
            if queries == key.shape[1]:
                self.unpadded_queries = real_keys
            elif queries == 1:
                self.unpadded_queries = torch.ones(rows, 1, dtype=torch.bool, device=query.device)
            else:
                self.unpadded_queries = real_keys[:, -queries:]
            # A row with no real key has no attention to report, whatever the kernel wrote for it.
            self.real_queries = self.unpadded_queries & real_keys.any(-1, keepdim=True)
            allowed = self.real_queries[:, None, :, None] & real_keys[:, None, None, :]
        elif self.packed:
            query_index, query_offset, query_length = _spans(passed["cu_seqlens_q"].to(query.device), queries)
            key_index, key_offset, key_length = _spans(passed["cu_seqlens_k"].to(query.device), key.shape[1])
            allowed = query_index[:, None] == key_index[None, :]
            if self.causal:
                # The kernel aligns the causal mask bottom right inside each sequence.
                allowed = allowed & (key_offset[None, :] <= query_offset[:, None] + (key_length[None, :] - query_length[:, None]))
                self.causal = False
            allowed = allowed[None, None]
        super().__init__({"query": query.transpose(1, 2), "key": key.transpose(1, 2), "attn_mask": allowed, "scale": passed.get("softmax_scale")}, None)
        self.kernel_lse = lse

    @property
    def op_name(self) -> str:
        return "flash_attn (FlashAttention 2)"

    @functools.cached_property
    def exported(self) -> torch.Tensor | None:
        """Return the kernel's logsumexp as fp32 ``[batch, heads, queries]``, or ``None`` when its layout cannot be placed back."""
        lse = self.kernel_lse.float()
        if not self.packed:
            return lse
        if lse.dim() != 2:
            # flash_attn 2.3.3 to 2.5.9 return [sequences, heads, longest sequence] from the packed call.
            return None
        if self.real_queries is None:
            return lse.unsqueeze(0)
        rows, heads, queries = self.q.shape[:3]
        placed = torch.full((rows, queries, heads), float("-inf"), device=lse.device)
        placed[self.unpadded_queries.expand(rows, queries)] = lse.transpose(0, 1)
        return placed.masked_fill(~self.real_queries.expand(rows, queries)[:, :, None], float("-inf")).permute(0, 2, 1)

    def _bias(self, heads: torch.Tensor, rows: slice | torch.Tensor, cols: slice | torch.Tensor) -> torch.Tensor | None:
        bias = self._mask(heads, rows, cols)
        if self.causal:
            queries, keys = self.q.shape[2], self.k.shape[2]
            row = sites.positions(rows, queries, self.q.device)
            col = sites.positions(cols, keys, self.q.device)
            # FlashAttention aligns the causal mask bottom right: the last query row attends to every key.
            causal = torch.zeros(len(row), len(col), device=self.q.device).masked_fill(col[None, :] > row[:, None] + (keys - queries), float("-inf"))
            bias = causal if bias is None else bias + causal
        return bias
