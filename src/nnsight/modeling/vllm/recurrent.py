"""Recurrent kernel sites on vLLM's gated delta rule layers (Qwen3-Next, Qwen3.5).

vLLM's ``QwenGatedDeltaNetAttention`` runs its recurrence in ``_forward_core``,
once per step for every scheduled request, and keeps each request's state in a
slot of its state cache. Inside, a request goes through one of three kernels:

* **Prefill** (Triton/FLA backend): ``chunk_gated_delta_rule`` from vLLM's copy
  of flash-linear-attention, over the prefill requests packed into one strip of
  tokens. Its ``chunk_gated_delta_rule_fwd_h`` returns the state entering every
  chunk and each request's final state.
* **Decode**: ``fused_recurrent_gated_delta_rule_packed_decode``, or
  ``fused_sigmoid_gating_delta_rule_update`` for the decoding requests of a step
  that also prefills. Both advance the cache slot in place by one token.

A step can run two of them, so `serve` is registered on ``_forward_core`` and
records what the kernels underneath received and returned. The `Scan` reader
finds the requesting worker's request by its cache slot and serves it from the
call it went through, with the sites of
[`Recurrence`][nnsight.modeling.kernels.recurrent.Recurrence] and vLLM's layout:
no batch axis.

vLLM's kernels and its state cache hold a state as ``[heads, V, K]``; the sites
serve ``[heads, K, V]``, the layout of the transformers sites.

* ``chunk_states``: ``[chunks, heads, K, V]``. On a prefill step, the state
  entering each chunk of the request's tokens in this step and its final state.
  On a decode step, the state before the token and the state after it.
* ``states``: ``[tokens, heads, K, V]``, the state after each token of the step.

Not served: a prefill on another backend (FlashInfer, CuteDSL, AITER, SYCL),
whose kernels keep the chunk states to themselves; speculative decoding; and
tensor parallelism, where the layout of the heads across ranks is not surveyed.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch

from ...intervention import sites
from ...intervention.kernel import Attr, Located, Spec, unshimmed
from ...intervention.sites import Batch, Provenance, Sites, Unavailable
from ..kernels.recurrent import Recurrence, _gated_delta_step, _walk, recurrence

NAME = "QwenGatedDeltaNetAttention._forward_core"
#: The two kernels that advance a request's cache slot in place by one token.
_DECODE = ("fused_recurrent_gated_delta_rule_packed_decode", "fused_sigmoid_gating_delta_rule_update")


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
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as layer
        from vllm.third_party.flash_linear_attention.ops import chunk

        prefill, states = layer.fla_chunk_gated_delta_rule, chunk.chunk_gated_delta_rule_fwd_h

        def recording_prefill(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            if calls is None or args:
                return prefill(*args, **kwargs)
            calls.append(call := {"kind": "chunk", **kwargs})
            output = prefill(**kwargs)
            call["closed"] = True
            return output

        def recording_states(*args: Any, **kwargs: Any) -> Any:
            result = states(*args, **kwargs)
            calls = _capture.calls
            if calls and calls[-1]["kind"] == "chunk" and "closed" not in calls[-1]:
                calls[-1]["h"], calls[-1]["final"] = result[0], result[2]
            return result

        def recording_decode(name: str) -> Callable:
            inner = getattr(layer, name)

            def decode(*args: Any, **kwargs: Any) -> Any:
                calls = _capture.calls
                if calls is None or args:
                    return inner(*args, **kwargs)
                slots, cache = kwargs["ssm_state_indices"], kwargs["initial_state"]
                lengths = kwargs.get("cu_seqlens")
                count = kwargs["mixed_qkv"].shape[0] if lengths is None else len(lengths) - 1
                # One token per request and one slot per request: the non-speculative layout.
                if slots.dim() != 1 or (lengths is not None and int(lengths[-1]) != count):
                    calls.append({"kind": "unsupported", "name": name})
                    return inner(**kwargs)
                slots = slots[:count].long()
                before = cache[slots].clone()
                output = inner(**kwargs)
                calls.append({"kind": "decode", "name": name, "slots": slots.tolist(), "before": before, "after": cache[slots].clone()})
                return output

            return decode

        layer.fla_chunk_gated_delta_rule = recording_prefill
        chunk.chunk_gated_delta_rule_fwd_h = recording_states
        for name in _DECODE:
            setattr(layer, name, recording_decode(name))
        _installed = True


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run ``_forward_core`` with its kernels recording, and hand back a `Scan` over what they did."""
    from vllm.forward_context import get_forward_context

    from .fragments import _tp_world_size

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
    if _tp_world_size() > 1:
        return output, Unavailable(f"{NAME}: kernel sites are not served under tensor parallelism")
    if metadata.spec_sequence_masks is not None:
        return output, Unavailable(f"{NAME}: kernel sites are not served under speculative decoding")
    return output, Scan(calls, metadata)


class Scan(Sites):
    """Site reader for one ``_forward_core`` call: every request of the step, each served from the kernel it went through."""

    names = Recurrence.names
    axes = Recurrence.axes
    narrowed = True

    def __init__(self, calls: list[dict[str, Any]], metadata: Any) -> None:
        self.calls = calls
        #: First token row of each request in the step's flat buffer, and one past the last request.
        self.starts: list[int] = metadata.non_spec_query_start_loc.tolist()
        count = len(self.starts) - 1
        #: Each request's slot in the layer's state cache, which is how a kernel call names it.
        self.slots: list[int] = metadata.non_spec_state_indices_tensor[:count].tolist()
        prefill = metadata.prefill_state_indices
        self.prefill_slots: list[int] = [] if prefill is None else prefill.tolist()

    def _request(self, batch: Batch | None) -> int:
        """The requesting worker's request index in this step."""
        if batch is None:
            if len(self.slots) != 1:
                raise ValueError(f"{NAME} ran {len(self.slots)} requests in one call and the trace has no batcher to tell them apart")
            return 0
        start, size = batch.start, batch.size
        if start not in self.starts[:-1] or self.starts[self.starts.index(start) + 1] != start + size:
            raise ValueError(f"kernel sites: the worker's rows [{start}, {start + size}) are not one request's rows in this call (request boundaries {self.starts})")
        return self.starts.index(start)

    def _reader(self, slot: int) -> Sites:
        """A one-request reader for the kernel call that advanced cache slot ``slot``."""
        from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

        for call in self.calls:
            if call["kind"] == "decode" and slot in call["slots"]:
                index = call["slots"].index(slot)
                states = torch.stack([call["before"][index : index + 1], call["after"][index : index + 1]], 1)
                return Recurrence(states.transpose(-1, -2), None, 1, 1, None, call["name"])
            if call["kind"] == "chunk" and slot in self.prefill_slots and "h" in call:
                index = self.prefill_slots.index(slot)
                bounds, chunks = call["cu_seqlens"].tolist(), call["chunk_offsets"].tolist()
                tokens = slice(bounds[index], bounds[index + 1])
                # The layer normalizes the keys and applies the gate before the call, so the step takes them as given.
                step = _gated_delta_step(*(call[name][:, tokens] for name in ("k", "v", "g", "beta")), False)
                return recurrence(
                    call["h"][:, chunks[index] : chunks[index + 1]].transpose(-1, -2),
                    call["final"][index : index + 1].transpose(-1, -2),
                    FLA_CHUNK_SIZE,
                    tokens.stop - tokens.start,
                    _walk(step, FLA_CHUNK_SIZE),
                    "chunk_gated_delta_rule",
                )
        ran = sorted({call.get("name", call["kind"]) for call in self.calls}) or ["no recorded kernel"]
        return Unavailable(
            f"{NAME}: this request's state was advanced by a kernel that exports no states on this call (ran: {', '.join(ran)}); "
            "kernel sites cover the Triton/FLA prefill kernel and the one-token decode kernels"
        )

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        if site not in self.names:
            raise ValueError(f"this kernel has no site {site!r}; available: {', '.join(self.names)}")
        sites.check_axes(spec, site, self.axes[site])
        value, provenance = self._reader(self.slots[self._request(batch)]).read(site, spec, None)
        return value.squeeze(0), provenance


def forward_core(context: None) -> Located:
    """Locate ``QwenGatedDeltaNetAttention._forward_core``, which the layer's custom op reads off the class on every call."""
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    except ImportError:  # a vLLM without this layer
        return
    yield unshimmed(QwenGatedDeltaNetAttention._forward_core), (Attr(QwenGatedDeltaNetAttention, "_forward_core"),)


CATALOG: list[Spec] = [
    Spec(NAME, "recurrent", forward_core, serve),
]
