"""Recurrent kernel sites on vLLM's delta rule layers (Qwen3-Next, Qwen3.5; OLMo hybrid; Kimi Linear).

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

Under tensor parallelism each rank's layer holds ``num_v_heads // tp_size``
value heads, in rank order, and a state of that many heads. The reader declares
``heads`` as `SPLIT`, so `Sites.read` assembles the ranks' blocks and ``heads=``
indexes the model's heads.

The OLMo hybrid layer (``OlmoHybridGatedDeltaNetAttention._forward_core``) and
the Kimi delta attention layer (``KimiGatedDeltaNetAttention._forward``) are
served the same way. Both run every request of a step that prefills through
their chunk kernel (``chunk_gated_delta_rule``, ``chunk_kda_with_fused_gate``),
and a decode-only step through a one-token kernel. Kimi delta attention decays
each key channel of the state separately; the chunk kernel's gate is recorded
as it leaves ``fused_kda_gate_chunk_cumsum``.

Not served: a prefill on another backend (FlashInfer, CuteDSL, AITER, SYCL),
whose kernels keep the chunk states to themselves, and speculative decoding,
whose draft tokens advance a state the layer later discards.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable

import torch

from ...intervention import sites
from ...intervention.kernel import Attr, Located, Spec, unshimmed
from ...intervention.sites import SPLIT, Batch, Provenance, Sites, Unavailable
from ..kernels.recurrent import Recurrence, _gated_delta_step, _l2norm, _walk, recurrence

NAME = "QwenGatedDeltaNetAttention._forward_core"
OLMO = "OlmoHybridGatedDeltaNetAttention._forward_core"
KIMI = "KimiGatedDeltaNetAttention._forward"
#: The two kernels that advance a request's cache slot in place by one token.
_DECODE = ("fused_recurrent_gated_delta_rule_packed_decode", "fused_sigmoid_gating_delta_rule_update")
_KDA = "chunk_kda_with_fused_gate"


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
        from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as olmo
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as layer
        from vllm.third_party.flash_linear_attention.ops import chunk

        # ponytail: the NVIDIA copy of Kimi's kernels only; the layer imports the AMD copy on ROCm.
        from vllm.models.kimi_k3.nvidia.ops.third_party import kda
        from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk as kda_chunk

        def recording_prefill(inner: Callable, name: str, overwritten: tuple[str, ...] = ()) -> Callable:
            """Record a chunk kernel's arguments, copying those in ``overwritten``, which the kernel writes its output into."""

            def prefill(*args: Any, **kwargs: Any) -> Any:
                calls = _capture.calls
                if calls is None or args:
                    return inner(*args, **kwargs)
                calls.append(call := {"kind": "chunk", "name": name, **kwargs, **{key: kwargs[key].clone() for key in overwritten}})
                output = inner(**kwargs)
                call["closed"] = True
                return output

            return prefill

        def recording(inner: Callable, names: dict[int, str]) -> Callable:
            """Keep results ``names`` of a function the open chunk call runs underneath it."""

            def inside(*args: Any, **kwargs: Any) -> Any:
                result = inner(*args, **kwargs)
                calls = _capture.calls
                if calls and calls[-1]["kind"] == "chunk" and "closed" not in calls[-1]:
                    calls[-1].update({name: result[index] for index, name in names.items()})
                return result

            return inside

        def recording_decode(module: Any, name: str) -> Callable:
            inner = getattr(module, name)

            def decode(*args: Any, **kwargs: Any) -> Any:
                calls = _capture.calls
                if calls is None or args:
                    return inner(*args, **kwargs)
                slots = kwargs["state_indices"] if "state_indices" in kwargs else kwargs["ssm_state_indices"]
                cache = kwargs["initial_state"]
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

        states = {0: "h", 2: "final"}
        layer.fla_chunk_gated_delta_rule = recording_prefill(layer.fla_chunk_gated_delta_rule, "chunk_gated_delta_rule")
        olmo.chunk_gated_delta_rule = recording_prefill(olmo.chunk_gated_delta_rule, "chunk_gated_delta_rule")
        chunk.chunk_gated_delta_rule_fwd_h = recording(chunk.chunk_gated_delta_rule_fwd_h, states)
        for name in _DECODE:
            setattr(layer, name, recording_decode(layer, name))
        olmo.fused_recurrent_gated_delta_rule = recording_decode(olmo, "fused_recurrent_gated_delta_rule")
        # The Kimi layer imports its kernels from the package on every call.
        kda.chunk_kda_with_fused_gate = recording_prefill(kda.chunk_kda_with_fused_gate, _KDA, ("v",))
        kda_chunk.chunk_gated_delta_rule_fwd_h = recording(kda_chunk.chunk_gated_delta_rule_fwd_h, states)
        kda_chunk.fused_kda_gate_chunk_cumsum = recording(kda_chunk.fused_kda_gate_chunk_cumsum, {0: "cumulative", 1: "gate"})
        kda.fused_recurrent_kda_packed_decode = recording_decode(kda, "fused_recurrent_kda_packed_decode")
        _installed = True


def _kda_step(key: torch.Tensor, value: torch.Tensor, cumulative: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> Callable:
    """An fp32 per-token step of Kimi delta attention over one request's ``[1, tokens, heads, ...]`` inputs.

    ``S_t = S' + k_t (beta_t (v_t - k_t^T S'))^T`` with ``S' = exp(g_t) S_{t-1}``
    per key channel. ``cumulative`` is the kernel's gate: base-2 logs summed
    from the start of each chunk, so a token's own gate is its difference from
    the token before it in the chunk.
    """
    key = _l2norm(key.transpose(1, 2).float())
    value, beta = value.transpose(1, 2).float(), beta.transpose(1, 2).float()
    log = cumulative.float() * math.log(2)
    previous = torch.cat([torch.zeros_like(log[:, :1]), log[:, :-1]], 1)
    previous[:, ::chunk_size] = 0
    decay = (log - previous).transpose(1, 2).exp()

    def step(state: torch.Tensor, t: int) -> torch.Tensor:
        k_t = key[:, :, t]
        state = state * decay[:, :, t].unsqueeze(-1)
        delta = (value[:, :, t] - (state * k_t.unsqueeze(-1)).sum(-2)) * beta[:, :, t, None]
        return state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

    return step


def serve(fn: Callable, args: tuple, kwargs: dict, every: bool = False) -> tuple[Any, Sites]:
    """Run a layer's core with its kernels recording, and hand back a `Scan` over what they did.

    ``every``: the layer runs every request of a prefilling step through its chunk kernel.
    """
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
        return output, Unavailable(f"{fn.__qualname__} ran without attention metadata; its kernel sites are not served on this call")
    if metadata.spec_sequence_masks is not None:
        return output, Unavailable(f"{fn.__qualname__}: kernel sites are not served under speculative decoding")
    return output, Scan(calls, metadata, layer.get_state_shape()[1][0], every)


def serve_every(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    return serve(fn, args, kwargs, True)


class Scan(Sites):
    """Site reader for one ``_forward_core`` call: every request of the step, each served from the kernel it went through."""

    names = Recurrence.names
    axes = Recurrence.axes
    #: Each rank's layer holds one block of the value heads, and so does its state.
    layout = {"tp": {"heads": SPLIT}}
    narrowed = True

    def __init__(self, calls: list[dict[str, Any]], metadata: Any, heads: int, every: bool) -> None:
        self.calls = calls
        #: Value heads this rank's layer holds.
        self.heads = heads
        self.batch: Batch | None = None
        #: First token row of each request in the step's flat buffer, and one past the last request.
        self.starts: list[int] = metadata.non_spec_query_start_loc.tolist()
        count = len(self.starts) - 1
        #: Each request's slot in the layer's state cache, which is how a kernel call names it.
        self.slots: list[int] = metadata.non_spec_state_indices_tensor[:count].tolist()
        prefill = metadata.prefill_state_indices
        #: Slots of the requests a chunk kernel call covers, in the order of its token strip.
        self.prefill_slots: list[int] = self.slots if every else [] if prefill is None else prefill.tolist()

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
                bounds = call["cu_seqlens"].tolist()
                tokens = slice(bounds[index], bounds[index + 1])
                # Each request's chunks start at its first token.
                counts = [-(-(stop - start) // FLA_CHUNK_SIZE) for start, stop in zip(bounds, bounds[1:])]
                first = sum(counts[:index])
                if call["name"] == _KDA:
                    step = _kda_step(*(call[name][:, tokens] for name in ("k", "v", "cumulative", "gate")), FLA_CHUNK_SIZE)
                else:
                    # The Qwen layer normalizes the keys before the call; the OLMo layer has the kernel do it.
                    step = _gated_delta_step(*(call[name][:, tokens] for name in ("k", "v", "g", "beta")), call.get("use_qk_l2norm_in_kernel", False))
                return recurrence(
                    call["h"][:, first : first + counts[index]].transpose(-1, -2),
                    call["final"][index : index + 1].transpose(-1, -2),
                    FLA_CHUNK_SIZE,
                    tokens.stop - tokens.start,
                    _walk(step, FLA_CHUNK_SIZE),
                    call["name"],
                )
        ran = sorted({call.get("name", call["kind"]) for call in self.calls}) or ["no recorded kernel"]
        return Unavailable(
            f"{NAME}: this request's state was advanced by a kernel that exports no states on this call (ran: {', '.join(ran)}); "
            "kernel sites cover the Triton/FLA prefill kernel and the one-token decode kernels"
        )

    def local(self, axis: str) -> int:
        return self.heads

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        self.batch = batch
        return super().read(site, spec, batch)

    def _site(self, site: str, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        value, provenance = self._reader(self.slots[self._request(self.batch)]).read(site, spec, None)
        return value.squeeze(0), provenance

    def chunk_states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        return self._site("chunk_states", spec)

    def states(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        return self._site("states", spec)


def forward_core(context: None) -> Located:
    """Locate ``QwenGatedDeltaNetAttention._forward_core``, which the layer's custom op reads off the class on every call."""
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    except ImportError:  # a vLLM without this layer
        return
    yield unshimmed(QwenGatedDeltaNetAttention._forward_core), (Attr(QwenGatedDeltaNetAttention, "_forward_core"),)


def olmo_forward_core(context: None) -> Located:
    try:
        from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import OlmoHybridGatedDeltaNetAttention as Layer
    except ImportError:  # a vLLM without this layer
        return
    yield unshimmed(Layer._forward_core), (Attr(Layer, "_forward_core"),)


def kimi_forward(context: None) -> Located:
    try:
        from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention as Layer
    except ImportError:  # a vLLM without this layer
        return
    # ``_forward`` is vLLM's eager-break wrapper. A replayed step calls the function the wrapper closes over,
    # so that function is the kernel and its closure cell is where the layer looks it up.
    inner = Layer._forward.__wrapped__
    cell = next(cell for cell in Layer._forward.__closure__ if unshimmed(cell.cell_contents) is inner)
    yield inner, (Attr(cell, "cell_contents"),)


CATALOG: list[Spec] = [
    Spec(NAME, "recurrent", forward_core, serve),
    Spec(OLMO, "recurrent", olmo_forward_core, serve_every),
    Spec(KIMI, "recurrent", kimi_forward, serve_every),
]
