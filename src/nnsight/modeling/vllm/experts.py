"""Per-(token, expert) rows from vLLM's fused mixture-of-experts kernel.

vLLM's MoE module (``MoERunner``) runs the routed experts of a layer through
``TritonExperts.apply``: one
fused matrix multiply for the up and gate projections of every routed
``(token, expert)`` pair, the activation, and a second fused multiply for the
down projection, which also multiplies each row by its routing weight. The rows
live in workspaces that later layers reuse.

`serve` is registered on ``MoERunner._apply_quant_method``, which runs under the
MoE module's forward, records ``TritonExperts.apply`` underneath, and serves the sites of the
transformers reader ([`Experts`][nnsight.modeling.kernels.experts.Experts]) in
vLLM's layout: no batch axis, and the token axis holds the rows this step
computed for the requesting invoke.

* ``expert_input``: ``[tokens, top_k, hidden]``, the layer's input row.
* ``expert_activation``: ``[tokens, top_k, intermediate]``, the kernel's own
  activation rows, copied before the workspace is reused.
* ``expert_output``: ``[tokens, top_k, hidden]``, the down projection before the
  routing weight. The kernel never stores it, so the selected rows are
  recomputed from the activation rows and the expert's down-projection weight.

Under expert parallelism each rank holds a share of the experts and runs only
the pairs routed to them, so the reader declares ``slot`` and ``expert`` as
`ROUTED`: a rank returns zeros for the pairs of experts it does not hold, and
`Sites.read` sums the ranks.

Under tensor parallelism without expert parallelism each rank holds every
expert, cut to one contiguous block of its intermediate width. The activation
rows' ``width`` axis is `SPLIT`, and a rank's down projection is its share of a
sum over the width, so ``expert_output`` is summed over the ranks.

Not served: other experts kernels (FlashInfer, batched Triton, quantized) and
LoRA.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import torch

from ...intervention.kernel import Attr, Located, Spec, unshimmed
from ...intervention.sites import ROUTED, SPLIT, Batch, Provenance, Sites, Unavailable
from ...intervention.source import arguments
from ..kernels.experts import Experts

NAME = "MoERunner._apply_quant_method"


class _Capture(threading.local):
    """Thread-local list collecting one record per ``TritonExperts.apply`` call during `serve` (``None`` outside a served call)."""

    calls: list | None = None


_capture = _Capture()
_installed = False
_lock = threading.Lock()


def _install() -> None:
    """Wrap ``TritonExperts.apply`` and its fused multiply once so that, inside a served call, they record what the reader needs."""
    global _installed
    with _lock:
        if _installed:
            return
        from vllm.model_executor.layers.fused_moe.experts import triton_moe

        apply, multiply = triton_moe.TritonExperts.apply, triton_moe.invoke_fused_moe_triton_kernel

        def recording_apply(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            if calls is None:
                return apply(*args, **kwargs)
            bound = arguments(apply, args, kwargs)
            # The workspaces and the input buffer are reused, so the rows are copied while they hold this call's values.
            calls.append(call := {"bound": bound, "hidden": bound["hidden_states"].detach().clone(), "index": bound["topk_ids"].detach().clone(), "inputs": []})
            output = apply(*args, **kwargs)
            call["closed"] = True
            return output

        def recording_multiply(*args: Any, **kwargs: Any) -> Any:
            calls = _capture.calls
            if calls and "closed" not in calls[-1]:
                calls[-1]["inputs"].append(args[0].detach().clone())
            return multiply(*args, **kwargs)

        triton_moe.TritonExperts.apply = recording_apply
        triton_moe.invoke_fused_moe_triton_kernel = recording_multiply
        _installed = True


def serve(fn: Callable, args: tuple, kwargs: dict) -> tuple[Any, Sites]:
    """Run the layer's routed experts with ``TritonExperts.apply`` recording, and hand back a `Routed` reader."""
    from vllm.forward_context import get_forward_context

    from .fragments import _tp_world_size

    _install()
    outer, _capture.calls = _capture.calls, (calls := [])
    try:
        output = fn(*args, **kwargs)
    finally:
        _capture.calls = outer
    if len(calls) != 1:
        return output, Unavailable(f"{NAME} ran TritonExperts.apply {len(calls)} times; kernel sites cover a layer whose routed experts run once through the Triton experts kernel")
    call = calls[0]
    bound, inputs, index = call["bound"], call["inputs"], call["index"]
    experts = bound["self"]
    local = bound["expert_map"]
    partial = _tp_world_size() > 1 and local is None
    if partial and experts.w2_bias is not None:
        return output, Unavailable(f"{NAME}: kernel sites are not served for experts with a bias under tensor parallelism")
    if experts._lora_context is not None or experts.quant_dtype is not None or experts.weight_quant_dtype is not None:
        return output, Unavailable(f"{NAME}: kernel sites are not served with LoRA or quantization")
    pairs = index.numel()
    if len(inputs) != 2 or inputs[1].shape[0] != pairs or inputs[1].shape[1] != bound["w2"].shape[2]:
        return output, Unavailable(f"{NAME} ran {len(inputs)} fused multiplies with inputs {[tuple(v.shape) for v in inputs]}; expected two, the second over {pairs} activation rows")
    # A replayed graph carries the rows of the batch size it was recorded for; the step fills the first of them.
    metadata = get_forward_context().attn_metadata
    actual = next(iter(metadata.values())).num_actual_tokens if isinstance(metadata, dict) and metadata else index.shape[0]
    return output, Routed(call["hidden"], inputs[1], index, bound["w2"], experts.w2_bias, local, bound["global_num_experts"], partial, actual)


class Routed(Sites):
    """Site reader for one ``TritonExperts.apply`` call over every request of the step."""

    names = Experts.names
    axes = Experts.axes
    narrowed = True

    def __init__(
        self,
        hidden: torch.Tensor,
        activation: torch.Tensor,
        index: torch.Tensor,
        w2: torch.Tensor,
        bias: torch.Tensor | None,
        local: torch.Tensor | None,
        num_experts: int,
        partial: bool,
        actual: int,
    ) -> None:
        self.hidden, self.activation, self.index, self.w2, self.bias = hidden, activation, index, w2, bias
        #: Under expert parallelism, each expert's row in this rank's weights, or -1 for an expert on another rank.
        self.expert_map = local
        self.num_experts = num_experts
        #: Token rows the step filled; a replayed graph's buffers can hold more.
        self.actual = actual
        #: Tensor parallelism without expert parallelism: this rank holds one block of every expert's intermediate width.
        self.partial = partial
        self.layout = {"tp": {"width": SPLIT}} if partial else {} if local is None else {"tp": {"slot": ROUTED, "expert": ROUTED}}
        self.batch: Batch | None = None

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        self.batch = batch
        return super().read(site, spec, batch)

    def _site(self, site: str, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        tokens, top_k = self.index.shape
        start, size = (0, self.actual) if self.batch is None else (self.batch.start, self.batch.size)
        if start + size > tokens:
            raise ValueError(f"kernel sites: the worker's rows [{start}, {start + size}) are outside the {tokens} token rows of this call")
        rows = slice(start, start + size)
        activation = self.activation.view(tokens, top_k, -1)[rows].flatten(0, 1)
        value, provenance = _Request(self, self.hidden[rows], activation, self.index[rows]).read(site, spec, None)
        if self.partial and site == "expert_output":
            # Each rank's down projection covers its block of the width; the row is their sum.
            value = self.ranks.reduce(value.contiguous())
        return value, provenance

    def local(self, axis: str) -> int:
        return self.activation.shape[-1]

    def dim(self, site: str, axis: str, spec: dict[str, Any]) -> int:
        return -1  # the width is the last dimension whichever of slot= and expert= is used

    def expert_input(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        return self._site("expert_input", spec)

    def expert_activation(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        return self._site("expert_activation", spec)

    def expert_output(self, spec: dict[str, Any]) -> tuple[torch.Tensor, Provenance]:
        return self._site("expert_output", spec)


class _Request(Experts):
    """One request's rows of a `Routed` call, read with the transformers reader's selection rules."""

    def __init__(self, call: Routed, hidden: torch.Tensor, activation: torch.Tensor, index: torch.Tensor) -> None:
        super().__init__({"expert_activation": activation}, None, index, call.num_experts, NAME)
        self.call, self.hidden = call, hidden

    def kind(self, name: str) -> str:
        return "replay" if name == "expert_output" else "exact export"

    def gather(self, name: str, pairs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        call = self.call
        # Each selected pair's expert as a row of this rank's weights; -1 for an expert on another rank.
        expert = self.top_k_index.reshape(-1)[pairs].long()
        if call.expert_map is not None:
            expert = call.expert_map[expert].long()
        held = (expert >= 0).unsqueeze(-1)
        if name == "expert_input":
            return self.hidden[pairs // self.top_k_index.shape[1]] * held
        # The kernel never writes the rows of another rank's experts, so those hold whatever the workspace had.
        activation = torch.where(held, super().gather("expert_activation", pairs, index), 0)
        if name == "expert_activation":
            return activation
        # The down projection of each selected row by its own expert, one multiply per expert present.
        w2 = call.w2
        output = activation.new_zeros((*pairs.shape, w2.shape[1]), dtype=w2.dtype)
        for row in expert[expert >= 0].unique().tolist():
            mask = expert == row
            output[mask] = activation[mask].to(w2.dtype) @ w2[row].T
            if call.bias is not None:
                output[mask] += call.bias[row]
        return output


def recorded_into_graph(module: torch.nn.Module) -> bool:
    """Whether ``module`` is a layer whose experts kernel vLLM records into a CUDA graph without a break of its own."""
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    except ImportError:
        return False
    return isinstance(module, MoERunner)


def routed_experts(context: None) -> Located:
    """Locate ``MoERunner._apply_quant_method``, the call under the MoE module's forward that runs its routed experts."""
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    except ImportError:  # a vLLM without the modular MoE runner
        return
    yield unshimmed(MoERunner._apply_quant_method), (Attr(MoERunner, "_apply_quant_method"),)


CATALOG: list[Spec] = [
    Spec(NAME, "experts", routed_experts, serve),
]
