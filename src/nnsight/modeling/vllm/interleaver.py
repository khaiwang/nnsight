"""An interleaver whose hooks survive CUDA-graph replay, at declared taps.

A replayed CUDA graph runs no Python, so the forward hooks
[`Interleaver.instrument`][nnsight.intervention.interleaver.Interleaver.instrument]
installs never fire under ``enforce_eager=False``. vLLM's *breakable* graphs
(``VLLM_USE_BREAKABLE_CUDAGRAPH=1``) leave a seam: a callable handed to the
recording's ``add_eager`` while the graph is being captured is run at that point
of every replay, against the recording's own tensors. A tap is
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle]
registered as one of those callables — the same handoff a hook makes, replayed.

Two things follow from replay that an eager hook never meets. The value at a
location lives at a fixed address the graph rewrites every step, so a kept
reference — ``.save()``, a list appended under ``tracer.iter`` — aliases memory
the next step overwrites: clone what you keep. And the callable's return is
discarded, so an edit has to land *in place*; in-place edits already do, and a
replacement swap is copied back into the live tensor, shape-checked.

A tap on a module's kernel sites, ``{path}.kernel``, is served by the kernel
shim (intervention/kernel.py), which vLLM's own eager break around the attention
op already runs on every replay. The shim attributes a kernel call to the module
whose forward is running, and a replayed step runs no forward. So while the
graph is recorded, two more callables go in around that module's body: one that
marks the module as running and one that unmarks it.

Taps are the only locations a replayed step reaches. Every other module's
handoff lives in its forward, which a replayed graph never runs; a block parked
on one is told so when its request ends (see ``Requests.finish_dangling``).
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from ...intervention import kernel
from ...intervention.interleaver import Interleaver
from ...util import apply


def _capture() -> Any:
    """vLLM's breakable graph capture while it is recording, else ``None``."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    capture = BreakableCUDAGraphCapture.current()
    return capture if capture is not None and capture._capturing else None


def _weak(value: Any) -> Any:
    """Weak-ref the CUDA tensors in a value, as vLLM's own eager breaks do.

    A strong reference from the recorded callable pins the recording's pool slot,
    and replay would then serve the warmup's tensors instead of the step's.
    """
    try:
        from vllm.utils.torch_utils import weak_ref_tensor
    except ImportError:
        from vllm.utils import weak_ref_tensor
    return apply(value, lambda t: weak_ref_tensor(t) if t.is_cuda else t, torch.Tensor)


def _copy_into(live: Any, edited: Any) -> None:
    """Land ``edited`` in ``live``'s memory, tensor by tensor — the graph reads from there.

    What arrives here has ``live``'s shape by construction: a swap goes through
    the batcher's widen, which raises inside the worker's handoff (and so errors
    that request alone) if the replacement does not fit its rows.
    """
    if edited is live:
        return
    if isinstance(live, torch.Tensor):
        live.copy_(edited)
    elif isinstance(live, (tuple, list)):
        for live_item, edited_item in zip(live, edited):
            _copy_into(live_item, edited_item)
    elif isinstance(live, dict):
        for key, live_item in live.items():
            _copy_into(live_item, edited[key])


class VLLMInterleaver(Interleaver):
    """An [`Interleaver`][nnsight.intervention.interleaver.Interleaver] that records its handoff into CUDA graphs at ``taps``.

    Attributes:
        taps: Full locations (``"model.model.layers.3.output"``) a replayed step
            serves. Empty on an ``enforce_eager`` engine, where this class behaves
            exactly as its base.
    """

    def __init__(self, taps: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.taps = frozenset(taps)
        # A module's controller hands off ``{path}.skip`` just before its body and
        # ``{path}.output`` just after (`Controller.__call__`), which is where the
        # callables around a kernel tap's body are recorded.
        paths = [tap[: -len(kernel.SUFFIX)] for tap in self.taps if tap.endswith(kernel.SUFFIX)]
        #: Modules with a kernel tap whose kernel vLLM records into the graph: their body is kept in Python.
        self._bodies: set[str] = set()
        #: Depth of such bodies running now. A tap reached inside one is served on the spot.
        self._live = 0
        self._before = {f"{path}.skip": path for path in paths}
        self._after = frozenset(f"{path}.output" for path in paths)
        #: Frames of modules marked as running by a replayed step.
        self._running: list = []

    def instrument(self, envoy: Any) -> None:
        """Route the module through this interleaver, and open its forward if a tap reaches inside.

        A tap on an operation — ``{path}.source.{op}.output`` — is served by the
        module's source-instrumented forward, which hands each operation to
        `handle` as it runs. That forward is installed lazily, on the first
        ``.source`` access, and the graph recording runs before any trace could
        ask for it; so install it here, as the tree is built, for exactly the
        modules the taps name — and check the op exists while a typo is still a
        load error rather than a request that parks forever.
        """
        super().instrument(envoy)
        if f"{envoy.path}{kernel.SUFFIX}" in self.taps:
            from .experts import recorded_into_graph

            if recorded_into_graph(envoy._module):
                self._bodies.add(envoy.path)
        prefix = f"{envoy.path}.source."
        ops = {tap[len(prefix):].rpartition(".")[0] for tap in self.taps if tap.startswith(prefix)}
        if not ops:
            return
        source = envoy.source  # installs the instrumented forward; raises SourceNotAvailable
        missing = sorted(op for op in ops if op not in source.names)
        if missing:
            raise ValueError(
                f"Tap on {envoy.path}.source: its forward has no operation "
                f"{missing[0]!r}. Its operations are {list(source.names)}."
            )

    def body(self, path: str, state: Any, module: Any, args: tuple, kwargs: dict) -> Any:
        """Run a module's body; while vLLM records a graph, keep the body of a kernel-tapped module in Python.

        vLLM's attention and state-space layers break the graph around their
        kernel themselves, so a replayed step runs that kernel in Python and its
        sites can be served. A layer vLLM records whole (the fused MoE) runs no
        Python on replay. For such a module the body is recorded as a break of
        its own, and its output is copied into a buffer the following graph
        segment reads, as vLLM's own breaks write into a caller's buffer.
        """
        capture = _capture()
        if path not in self._bodies or capture is None:
            return super().body(path, state, module, args, kwargs)
        live_args, live_kwargs = _weak(args), _weak(kwargs)
        buffer: list[torch.Tensor] = []

        def step() -> None:
            self._live += 1
            try:
                output = super(VLLMInterleaver, self).body(path, state, module, live_args, live_kwargs)
            finally:
                self._live -= 1
            if buffer:
                buffer[0].copy_(output)
            else:
                buffer.append(output.clone())

        capture.add_eager(step)
        return buffer[0]

    def _enter(self, path: str) -> None:
        """Mark the module at ``path`` as running, on a replayed step."""
        # A kernel that raised on an earlier step left its module marked.
        self._leave()
        self._running.append(kernel.push(self, path))

    def _leave(self, output: str | None = None) -> None:
        """Unmark the running module, and count the visit its kernel sites are scheduled by.

        A request for a kernel site at step ``n`` is matched against the count of
        the module's ``.output`` visits. A tapped ``.output`` is counted when it
        is served; one that is not a tap is never handed off, so it is counted here.
        """
        while self._running:
            kernel.pop(self._running.pop())
        if output is not None and output not in self.taps and self.interleaving:
            self.counts[output] = self.counts.get(output, 0) + 1

    def handle(self, provider: str, value: Any) -> Any:
        if self._live:
            # Inside a body kept in Python, this code runs on every step: nothing is recorded, and a tap is
            # served here, on the rows the step filled.
            if provider in self.taps and self.interleaving:
                self.replay(provider, value, super().handle)
            return value
        path = self._before.get(provider)
        if path is not None or provider in self._after:
            capture = _capture()
            if capture is not None:
                capture.add_eager((lambda: self._leave(provider)) if path is None else (lambda: self._enter(path)))
            elif path is None:
                # A step too long for any graph runs the module's forward itself.
                self._leave(provider)
        # While vLLM is recording a graph, a tap's handoff is not served now but
        # registered to be replayed at this point of every later step. The runner
        # enters the interleaver around that recording so the handoff reaches here.
        if provider in self.taps:
            capture = _capture()
            if capture is not None:
                live = _weak(value)
                capture.add_eager(lambda: self.replay(provider, live))
                return value
        elif self.taps and provider.rsplit(".", 1)[-1] in ("input", "output", "skip"):
            # A module location that is not a tap. A replayed graph never reaches
            # it — but a prompt too long for any graph runs eagerly through here,
            # and would be served on that step and not the next. Skipped either
            # way, so what a tapped engine serves does not depend on prompt length.
            return value
        return super().handle(provider, value)

    def replay(self, location: str, live: Any, handle: Any = None) -> None:
        """Serve ``location`` from a replayed graph: `handle`, then land any edit in place.

        A graph is recorded for a batch size, so its tensors carry that many rows
        and this step fills the first ``batcher.total`` of them; the rest is
        padding no worker owns. Trimmed to a view before the handoff, so a worker
        is narrowed out of the rows that exist — and an edit copied back into the
        view lands in the graph's memory all the same.
        """
        if not self.interleaving:
            return
        total = self.batcher.total
        live = apply(live, lambda t: t[:total] if t.dim() and t.shape[0] > total else t, torch.Tensor)
        _copy_into(live, (handle or self.handle)(location, live))
