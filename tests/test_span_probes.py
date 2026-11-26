"""Modules that reduce over each worker's own rows, run once for every worker.

A module registered with ``rows="span"`` is called with the workers' rows
concatenated and each worker's row count, and returns one item per worker
(``intervention/coalesce.py``). The probes are three published sequence-level
probes with seeded weights (``tests/probes.py``); the toy model stacks its
invokes on dim 0 and uses module names no hosted model has.
"""

from __future__ import annotations

import pytest
import torch
from probes import AggregateProbe, AttentionPoolProbe, Dictionary, StructuralProbe

import nnsight
from nnsight import NNsight
from nnsight.intervention.coalesce import Coalescer

WIDTH = 16
SPANS = {
    "attention_pool": lambda: AttentionPoolProbe(WIDTH, heads=3, values=5, seed=1),
    "aggregate": lambda: AggregateProbe(WIDTH, suffix=2, seed=2),
    "structure": lambda: StructuralProbe(WIDTH, rank=4, seed=3),
}


class Trunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixers = torch.nn.ModuleList(torch.nn.Linear(WIDTH, WIDTH) for _ in range(2))
        self.readers = torch.nn.ModuleDict({name: build() for name, build in SPANS.items()})
        self.feature_dictionary = Dictionary(WIDTH, features=32, seed=4)

    def forward(self, hidden):
        for mixer in self.mixers:
            hidden = torch.tanh(mixer(hidden))
        return hidden


class Stacked(NNsight):
    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}


@pytest.fixture
def model():
    torch.manual_seed(0)
    model = Stacked(Trunk().eval())
    coalescer = Coalescer()
    for name in SPANS:
        coalescer.register(model.readers[name]._module, rows="span")
    coalescer.register(model.feature_dictionary._module, rows="token")
    model.interleaver.coalescer = coalescer
    return model


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    # Ragged row counts, a one-row span among them.
    return [torch.randn(rows, WIDTH) for rows in (4, 1, 7, 3)]


def counted(module: torch.nn.Module) -> list:
    """A list that grows by the row count of every call of ``module``."""
    calls: list = []
    module.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    return calls


def read(model, inputs, name):
    """``name``'s result on the first mixer's output, one invoke per input."""
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                reader = model.readers[name] if name in SPANS else getattr(model, name)
                saved[index] = reader(model.mixers[0].output)
    return [saved[index] for index in range(len(inputs))]


def reference(model, rows, name):
    """The same read in plain torch, on one input's rows only."""
    with torch.no_grad():
        hidden = model._module.mixers[0](rows)
        if name in SPANS:
            return model._module.readers[name](hidden, sizes=[len(rows)])[0]
        return getattr(model._module, name)(hidden)


@pytest.mark.parametrize("name", [*SPANS, "feature_dictionary"])
def test_one_call_gives_every_invoke_its_own_result(model, inputs, name):
    module = model.readers[name]._module if name in SPANS else getattr(model, name)._module
    calls = counted(module)
    shared = read(model, inputs, name)

    assert calls == [sum(len(rows) for rows in inputs)]
    for rows, ours in zip(inputs, shared):
        expected = reference(model, rows, name)
        assert ours.shape == expected.shape
        assert torch.allclose(ours, expected, atol=1e-5)
    if name == "structure":
        assert [tuple(ours.shape) for ours in shared] == [(len(rows), len(rows)) for rows in inputs]


@pytest.mark.parametrize("name", SPANS)
def test_an_invokes_result_does_not_depend_on_the_other_invokes_rows(model, inputs, name):
    before = read(model, inputs, name)
    # Other values, and another length, for every invoke but the third.
    others = [rows if index == 2 else torch.randn(len(rows) + 2, WIDTH) * 1e3 for index, rows in enumerate(inputs)]
    after = read(model, others, name)

    assert torch.allclose(before[2], after[2], atol=1e-5)
    # The second entry: a structural probe's first is a token's distance to itself.
    assert not torch.allclose(before[0].flatten()[1], after[0].flatten()[1], atol=1e-5)


@pytest.mark.parametrize("name", SPANS)
def test_one_invoke_alone_is_a_span_of_its_own(model, inputs, name):
    calls = counted(model.readers[name]._module)
    with torch.no_grad(), model.trace(inputs[2]):
        alone = model.readers[name](model.mixers[0].output).save()
    assert calls == [len(inputs[2])]
    expected = reference(model, inputs[2], name)
    assert alone.shape == expected.shape
    assert torch.allclose(alone, expected, atol=1e-5)


def test_a_shared_span_call_that_raises_is_made_again_for_each_invoke(model, inputs):
    class Bounded(torch.nn.Module):
        """Each span's mean; raises on a row above its bound."""

        def forward(self, rows, sizes):
            if rows.abs().max() > 100:
                raise RuntimeError("out of bounds")
            return [span.mean(dim=0) for span in rows.split(sizes)]

    model.bounded_mean = Bounded()
    model.interleaver.coalescer.register(model.bounded_mean._module, rows="span")
    calls = counted(model.bounded_mean._module)
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                hidden = model.mixers[0].output
                try:
                    saved[index] = model.bounded_mean(hidden * 1e4 if index == 1 else hidden)
                except RuntimeError:
                    saved[index] = "raised"
    # The call on every invoke's rows, then one per invoke.
    assert calls[0] == sum(len(rows) for rows in inputs)
    assert sorted(calls[1:]) == sorted(len(rows) for rows in inputs)
    assert saved[1] == "raised"
    for index in (0, 2, 3):
        expected = model._module.mixers[0](inputs[index]).mean(dim=0)
        assert torch.allclose(saved[index], expected, atol=1e-5)


def errors(model, inputs, call):
    """The message of the error ``call(model, hidden)`` raises in each invoke."""
    model.interleaver.defer_exceptions = True
    with torch.no_grad(), model.trace() as tracer:
        saved = nnsight.save({})
        for index, rows in enumerate(inputs):
            with tracer.invoke(rows):
                hidden = model.mixers[0].output
                try:
                    call(model, hidden)
                    saved[index] = "returned"
                except Exception as error:
                    saved[index] = str(error)
    return [saved[index] for index in range(len(inputs))]


@pytest.mark.parametrize(
    "result, message",
    [
        (lambda rows, sizes: rows.new_zeros(len(sizes) + 1, 2), "one item per caller"),
        (lambda rows, sizes: rows.sum(), "as a tensor or a list; got Tensor"),
        (lambda rows, sizes: (rows.new_zeros(len(sizes), 2), {}), "as a tensor or a list; got tuple"),
    ],
    ids=["another-count", "one-value", "tuple"],
)
def test_a_span_result_that_is_not_one_item_per_invoke_is_an_error(model, inputs, result, message):
    class Returns(torch.nn.Module):
        def forward(self, rows, sizes):
            return result(rows, sizes)

    model.odd_reader = Returns()
    model.interleaver.coalescer.register(model.odd_reader._module, rows="span")
    # With every invoke, and with one alone.
    for given in (inputs, inputs[:1]):
        found = errors(model, given, lambda model, hidden: model.odd_reader(hidden))
        assert all(message in error for error in found)


@pytest.mark.parametrize(
    "call",
    [
        lambda model, hidden: model.readers["aggregate"](rows=hidden),
        lambda model, hidden: model.readers["aggregate"](hidden, hidden * 2.0),
        lambda model, hidden: model.readers["aggregate"](hidden, sizes=[len(hidden)]),
    ],
    ids=["rows-by-keyword", "a-second-tensor", "sizes-from-the-caller"],
)
def test_a_span_call_in_another_form_is_the_same_error_shared_or_alone(model, inputs, call):
    for given in (inputs, inputs[:1]):
        found = errors(model, given, call)
        assert all("a span module is called as module(rows, ...)" in error for error in found)


def test_register_raises_on_an_unknown_rows_value(model):
    with pytest.raises(ValueError, match="'token' or 'span'"):
        model.interleaver.coalescer.register(model.mixers[0]._module, rows="sequence")
