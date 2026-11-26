"""Modules of your own attached to an engine's model (`VLLM.attach`).

An attached module is held once in the model's tree on every worker, so the
requests of an engine step can share its calls: a token-wise module registered
with ``rows="token"``, a module that reduces over each request's tokens with
``rows="span"``. The span modules are three published sequence-level probes
with seeded weights (``tests/probes.py``). A request's result in a trace of
several requests of different lengths is compared with the module run here on
that request's own rows, which no other request's rows can enter, and with the
same request traced alone, on one rank and across ranks. The engines are built
before anything is attached.
"""

import pytest
import torch
from probes import AggregateProbe, AttentionPoolProbe, Dictionary, StructuralProbe

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "Madison Square Garden is in",
    "Water boils at a temperature of one hundred degrees Celsius at sea level, and",
    "The capital of Japan is",
]
SAMPLING = dict(temperature=0.0, top_p=1, max_tokens=1)
SPANS = ["attention_pool", "aggregate", "structure"]


def _attach(qwen):
    """Attach every probe once; the engine is shared by the module's tests."""
    width = qwen.model.norm.weight.shape[0]
    qwen.attach("attention_pool", AttentionPoolProbe(width, seed=1), rows="span")
    qwen.attach("aggregate", AggregateProbe(width, seed=2), rows="span")
    qwen.attach("structure", StructuralProbe(width, rank=16, seed=3), rows="span")
    qwen.attach("feature_dictionary", Dictionary(width, features=256, seed=4), rows="token")
    # Nothing stated about its rows: its linear child is shared by its class.
    qwen.attach("plain_reader", StructuralProbe(width, rank=8, seed=5))
    return qwen


@pytest.fixture(scope="module")
def qwen_ref(vllm_qwen_ref):
    return _attach(vllm_qwen_ref)


@pytest.fixture(scope="module")
def qwen_tp(vllm_qwen_tp):
    return _attach(vllm_qwen_tp)


def _read(model, name):
    """One layer's residual stream for the request being traced, and ``name``'s
    result on it."""
    out = model.model.layers[3].output
    stream = (out[0] + out[1]).clone()
    return getattr(model, name)(stream), stream


@torch.no_grad()
def _match(qwen, name):
    with qwen.trace(**SAMPLING) as tracer:
        together = [None] * len(PROMPTS)
        together = together.save()
        for index, prompt in enumerate(PROMPTS):
            with tracer.invoke(prompt):
                together[index] = _read(qwen, name)

    lengths = [len(stream) for _, stream in together]
    assert len(set(lengths)) > 2  # ragged spans
    module = getattr(qwen, name)._module  # this process's copy, same weights
    for index, prompt in enumerate(PROMPTS):
        ours, stream = (value.float().cpu() for value in together[index])
        # On the request's own rows, as the engine served them.
        own = module(stream, sizes=[len(stream)])[0] if name in SPANS else module(stream)
        _close(ours, own, 1e-3)
        with qwen.trace(prompt, **SAMPLING):
            alone = _read(qwen, name)[0].save()
        # bf16 arithmetic depends on the batch a request ran in.
        _close(ours, alone.float().cpu(), 0.05)
    if name == "structure":
        assert [tuple(value.shape) for value, _ in together] == [(n, n) for n in lengths]


def _close(ours, theirs, tolerance):
    assert ours.shape == theirs.shape
    assert (ours - theirs).abs().max() / theirs.abs().max().clamp_min(1e-6) < tolerance


@pytest.mark.parametrize("name", [*SPANS, "feature_dictionary"])
def test_an_attached_modules_result_matches_the_request_traced_alone(qwen_ref, name):
    _match(qwen_ref, name)


@pytest.mark.parametrize("name", [*SPANS, "feature_dictionary"])
def test_a_sharded_engines_attached_module_matches_the_request_traced_alone(qwen_tp, name):
    _match(qwen_tp, name)


@torch.no_grad()
@pytest.mark.parametrize(
    "name, child",
    [("attention_pool", None), ("aggregate", None), ("structure", None), ("feature_dictionary", None), ("plain_reader", "projection")],
)
def test_requests_of_one_step_share_one_call(qwen_ref, name, child):
    qwen = qwen_ref

    def watched(model):
        envoy = getattr(model, name)
        return (envoy if child is None else getattr(envoy, child))._module

    with qwen.trace(PROMPTS[0], **SAMPLING):
        module = watched(qwen)
        module.seen = []
        module.watch = module.register_forward_pre_hook(
            lambda module, args: module.seen.append(args[0].shape[0])
        )

    with qwen.trace(**SAMPLING) as tracer:
        rows = [None] * len(PROMPTS)
        rows = rows.save()
        for index, prompt in enumerate(PROMPTS):
            with tracer.invoke(prompt):
                out = qwen.model.layers[3].output
                stream = (out[0] + out[1]).clone()
                envoy = getattr(qwen, name)
                (envoy if child is None else getattr(envoy, child))(stream.float() if child else stream)
                rows[index] = stream.shape[0]

    with qwen.trace(PROMPTS[0], **SAMPLING):
        module = watched(qwen)
        seen = list(module.seen).save()
        module.watch.remove()
        del module.seen, module.watch

    # Every request's rows went through, in fewer calls than requests.
    assert sum(seen) == sum(rows) and len(seen) < len(PROMPTS)


def test_attach_raises_on_a_name_or_module_it_cannot_take(qwen_ref):
    for name in ("aggregate", "lm_head", "output", "trace", "class", "_hidden"):
        with pytest.raises(ValueError, match="does not already use"):
            qwen_ref.attach(name, torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="'token' or 'span'"):
        qwen_ref.attach("another_reader", torch.nn.Linear(4, 4), rows="sequence")
    with pytest.raises(TypeError, match="torch.nn.Module"):
        qwen_ref.attach("another_reader", lambda rows: rows)
    # The same module object under a second name, and a module holding one of
    # the model's.
    with pytest.raises(ValueError, match="already in this model's tree"):
        qwen_ref.attach("another_reader", qwen_ref.aggregate._module)
    with pytest.raises(ValueError, match="already in this model's tree"):
        qwen_ref.attach("another_reader", torch.nn.Sequential(qwen_ref.model.norm._module))
    assert "another_reader" not in dict(qwen_ref._module.named_children())


@torch.no_grad()
def test_a_module_serialized_with_another_tree_is_sent_with_its_weights(qwen_ref):
    from nnsight import NNsight
    from nnsight.intervention.serialization import dumps

    # Serializing a tree leaves that tree's path on each of its modules.
    holder = NNsight(torch.nn.Sequential())
    holder.reader = Dictionary(qwen_ref.model.norm.weight.shape[0], features=8, seed=6)
    dumps(holder)
    module = holder.reader._module
    qwen_ref.attach("carried_dictionary", module, rows="token")

    with qwen_ref.trace(PROMPTS[0], **SAMPLING):
        ours, stream = _read(qwen_ref, "carried_dictionary")
        ours, stream = ours.save(), stream.save()
    _close(ours.float().cpu(), module(stream.float().cpu()), 1e-3)
