"""Recurrent kernel sites on vLLM's Mamba2 layer (mamba2-130m), checked against the transformers sites.

vLLM runs the model in bf16 and the transformers reference runs in fp32, so the
states agree to bf16 rounding; a state of the wrong request, chunk or head
differs by the size of the state.

The checkpoint's config names no architecture, so the engine is told which
vLLM model class to load.
"""

import pytest
import torch

REPO = "AntonV/mamba2-130m-hf"
# 391 tokens with the model's chunk size of 256: two chunks.
LONG = "The Eiffel Tower is located in the city of Paris and " * 30
SHORT = "The capital of Japan is"
TOKENS = [0, 100, 250, 251, 255, 256, 300, 390]
HEADS = [20, 3, -1]

#: Values the one-rank engine read, for the tensor-parallel engine to compare against.
ONE_RANK: dict = {}


@pytest.fixture(scope="module", params=["one_rank", "tensor_parallel"])
def engine(request):
    from nnsight.modeling.vllm import VLLM

    kwargs = {}
    if request.param == "tensor_parallel":
        if torch.cuda.device_count() < 2:
            pytest.skip("tensor parallelism needs two GPUs")
        kwargs = {"tensor_parallel_size": 2}
    return VLLM(REPO, gpu_memory_utilization=0.2, dispatch=True, hf_overrides={"architectures": ["Mamba2ForCausalLM"]}, **kwargs)


@pytest.fixture(scope="module")
def reference():
    """The transformers sites of layer 0 for `LONG`: the states after `TOKENS`."""
    from nnsight import TransformersModel

    model = TransformersModel(REPO, task="text-generation", device="cuda", dtype=torch.float32, dispatch=True)
    with model.trace(LONG):
        states = model.backbone.layers[0].mixer.kernel.states(tokens=TOKENS).save()
    return states[0].float().cpu()


def _relative(value, expected):
    return float(((value.float().cpu() - expected).abs().amax((-1, -2, -3)) / expected.abs().amax((-1, -2, -3))).max())


def test_states_match_hf(engine, reference, request):
    kernel = engine.backbone.layers[0].mixer.kernel
    with engine.trace(temperature=0.0, max_tokens=1) as tracer:
        # A second request shares the prefill strip. vLLM cuts chunks along the strip, so the long request's
        # first chunk ends at token 255 when it is first in the strip and at token 250 when the 5-token request is.
        with tracer.invoke(SHORT):
            other = kernel.chunk_states().shape.save()
        with tracer.invoke(LONG):
            chunks = kernel.chunk_states().clone().save()
            states = kernel.states(tokens=TOKENS).clone().save()
            some = kernel.states(tokens=TOKENS, heads=HEADS).clone().save()
            one = kernel.chunk_states(chunks=1, heads=7).clone().save()
    assert tuple(other) == (2, 24, 64, 128)
    assert chunks.shape == (3, 24, 64, 128) and (chunks[0] == 0).all()
    assert states.shape == reference.shape
    assert _relative(states, reference) < 2e-2
    assert torch.equal(some, states[:, HEADS])
    assert torch.equal(one, chunks[1, 7])
    # The tokens that end the kernel's chunks read the kernel's own states.
    assert torch.equal(states[2], chunks[1].float()) or torch.equal(states[4], chunks[1].float())
    assert torch.equal(states[-1], chunks[-1].float())
    if "one_rank" in request.node.name:
        ONE_RANK["states"] = states.float().cpu()
    else:
        assert _relative(states, ONE_RANK["states"]) < 2e-2


def test_decode_steps_continue_each_request(engine):
    kernel = engine.backbone.layers[0].mixer.kernel
    with engine.trace(temperature=0.0, max_tokens=3) as tracer:
        with tracer.invoke(LONG):
            long = list().save()
            for _ in tracer.iter[:3]:
                long.append((kernel.chunk_states().clone(), kernel.states(tokens=-1).clone()))
        with tracer.invoke(SHORT):
            short = list().save()
            for _ in tracer.iter[:3]:
                short.append((kernel.chunk_states().clone(), kernel.states(tokens=-1).clone()))
    assert [chunks.shape[0] for chunks, _ in long] == [3, 2, 2]
    assert [chunks.shape[0] for chunks, _ in short] == [2, 2, 2]
    assert not torch.equal(long[0][0][-1], short[0][0][-1])
    for steps in (long, short):
        for index, (chunks, last) in enumerate(steps):
            assert torch.equal(last, chunks[-1].float())
            if index:
                assert torch.equal(chunks[0], steps[index - 1][0][-1])
