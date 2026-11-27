"""Recurrent kernel sites on vLLM's gated delta rule layers (Qwen3.5), checked against the transformers sites.

Each test runs on three engines, one alive at a time: the eager engine, an
engine replaying CUDA graphs with the layer's kernel sites tapped, and an engine
on two tensor-parallel ranks, where each rank holds half of the heads.

The comparison uses layer 0, whose inputs both runtimes compute from the same
embeddings. vLLM's Triton kernel runs in bf16 and transformers' reference in
fp32, so they agree to bf16 rounding; a state read from the wrong rank, request
or chunk differs by the size of the state itself.
"""

import pytest
import torch

REPO = "Qwen/Qwen3.5-0.8B"
# 145 tokens: three chunks of 64, so the states cross two chunk boundaries.
PROMPT = "The Eiffel Tower is located in the city of Paris and " * 12
SHORT = "The capital of Japan is"
TOKENS = [0, 5, 63, 64, 100, 144]
HEADS = [15, 2, -9]


@pytest.fixture(scope="module", params=["eager", "taps", "two_ranks"])
def engine(request):
    from nnsight.modeling.vllm import VLLM

    kwargs = {}
    if request.param == "taps":
        kwargs["taps"] = ["language_model.model.layers.*.linear_attn.kernel", "language_model.model.layers.*.output"]
    if request.param == "two_ranks":
        if torch.cuda.device_count() < 2:
            pytest.skip("two tensor-parallel ranks need two GPUs")
        kwargs["tensor_parallel_size"] = 2
    return VLLM(REPO, gpu_memory_utilization=0.45, dispatch=True, **kwargs)


@pytest.fixture(scope="module")
def reference():
    """The transformers sites of layer 0 for `PROMPT`: chunk states, states at `TOKENS`, and those states at `HEADS`."""
    from nnsight import TransformersModel

    model = TransformersModel(REPO, task="text-generation", device="cuda", dtype=torch.bfloat16, dispatch=True)
    attn = model.model.layers[0].linear_attn
    with model.trace(PROMPT):
        chunks = attn.kernel.chunk_states().save()
        states = attn.kernel.states(tokens=TOKENS).save()
    return chunks[0].float().cpu(), states[0].float().cpu()


def _relative(value, expected):
    return ((value.float().cpu() - expected).abs().max() / expected.abs().max()).item()


def test_states_match_hf(engine, reference):
    attn = engine.language_model.model.layers[0].linear_attn
    with engine.trace(PROMPT, temperature=0.0, max_tokens=1):
        chunks = attn.kernel.chunk_states().clone().save()
        states = attn.kernel.states(tokens=TOKENS).clone().save()
        some = attn.kernel.states(tokens=TOKENS, heads=HEADS).clone().save()
        one = attn.kernel.chunk_states(chunks=-1, heads=11).clone().save()
    expected_chunks, expected_states = reference
    assert chunks.shape == expected_chunks.shape == (4, 16, 128, 128)
    assert (chunks[0] == 0).all()
    assert _relative(chunks[1:], expected_chunks[1:]) < 2e-2
    assert states.shape == expected_states.shape
    assert _relative(states, expected_states) < 2e-2
    assert torch.equal(some, states[:, HEADS])
    assert torch.equal(one, chunks[-1, 11])


def test_decode_steps_continue_each_request(engine):
    attn = engine.language_model.model.layers[0].linear_attn
    with engine.trace(temperature=0.0, max_tokens=3) as tracer:
        with tracer.invoke(PROMPT):
            long = list().save()
            for _ in tracer.iter[:3]:
                long.append((attn.kernel.chunk_states().clone(), attn.kernel.states().clone()))
        with tracer.invoke(SHORT):
            short = list().save()
            for _ in tracer.iter[:3]:
                short.append((attn.kernel.chunk_states().clone(), attn.kernel.states().clone()))
    # A prefill serves the state entering each chunk and the final one; a decode step the state before and after its token.
    assert [chunks.shape[0] for chunks, _ in long] == [4, 2, 2]
    assert [states.shape[0] for _, states in long] == [145, 1, 1]
    assert short[0][0].shape[0] == 2 and short[0][1].shape[0] < 64
    assert not torch.equal(long[0][0][-1], short[0][0][-1])
    for steps in (long, short):
        assert torch.equal(steps[0][1][-1], steps[0][0][-1])
        for previous, (chunks, states) in zip(steps, steps[1:]):
            assert torch.equal(chunks[0], previous[0][-1])
            assert torch.equal(states[0], chunks[1])


# ---------------------------------------------------------------------------
# Kimi delta attention (a tiny random Kimi Linear): a per-key-channel decay
# ---------------------------------------------------------------------------

KIMI = "yujiepan/kimi-linear-tiny-random"


@pytest.mark.parametrize("mode", ["eager", "taps", "two_ranks"])
def test_kimi_states_match_hf(mode):
    """Every token's state of the first delta attention layer, on a step that prefills two requests."""
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from nnsight import TransformersModel
    from nnsight.modeling.vllm import VLLM

    # The tokenizer's code lives in the released model's repository.
    kwargs = {}
    if mode == "taps":
        kwargs = {"taps": ["model.layers.*.self_attn.kernel"]}
    if mode == "two_ranks":
        if torch.cuda.device_count() < 2:
            pytest.skip("two tensor-parallel ranks need two GPUs")
        kwargs = {"tensor_parallel_size": 2}
    engine = VLLM(KIMI, gpu_memory_utilization=0.4, dispatch=True, trust_remote_code=True, max_model_len=1024, **kwargs)
    kernel = engine.model.layers[0].self_attn.kernel
    with engine.trace(temperature=0.0, max_tokens=2) as tracer:
        with tracer.invoke(SHORT):
            other = kernel.chunk_states().shape.save()
        with tracer.invoke(PROMPT):
            steps = list().save()
            for _ in tracer.iter[:2]:
                steps.append((kernel.chunk_states().clone(), kernel.states().clone()))

    # transformers' own Kimi Linear class wants every layer listed; the checkpoint leaves its last one out.
    # vLLM registers a config class of its own for this model type, so transformers' classes are named here.
    import transformers

    config = json.loads((Path(snapshot_download(KIMI)) / "config.json").read_text())
    config["linear_attn_config"]["full_attn_layers"].append(config["num_hidden_layers"])
    config = transformers.KimiLinearConfig.from_dict(config)
    module = transformers.KimiLinearForCausalLM.from_pretrained(KIMI, config=config, dtype=torch.float32)
    reference = TransformersModel(module, task="text-generation", tokenizer=engine.tokenizer, device="cuda")
    with reference.trace(PROMPT):
        expected = reference.model.layers[0].self_attn.kernel.states().save()
    expected = expected[0].float().cpu()

    chunks, states = steps[0]
    count = expected.shape[0]
    assert tuple(other) == (2, *expected.shape[1:])
    assert states.shape == expected.shape and chunks.shape[0] == -(-count // 64) + 1
    scale = expected.abs().amax((1, 2, 3))
    assert float(((states.float().cpu() - expected).abs().amax((1, 2, 3)) / scale).max()) < 3e-2
    # The decode step starts from the state the prefill left in the cache.
    assert _relative(steps[1][0][0], chunks[-1].float().cpu()) < 1e-3
    assert tuple(steps[1][1].shape) == (1, *expected.shape[1:])
