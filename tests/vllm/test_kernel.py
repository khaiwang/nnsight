"""Attention kernel sites on vLLM's FlashAttention backend, checked against the HF site.

vLLM values have no batch axis: ``probs`` is ``[queries, heads, keys]`` where HF's
is ``[batch, heads, queries, keys]``.

The comparison uses layer 0, whose queries and keys both engines compute from the
same embeddings. Deeper layers inherit bf16 rounding that differs between the
engines' kernels (layer 2 probabilities differ by up to 2e-2 on the prompt below,
less than HF bf16 differs from HF fp32).
"""

import pytest
import torch

PROMPT = "The Eiffel Tower is located in the city of"
LAYER = 0


@pytest.fixture(scope="module")
def reference():
    """HF gpt2 in vLLM's dtype: its SDPA site's ``probs`` for a prompt, as ``[queries, heads, keys]``."""
    from nnsight import TransformersModel

    model = TransformersModel("openai-community/gpt2", task="text-generation", device="cuda", dtype=torch.bfloat16, dispatch=True)

    def probs(prompt: str) -> torch.Tensor:
        with model.trace(prompt):
            value = model.transformer.h[LAYER].attn.kernel.probs().save()
        return value[0].transpose(0, 1).float().cpu()

    return probs


def test_prefill_probs_match_hf(vllm_gpt2, reference):
    attn = vllm_gpt2.transformer.h[LAYER].attn.attn
    with vllm_gpt2.trace(PROMPT, max_tokens=1, temperature=0.0):
        probs = attn.kernel.probs().save()
        scores = attn.kernel.scores().save()
        lse = attn.kernel.lse().save()
    expected = reference(PROMPT)
    assert probs.shape == expected.shape
    assert probs.provenance.kind == "exact reconstruction"
    assert lse.provenance.kind == "exact export"
    assert torch.allclose(probs.float().cpu(), expected, atol=1e-5)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-4)
    assert lse.shape == probs.shape[:2]
    assert torch.allclose(torch.logsumexp(scores, -1), lse, atol=1e-3)
    # Causal: a query row sees no later key.
    assert torch.isinf(scores[0, :, 1:]).all()


def test_decode_steps_serve_one_row_over_all_keys(vllm_gpt2):
    attn = vllm_gpt2.transformer.h[LAYER].attn.attn
    with vllm_gpt2.trace(PROMPT, max_tokens=3, temperature=0.0) as tracer:
        shapes = list().save()
        sums = list().save()
        for _ in tracer.iter[:3]:
            probs = attn.kernel.probs()
            shapes.append(tuple(probs.shape))
            sums.append(probs.sum(-1).clone())
    prompt_length = shapes[0][0]
    assert shapes == [(prompt_length, 12, prompt_length), (1, 12, prompt_length + 1), (1, 12, prompt_length + 2)]
    for value in sums:
        assert torch.allclose(value, torch.ones_like(value), atol=1e-4)


def test_each_invoke_reads_its_own_request(vllm_gpt2, reference):
    """Decode steps pack both requests' query rows into one kernel call; each invoke gets its own."""
    attn = vllm_gpt2.transformer.h[LAYER].attn.attn
    prompts = [PROMPT, "Hello world"]

    def alone(prompt: str) -> list:
        with vllm_gpt2.trace(prompt, max_tokens=3, temperature=0.0) as tracer:
            steps = list().save()
            for _ in tracer.iter[:3]:
                steps.append(attn.kernel.probs().clone())
        return steps

    with vllm_gpt2.trace(max_tokens=3, temperature=0.0) as tracer:
        with tracer.invoke(prompts[0]):
            first = list().save()
            for _ in tracer.iter[:3]:
                first.append(attn.kernel.probs().clone())
        with tracer.invoke(prompts[1]):
            second = list().save()
            for _ in tracer.iter[:3]:
                second.append(attn.kernel.probs().clone())
    for together, prompt in zip((first, second), prompts):
        expected = reference(prompt)
        assert torch.allclose(together[0].float().cpu(), expected, atol=1e-5)
        for value, solo in zip(together, alone(prompt)):
            assert value.shape == solo.shape
            assert torch.allclose(value, solo, atol=1e-5)


def test_axes_select_rows_heads_and_keys(vllm_gpt2):
    attn = vllm_gpt2.transformer.h[LAYER].attn.attn
    with vllm_gpt2.trace(PROMPT, max_tokens=1, temperature=0.0):
        full = attn.kernel.probs().save()
        last = attn.kernel.probs(query=-1, heads=[0, 5]).save()
        column = attn.kernel.probs(key=slice(0, 3), heads=7).save()
        lse = attn.kernel.lse(query=-1).save()
        whole_lse = attn.kernel.lse().save()
    assert torch.allclose(last, full[-1, [0, 5]], atol=1e-6)
    assert torch.allclose(column, full[:, 7, :3], atol=1e-6)
    assert torch.equal(lse, whole_lse[-1])


def test_reading_a_site_leaves_the_logits_unchanged(vllm_gpt2):
    with vllm_gpt2.trace(PROMPT, max_tokens=1, temperature=0.0):
        plain = vllm_gpt2.logits.clone().save()
    with vllm_gpt2.trace(PROMPT, max_tokens=1, temperature=0.0):
        read = vllm_gpt2.transformer.h[LAYER].attn.attn.kernel.probs().save()
        observed = vllm_gpt2.logits.clone().save()
    assert read.shape[1] == 12
    assert torch.equal(plain, observed)


def test_tensor_parallel_ranks_serve_every_head(vllm_qwen_ref, vllm_qwen_tp):
    """Each rank holds a block of the heads; a site reads all of them, indexed as on one rank."""

    def read(model):
        attn = model.model.layers[LAYER].self_attn.attn
        with model.trace(PROMPT, max_tokens=2, temperature=0.0) as tracer:
            steps = list().save()
            for _ in tracer.iter[:2]:
                steps.append(
                    (
                        attn.kernel.lse().clone(),
                        attn.kernel.probs().clone(),
                        attn.kernel.probs(query=-1, heads=9).clone(),
                        attn.kernel.scores(heads=[13, 2, -8], key=slice(0, 4)).clone(),
                    )
                )
        return steps

    for whole, sharded in zip(read(vllm_qwen_ref), read(vllm_qwen_tp)):
        assert whole[1].shape[1] == 14
        for expected, value in zip(whole, sharded):
            assert value.shape == expected.shape
            assert torch.allclose(value, expected, atol=1e-5)


@pytest.fixture(scope="module")
def vllm_gpt2_kernel_taps():
    """A gpt2 replaying CUDA graphs, with every attention layer's kernel sites tapped.

    Layer 0's attention output is tapped as well and layer 5's is not, so both
    ways a replayed step counts the visit a kernel site is scheduled by are run.
    """
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        "gpt2",
        gpu_memory_utilization=0.1,
        dispatch=True,
        taps=["transformer.h.*.attn.attn.kernel", "transformer.h.0.attn.attn.output", "transformer.h.7.output"],
    )


def _steps(model, prompt, layers=(0, 5), count=3):
    with model.trace(prompt, max_tokens=count, temperature=0.0) as tracer:
        steps = list().save()
        for _ in tracer.iter[:count]:
            row = []
            for layer in layers:
                attn = model.transformer.h[layer].attn.attn
                row.append(attn.kernel.probs().clone())
                row.append(attn.kernel.lse(query=-1, heads=[3, 9]).clone())
            steps.append(row)
    return steps


def test_taps_serve_kernel_sites_on_every_replayed_step(vllm_gpt2_kernel_taps, vllm_gpt2):
    assert "model.transformer.h.5.attn.attn.kernel" in vllm_gpt2_kernel_taps.taps
    eager = _steps(vllm_gpt2, PROMPT)
    tapped = _steps(vllm_gpt2_kernel_taps, PROMPT)
    assert len(tapped) == len(eager) == 3
    for expected_row, row in zip(eager, tapped):
        for expected, value in zip(expected_row, row):
            assert value.shape == expected.shape
            assert torch.allclose(value, expected, atol=1e-5)


def test_taps_serve_a_kernel_site_beside_the_module_output(vllm_gpt2_kernel_taps):
    attn = vllm_gpt2_kernel_taps.transformer.h[0].attn.attn
    with vllm_gpt2_kernel_taps.trace(PROMPT, max_tokens=3, temperature=0.0) as tracer:
        keys = list().save()
        for _ in tracer.iter[:3]:
            probs = attn.kernel.probs(query=-1)
            attn.output.clone()
            keys.append(probs.shape[-1])
    assert keys[1] == keys[0] + 1 and keys[2] == keys[0] + 2


def test_taps_serve_each_invoke_its_own_request(vllm_gpt2_kernel_taps, vllm_gpt2):
    other = "Madison Square Garden is in"

    def read(model):
        attn = model.transformer.h[LAYER].attn.attn
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPT):
                first = attn.kernel.probs().save()
            with tracer.invoke(other):
                second = attn.kernel.probs().save()
        return first, second

    for expected, value in zip(read(vllm_gpt2), read(vllm_gpt2_kernel_taps)):
        assert value.shape == expected.shape
        assert torch.allclose(value, expected, atol=1e-5)
