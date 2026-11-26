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
