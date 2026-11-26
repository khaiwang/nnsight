"""Attention kernel sites on vLLM's multi-head latent attention (DeepSeek-V2-Lite).

The reference is transformers' own DeepSeek-V2 with eager attention, run on the
prompt followed by the tokens vLLM generated: its rows for the prompt are the
prefill step, and each later row is one decode step. Both run in bf16 and round
their scores differently, so probabilities agree to a few hundredths; a row read
from the wrong request, head or key differs by the size of a probability.

The engine runs with the rotary embedding custom op turned off: on this vLLM
build it needs a FlashInfer kernel compiled with a newer CUDA toolkit.
"""

import glob
import os

import pytest
import torch

REPO = "deepseek-ai/DeepSeek-V2-Lite"
LONG = "The Eiffel Tower is located in the city of Paris and " * 3
SHORT = "The capital of Japan is"
LAYERS = (0, 3)
STEPS = 3

#: Values the one-rank engine read, for the tensor-parallel engine to compare against.
ONE_RANK: dict = {}


@pytest.fixture(scope="module", params=["one_rank", "tensor_parallel"])
def engine(request):
    from nnsight.modeling.vllm import VLLM

    kwargs = {}
    if request.param == "tensor_parallel":
        if torch.cuda.device_count() < 2:
            pytest.skip("tensor parallelism needs two GPUs")
        # Two ranks, each holding half of the heads.
        kwargs = {"tensor_parallel_size": 2}
    return VLLM(
        REPO,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        trust_remote_code=True,
        dispatch=True,
        compilation_config={"custom_ops": ["-rotary_embedding"]},
        **kwargs,
    )


def _attention(engine, index):
    return engine.model.layers[index].self_attn.mla_attn.mla_attn


def _read(engine):
    """Per step, the probabilities of `LAYERS` for `LONG`, read beside a second request; and the sampled tokens."""
    with engine.trace(temperature=0.0, max_tokens=STEPS) as tracer:
        with tracer.invoke(SHORT):
            other = list().save()
            for _ in tracer.iter[:STEPS]:
                other.append(_attention(engine, 0).kernel.probs().shape)
        with tracer.invoke(LONG):
            steps = list().save()
            tokens = list().save()
            for _ in tracer.iter[:STEPS]:
                steps.append([_attention(engine, index).kernel.probs().clone() for index in LAYERS])
                tokens.append(engine.samples.item())
    return [[value.float().cpu() for value in step] for step in steps], list(tokens), [tuple(shape) for shape in other]


def test_probabilities_match_hf_eager(engine, request):
    steps, tokens, other = _read(engine)
    ids = engine.tokenizer(LONG)["input_ids"]
    count = len(ids)
    assert other == [(5, 16, 5), (1, 16, 6), (1, 16, 7)]
    assert [tuple(step[0].shape) for step in steps] == [(count, 16, count), (1, 16, count + 1), (1, 16, count + 2)]
    if "one_rank" not in request.node.name:
        for step, expected in zip(steps, ONE_RANK["steps"]):
            for value, reference in zip(step, expected):
                assert float((value - reference).abs().max()) < 5e-2
        return
    ONE_RANK["steps"] = steps
    if torch.cuda.device_count() < 2:
        pytest.skip("the transformers reference needs a second GPU beside the engine")
    from transformers import DeepseekV2ForCausalLM

    path = glob.glob(os.path.join(os.environ["HF_HUB_CACHE"], "models--deepseek-ai--DeepSeek-V2-Lite", "snapshots", "*"))[0]
    reference = DeepseekV2ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda:1").eval()
    with torch.no_grad():
        result = reference(torch.tensor([ids + tokens[: STEPS - 1]], device="cuda:1"), output_attentions=True)
    assert result.logits[0, count - 1 :].argmax(-1).tolist() == tokens
    for step, values in enumerate(steps):
        rows = slice(0, count) if step == 0 else slice(count + step - 1, count + step)
        for layer, value in zip(LAYERS, values):
            # transformers lays attention out as [heads, query, key].
            expected = result.attentions[layer][0].float().cpu()[:, rows, : count + step].permute(1, 0, 2)
            assert float((value - expected).abs().max()) < 1e-1
    del reference, result
    torch.cuda.empty_cache()


def test_sites_agree_and_select(engine):
    kernel = _attention(engine, 3).kernel
    with engine.trace(LONG, temperature=0.0, max_tokens=2) as tracer:
        steps = list().save()
        for _ in tracer.iter[:2]:
            steps.append(
                [
                    kernel.lse().clone(),
                    kernel.scores().clone(),
                    kernel.probs().clone(),
                    kernel.probs(query=-1, heads=[11, 1], key=[0, -1]).clone(),
                    kernel.scores(heads=9, key=2).clone(),
                ]
            )
    for step, (lse, scores, probs, some, column) in enumerate(steps):
        # The prefill kernel's logsumexp is fp32; the decode kernel returns it in bf16.
        assert float((lse - torch.logsumexp(scores, -1)).abs().max()) < (1e-3 if step == 0 else 5e-2)
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-3)
        assert torch.allclose(some, probs[-1][[11, 1]][:, [0, -1]], atol=1e-5)
        assert torch.allclose(column, scores[:, 9, 2])
    scores = steps[0][1]
    later = torch.ones(scores.shape[0], scores.shape[2], dtype=torch.bool, device=scores.device).triu(1)
    assert torch.isinf(scores[:, 0][later]).all() and torch.isfinite(scores[:, 0][~later]).all()
