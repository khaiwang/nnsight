"""Expert kernel sites on vLLM's fused MoE kernel (DeepSeek-V2-Lite), checked against the layer's own weights.

The reference is computed inside the trace in fp32: each routed pair's
activation and down projection from the routed experts' weights, and the shared
experts from theirs. vLLM's kernel runs in bf16, so the two agree to bf16
rounding; a row of the wrong token, slot or expert differs by the size of the row.

This vLLM build's MLA attention and its rotary embedding op need FlashInfer
kernels compiled with a CUDA toolkit newer than this machine's, so the engine
runs with both turned off. Neither touches the MoE layers.
"""

import os

import pytest
import torch

REPO = "deepseek-ai/DeepSeek-V2-Lite"
LONG = "The Eiffel Tower is located in the city of Paris and " * 3
SHORT = "The capital of Japan is"


#: Values the one-rank engine read, for the expert-parallel engine to compare against.
ONE_RANK: dict = {}


@pytest.fixture(scope="module", params=["one_rank", "expert_parallel"])
def engine(request):
    from nnsight.modeling.vllm import VLLM

    os.environ["VLLM_MLA_DISABLE"] = "1"
    kwargs = {}
    if request.param == "expert_parallel":
        if torch.cuda.device_count() < 2:
            pytest.skip("expert parallelism needs two GPUs")
        # Two ranks, each holding half of every layer's routed experts.
        kwargs = {"tensor_parallel_size": 2, "enable_expert_parallel": True}
    return VLLM(
        REPO,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        trust_remote_code=True,
        dispatch=True,
        compilation_config={"custom_ops": ["-rotary_embedding"]},
        **kwargs,
    )


def test_rows_match_the_layers_weights(engine, request):
    if "expert_parallel" in request.node.name:
        pytest.skip("a rank holds half of the experts' weights, so the reference is computed on one rank")
    mlp = engine.model.layers[3].mlp
    experts = mlp.experts
    with engine.trace(temperature=0.0, max_tokens=2) as tracer:
        with tracer.invoke(SHORT):
            other = list().save()
            for _ in tracer.iter[:2]:
                other.append(experts.kernel.expert_input().shape[0])
        with tracer.invoke(LONG):
            steps = list().save()
            for _ in tracer.iter[:2]:
                x = experts.inputs[1]["hidden_states"].float()
                logits = mlp.gate.output[0].float()
                kernel = experts.kernel
                given, activation, output = kernel.expert_input(), kernel.expert_activation(), kernel.expert_output()
                top_k = activation.shape[1]
                ranked = torch.softmax(logits, -1).topk(top_k + 1, dim=-1)
                weights, ids = ranked.values[:, :top_k], ranked.indices[:, :top_k]
                # vLLM routes on bf16 scores, so two experts with nearly equal scores can come in the other
                # order, or swap across the top-k boundary. Rows are compared where the fp32 order is clear.
                clear = (ranked.values[:, :-1] - ranked.values[:, 1:]).amin(-1) > 1e-3
                w13, w2 = experts.routed_experts.w13_weight.data.float(), experts.routed_experts.w2_weight.data.float()
                gate, up = torch.einsum("th,tsih->tsi", x, w13[ids]).chunk(2, dim=-1)
                expected_activation = torch.nn.functional.silu(gate) * up
                expected_output = torch.einsum("tsi,tshi->tsh", expected_activation, w2[ids])
                gate, up = (x @ mlp.shared_experts.gate_up_proj.weight.data.float().T).chunk(2, dim=-1)
                shared = (torch.nn.functional.silu(gate) * up) @ mlp.shared_experts.down_proj.weight.data.float().T
                total = experts.output.float()
                summed = (weights.unsqueeze(-1) * output.float()).sum(1) + shared

                def relative(value, expected, rows=slice(None)):
                    return float(((value.float() - expected).abs().amax(-1) / expected.abs().amax(-1))[rows].max())

                steps.append(
                    {
                        "tokens": x.shape[0],
                        "input": bool((given.float() == x.unsqueeze(1)).all()),
                        "clear": int(clear.sum()),
                        "activation": relative(activation, expected_activation, clear),
                        "output": relative(output, expected_output, clear),
                        # The module's output is the weighted sum of the routed rows plus the shared experts.
                        "sum": relative(summed, total),
                    }
                )
    assert other == [5, 1]
    assert [step["tokens"] for step in steps] == [37, 1]
    assert steps[0]["clear"] >= 15 and steps[1]["clear"] == 1
    for step in steps:
        assert step["input"]
        assert step["activation"] < 3e-2 and step["output"] < 3e-2 and step["sum"] < 3e-2


def test_selection(engine):
    mlp = engine.model.layers[3].mlp
    kernel = mlp.experts.kernel
    with engine.trace(LONG, temperature=0.0, max_tokens=1):
        logits = mlp.gate.output[0].float().save()
        activation = kernel.expert_activation().clone().save()
        output = kernel.expert_output().float().save()
        some = kernel.expert_activation(tokens=-1, slot=[4, 1]).clone().save()
        rows = kernel.expert_output(tokens=[0, -1], slot=2).float().save()
        one = kernel.expert_output(expert=int(logits[0].argmax())).float().save()
    expert = int(logits[0].argmax())
    ids = logits.topk(activation.shape[1], dim=-1).indices
    assert activation.shape == (37, 6, 1408) and output.shape == (37, 6, 2048)
    assert torch.equal(some, activation[-1, [4, 1]])
    # A recomputed subset is a different bf16 matrix product from the full one.
    assert torch.allclose(rows, output[[0, -1], 2], rtol=2e-2, atol=2e-2 * float(output[[0, -1], 2].abs().max()))
    routed = (ids == expert).any(-1)
    assert routed[0] and (one[~routed] == 0).all()
    assert torch.allclose(one[0], output[0, 0], rtol=2e-2, atol=2e-2 * float(output[0, 0].abs().max()))


def test_expert_parallel_matches_one_rank(engine, request):
    """Every routed pair is held by exactly one rank, so the summed ranks equal the one-rank rows; a dropped pair would read 0."""
    mlp = engine.model.layers[3].mlp
    kernel = mlp.experts.kernel
    with engine.trace(temperature=0.0, max_tokens=2) as tracer:
        with tracer.invoke(SHORT):
            pass
        with tracer.invoke(LONG):
            steps = list().save()
            for _ in tracer.iter[:2]:
                logits = mlp.gate.output[0].float().clone()
                steps.append(
                    [
                        kernel.expert_input().float().clone(),
                        kernel.expert_activation().float().clone(),
                        kernel.expert_output().float().clone(),
                        kernel.expert_output(tokens=-1, slot=[4, 1]).float().clone(),
                        kernel.expert_activation(expert=int(logits[0].argmax())).float().clone(),
                    ]
                )
    steps = [[value.cpu() for value in step] for step in steps]
    if "one_rank" in request.node.name:
        ONE_RANK["steps"] = steps
        return
    # The two engines' layer inputs already differ by 2% of the largest entry.
    for step, expected in zip(steps, ONE_RANK["steps"]):
        for value, reference in zip(step, expected):
            assert value.shape == reference.shape
            assert float((value - reference).abs().max() / reference.abs().max()) < 5e-2
        assert (step[2].abs().amax(-1) > 0).all()
