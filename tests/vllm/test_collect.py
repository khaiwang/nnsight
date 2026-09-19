"""What crosses from the worker when a trace saves, measured on a live engine."""

import torch

import nnsight.modeling.vllm.engines.engine as engine_module


def _shipped_bytes(monkeypatch):
    seen = []
    real = engine_module.merge_collected

    def counting(payloads):
        for payload in payloads:
            if payload is not None:
                seen.append(sum(memoryview(b).nbytes for b in payload["buffers"]))
        return real(payloads)

    monkeypatch.setattr(engine_module, "merge_collected", counting)
    return seen


def test_a_saved_slice_ships_only_its_own_bytes(vllm_gpt2, ET_prompt, monkeypatch):
    seen = _shipped_bytes(monkeypatch)
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, max_tokens=1):
        row = vllm_gpt2.logits[-1:, :].save()
    assert row.shape[0] == 1
    assert sum(seen) == row.numel() * row.element_size()


def test_a_saved_value_comes_back_on_its_device(vllm_gpt2, ET_prompt):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, max_tokens=1):
        hidden = vllm_gpt2.transformer.h[5].output.save()
        total = hidden.float().sum().save()
    assert hidden.device.type == "cuda" and total.device.type == "cuda"
    assert torch.isclose(hidden.float().sum(), total)
