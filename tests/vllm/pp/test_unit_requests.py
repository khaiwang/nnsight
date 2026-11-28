"""The request table's exit: what the engine lets go of when a request leaves;
and what an earlier pipeline stage sends home at collect."""

import pickle
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from nnsight.intervention.interleaver import Mediator
from nnsight.modeling.vllm.model_runners.GPUModelRunner import (
    NNsightGPUModelRunner,
    Request,
    Requests,
)
from nnsight.modeling.vllm.model_runners import GPUModelRunner as runner_module


def _registered(requests, request_id, registration_id, lcls):
    """A request nobody traced, carrying one copy of a registered block."""
    requests.templates[registration_id] = object()
    request = Request(request_id)
    copy = Mediator(compile("pass", "<registered>", "exec"), {}, lcls)
    copy.batch_group = [0, 1]  # scheduled this step, so the step records it
    request.copies[registration_id] = copy
    requests.requests[request.id] = request
    return request


def test_a_request_that_leaves_through_harvest_is_released():
    """An untraced request whose registered block saved nothing leaves when the
    scheduler reports it finished, before any collect; whatever else the engine
    keeps under its id is let go then too."""
    released = []
    requests = Requests(release=released.append)
    request = _registered(requests, "chat-00c0ffee", "quiet-edit", {})

    requests.harvest({request.id})

    assert request.id not in requests.requests
    assert released == [request.id]


def test_a_stage_whose_saves_are_not_needed_sends_none_home(monkeypatch):
    """A stage before the last whose caches observed nothing replies with its
    request's entry but no saves: the last stage's copy of the block holds
    every value, and the engine keeps that one."""
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: SimpleNamespace(is_first_rank=True))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)  # runs without a GPU
    runner = object.__new__(NNsightGPUModelRunner)
    runner.nnsight_requests = Requests()
    runner.nnsight_model = SimpleNamespace(interleaver=SimpleNamespace(taps=frozenset()))
    runner._reports_saves = lambda mediator: False  # a stage before the last, no cache here
    runner._parked_on_later_stage = lambda mediator: False
    down_proj_weight = torch.zeros(3)
    request = Request("probe-0000abcd")
    request.mediator = Mediator(compile("pass", "<traced>", "exec"), {}, {"down_w": down_proj_weight})
    request.mediator.saves = {id(down_proj_weight)}
    runner.nnsight_requests.requests[request.id] = request

    collected = pickle.loads(runner.collect_nnsight([request.id], [request.id]))

    assert collected["probe"]["saves"] == {}
    assert request.id not in runner.nnsight_requests.requests
