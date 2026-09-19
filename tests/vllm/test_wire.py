"""The wire form of saved values: pack/unpack for the RPC hop, frame/unframe for serve."""

import collections
import pickle

import pytest
import torch

from nnsight.modeling.vllm.wire import Slot, frame, pack, raw, unframe, unpack

Point = collections.namedtuple("Point", "x y")


class Box:
    """A container pytree does not know: its tensor stays inside the pickle."""

    def __init__(self, tensor):
        self.tensor = tensor


def roundtrip(value, **kwargs):
    meta, buffers = pack(value)
    return unpack(meta, buffers, **kwargs), meta, buffers


class TestPack:
    def test_structure_and_dtypes_survive(self):
        value = {
            "a": [torch.arange(6, dtype=torch.int64).reshape(2, 3), None, 3, "s"],
            "b": Point(torch.ones(2, dtype=torch.bool), torch.randn(3).bfloat16()),
            ("k", 1): (torch.tensor(2.5), torch.empty(0)),
        }
        out, _, buffers = roundtrip(value)
        assert torch.equal(out["a"][0], value["a"][0]) and out["a"][1:] == [None, 3, "s"]
        assert isinstance(out["b"], Point) and out["b"].x.dtype == torch.bool
        assert torch.equal(out["b"].y, value["b"].y) and out["b"].y.dtype == torch.bfloat16
        assert out[("k", 1)][0].shape == () and out[("k", 1)][0].item() == 2.5
        assert out[("k", 1)][1].numel() == 0
        assert len(buffers) == 5

    def test_a_view_ships_only_its_own_elements(self):
        full = torch.randn(8, 50304)
        row = full[-1:, :]
        meta, buffers = pack({"row": row})
        assert buffers[0].untyped_storage().nbytes() == row.numel() * row.element_size()
        assert torch.equal(unpack(meta, buffers)["row"], row)
        assert len(meta) < 1000

    def test_non_contiguous_and_expanded_tensors(self):
        transposed = torch.randn(4, 6).t()
        expanded = torch.randn(1, 5).expand(3, 5)
        out, _, buffers = roundtrip([transposed, expanded])
        assert torch.equal(out[0], transposed) and torch.equal(out[1], expanded)
        assert all(b.is_contiguous() for b in buffers)

    def test_a_tensor_under_two_names_is_sent_once(self):
        tensor = torch.randn(3)
        out, _, buffers = roundtrip({"a": tensor, "b": tensor})
        assert len(buffers) == 1 and out["a"] is out["b"]

    def test_a_shared_object_keeps_its_identity(self):
        shared = Point(1, 2)
        out, _, _ = roundtrip({"a": shared, "b": shared, "c": [shared]})
        assert out["a"] is out["b"] is out["c"][0]

    def test_an_unregistered_container_still_travels(self):
        out, _, buffers = roundtrip(Box(torch.ones(2)))
        assert torch.equal(out.tensor, torch.ones(2)) and buffers == []

    def test_meta_is_a_pickle_of_slots(self):
        meta, buffers = pack(torch.zeros(2, 3))
        slot = pickle.loads(meta)
        assert isinstance(slot, Slot)
        assert (slot.index, slot.dtype, slot.shape, slot.nbytes) == (0, torch.float32, (2, 3), 24)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_device_is_recorded_and_restored(self):
        tensor = torch.randn(2, device="cuda")
        meta, buffers = pack({"t": tensor})
        assert buffers[0].device.type == "cpu" and buffers[0].is_pinned()
        assert unpack(meta, buffers)["t"].device == tensor.device
        assert unpack(meta, buffers, device="cpu")["t"].device.type == "cpu"


class TestRaw:
    def test_bytes_come_back_as_writable_tensors(self):
        value = {"x": torch.randn(3, 5), "y": torch.randn(4).bfloat16(), "z": torch.empty(0)}
        meta, buffers = pack(value)
        views = raw(buffers)
        assert [v.nbytes for v in views] == [60, 8, 0]
        out = unpack(meta, [bytes(v) for v in views], device="cpu")   # what msgpack hands back
        assert torch.equal(out["x"], value["x"]) and torch.equal(out["y"], value["y"])
        out["x"][0, 0] = 7.0
        assert out["x"][0, 0] == 7.0 and out["z"].numel() == 0


class TestFrame:
    def test_roundtrip_through_bytes(self):
        value = {
            "x": torch.randn(3, 5),
            "y": [torch.arange(4, dtype=torch.int16), None],
            "z": torch.randn(7).bfloat16(),
            "w": (torch.empty(0), torch.tensor(1)),
        }
        out = unframe(bytearray(frame(*pack(value))), device="cpu")
        for key in ("x", "z"):
            assert torch.equal(out[key], value[key])
        assert torch.equal(out["y"][0], value["y"][0]) and out["y"][1] is None
        assert out["w"][0].numel() == 0 and out["w"][1].item() == 1

    def test_views_from_a_bytearray_are_writable(self):
        out = unframe(bytearray(frame(*pack([torch.zeros(3), torch.zeros(5)]))), device="cpu")
        out[0][0] = 1.0
        assert out[0][0] == 1.0 and out[1].sum() == 0
