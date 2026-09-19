"""Saved values on the wire.

A worker's saved values reach the engine's client as one utility-call result,
which vLLM msgpack-encodes. Byte strings are a native msgpack type, carried in
either serialization mode with one copy on each side; a pickled tensor costs a
device-to-host copy, a ``torch.save`` archive inside the pickle, and a full
replay of that archive on the client. So a value is sent as two parts:
``meta``, the value pickled with every tensor replaced by a :class:`Slot`, and
``buffers``, the tensors' bytes, each owning exactly its own elements.
[`pack`][nnsight.modeling.vllm.wire.pack] splits a value that way,
[`raw`][nnsight.modeling.vllm.wire.raw] turns the buffers into byte views for
the RPC, and [`unpack`][nnsight.modeling.vllm.wire.unpack] puts the value back
together, on the device the slot recorded or one the caller names.

The serve hop has no msgpack: [`frame`][nnsight.modeling.vllm.wire.frame] lays
the two parts into one byte string (the ``meta`` length, ``meta``, then each
buffer at a 64-byte-aligned offset) and
[`unframe`][nnsight.modeling.vllm.wire.unframe] reads them back as views.
"""

from __future__ import annotations

import pickle
import struct
from typing import Any, Optional, Sequence, Union

import torch

_ALIGN = 64
_HEADER = struct.Struct("<Q")


def _walk(x: Any, fn: Any, memo: dict) -> Any:
    """``fn`` over the leaves of ``x``, rebuilding dicts, lists, tuples and namedtuples.

    An object reached twice is rebuilt once (``memo`` by id), so what was one
    object stays one object: the pickle on the other side keeps shared
    references, and the engine's collect tells a value shared by several
    sequences from several values by identity.
    """
    key = id(x)
    if key in memo:
        return memo[key]
    kind = type(x)
    if kind is dict:
        out: Any = {}
        memo[key] = out
        out.update((k, _walk(v, fn, memo)) for k, v in x.items())
    elif kind is list:
        out = []
        memo[key] = out
        out.extend(_walk(v, fn, memo) for v in x)
    elif kind is tuple or (isinstance(x, tuple) and hasattr(kind, "_fields")):
        items = [_walk(v, fn, memo) for v in x]
        out = kind(*items) if kind is not tuple else tuple(items)
        memo[key] = out
    else:
        out = fn(x)
        memo[key] = out
    return out


class Slot:
    """Where a tensor went: its index in ``buffers``, and what it looked like."""

    def __init__(
        self, index: int, dtype: torch.dtype, shape: tuple, device: str, nbytes: int
    ) -> None:
        self.index = index
        self.dtype = dtype
        self.shape = shape
        self.device = device
        self.nbytes = nbytes


def pack(value: Any) -> tuple[bytes, list[torch.Tensor]]:
    """Split ``value`` into its pickled structure and its tensors as CPU buffers.

    Each buffer owns exactly the bytes of the tensor it stands for: a view onto
    a larger storage (``logits[-1:]``) ships its elements, not the storage
    pickling would drag along. A tensor reachable under two names is sent once.
    Device tensors are copied into pinned host memory on the current stream and
    waited for together.
    """
    buffers: list[torch.Tensor] = []
    pending = False

    def leaf(x: Any) -> Any:
        nonlocal pending
        if not isinstance(x, torch.Tensor):
            return x
        tensor = x.detach().contiguous()
        if tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
            tensor = tensor.clone()
        if tensor.device.type != "cpu":
            host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            tensor, pending = host, True
        slot = Slot(
            len(buffers), x.dtype, tuple(x.shape), str(x.device),
            tensor.numel() * tensor.element_size(),
        )
        buffers.append(tensor)
        return slot

    meta = _walk(value, leaf, {})
    if pending:
        torch.cuda.synchronize()
    return pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL), buffers


def _rebuild(tree: Any, buffers: Sequence[Any], device: Optional[Union[str, torch.device]]) -> Any:
    def leaf(x: Any) -> Any:
        if not isinstance(x, Slot):
            return x
        data = buffers[x.index]
        if isinstance(data, torch.Tensor):
            tensor = data
        elif not x.nbytes:
            tensor = torch.empty(0, dtype=x.dtype)
        else:
            # A writable view is used in place; anything else (the bytes msgpack
            # decoded, a read-only view) is copied once so the tensor owns
            # memory it may write.
            if not (isinstance(data, memoryview) and not data.readonly):
                data = bytearray(data)
            tensor = torch.frombuffer(data, dtype=x.dtype)
        tensor = tensor.reshape(x.shape)
        target = x.device if device is None else str(device)
        if str(tensor.device) != target:
            tensor = tensor.to(target)
        return tensor

    return _walk(tree, leaf, {})


def unpack(meta: bytes, buffers: Sequence[Any], device: Optional[Union[str, torch.device]] = None) -> Any:
    """The value ``pack`` split, with each slot filled from ``buffers``.

    A buffer is a tensor, the bytes the RPC hands back, or a view ``unframe``
    cut. Tensors land on the device the slot recorded, or on ``device`` when
    given.
    """
    return _rebuild(pickle.loads(meta), buffers, device)


def raw(buffers: Sequence[torch.Tensor]) -> list[memoryview]:
    """Each buffer's bytes as a view, for msgpack to carry natively."""
    return [memoryview(tensor.reshape(-1).view(torch.uint8).numpy()) for tensor in buffers]


def frame(meta: bytes, buffers: Sequence[torch.Tensor]) -> bytes:
    """One byte string: the ``meta`` length, ``meta``, then each buffer 64-byte aligned."""
    out = bytearray(_HEADER.pack(len(meta)))
    out += meta
    for tensor in buffers:
        out += bytes(-len(out) % _ALIGN)
        out += tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return bytes(out)


def unframe(content: Union[bytes, bytearray, memoryview], device: Optional[Union[str, torch.device]] = None) -> Any:
    """The value ``frame`` wrote, its tensors as views into ``content``.

    Pass a ``bytearray`` for writable tensors; ``bytes`` gives read-only views.
    """
    view = memoryview(content)
    (length,) = _HEADER.unpack_from(view, 0)
    start = _HEADER.size
    tree = pickle.loads(view[start : start + length])
    slots: dict[int, Slot] = {}
    _walk(tree, lambda x: slots.setdefault(x.index, x) if isinstance(x, Slot) else x, {})
    buffers = []
    position = start + length
    for index in range(len(slots)):
        position += -position % _ALIGN
        buffers.append(view[position : position + slots[index].nbytes])
        position += slots[index].nbytes
    return _rebuild(tree, buffers, device)
