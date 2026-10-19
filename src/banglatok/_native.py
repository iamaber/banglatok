"""Owned-buffer C interfaces; the shared library loads on first use."""

import ctypes
import json
import os
import struct
import sys
import threading
import weakref
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, cast


class Merge(TypedDict):
    left: int
    right: int
    result: int


class Response(TypedDict, total=False):
    handle: int
    vocabulary: int
    algorithm: str
    pieces: list[str]
    space_fusion: list[int]
    merges: list[Merge]
    text: str
    error: str


# Status codes returned by bntok_encode_into instead of an ID count.
_CLOSED = -1
_INVALID = -2
_MAX_PAYLOAD = 2**31 - 1


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    path = Path(__file__).with_name("_go" + suffix)
    lib = ctypes.CDLL(str(path))
    lib.bntok_call.argtypes = [ctypes.c_char_p]
    # Preserve the allocated address until bntok_free; c_char_p would copy and lose it.
    lib.bntok_call.restype = ctypes.c_void_p
    lib.bntok_free.argtypes = [ctypes.c_void_p]
    lib.bntok_free.restype = None
    lib.bntok_encode.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.bntok_encode.restype = ctypes.c_void_p
    lib.bntok_encode_into.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.bntok_encode_into.restype = ctypes.c_longlong
    return lib


_FORKED = (
    "banglatok cannot run in a child process forked after the parent loaded it; "
    "use the 'spawn' or 'forkserver' multiprocessing start method"
)
_forked = False


def _after_fork_in_child() -> None:
    # The Go runtime's threads do not survive fork(); any call would hang.
    global _forked
    _forked = _forked or library.cache_info().currsize > 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _check_process() -> None:
    if _forked:
        raise RuntimeError(_FORKED)


def call(lib: ctypes.CDLL, request: dict[str, object]) -> Response:
    _check_process()
    payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
    pointer = lib.bntok_call(payload)
    if not pointer:
        raise MemoryError("Go tokenizer returned a null response")
    try:
        response = cast(Response, json.loads(ctypes.string_at(pointer)))
    finally:
        lib.bntok_free(pointer)
    if error := response.get("error"):
        raise ValueError(error)
    return response


def release(lib: ctypes.CDLL, handle: int) -> None:
    if not _forked:
        call(lib, {"op": "close", "handle": handle})


def normalize(text: str) -> str:
    return call(library(), {"op": "normalize", "text": text})["text"]


class _Buffers(threading.local):
    """Per-thread output buffer: ctypes releases the GIL during native calls."""

    def __init__(self) -> None:
        self.grow(1024)

    def grow(self, size: int) -> None:
        self.capacity = size
        self.ids = (ctypes.c_uint32 * size)()
        self.address = ctypes.addressof(self.ids)


class NativeHandle:
    """Own a loaded model and keep its wire formats private to this module."""

    def __init__(self, path: str) -> None:
        self._lib = library()
        response = call(self._lib, {"op": "load", "path": path})
        self._handle = response["handle"]
        self.vocab_size = response["vocabulary"]
        self.algorithm = response.get("algorithm", "")
        self._encode_into = self._lib.bntok_encode_into
        self._buffers = _Buffers()
        self._finalizer = weakref.finalize(self, release, self._lib, self._handle)

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def check_open(self) -> None:
        _check_process()
        if not self._finalizer.alive:
            raise RuntimeError("tokenizer is closed")

    def encode_one(self, data: bytes) -> list[int]:
        """Encode one UTF-8 payload straight into a reusable per-thread buffer."""
        buffers = self._buffers
        count = self._encode_into(self._handle, data, len(data), buffers.address, buffers.capacity)
        if count > buffers.capacity:
            buffers.grow(max(count, 2 * buffers.capacity))
            count = self._encode_into(
                self._handle, data, len(data), buffers.address, buffers.capacity
            )
        if count < 0:
            if count == _CLOSED:
                raise ValueError("unknown or closed tokenizer handle")
            raise ValueError("encode input exceeds 2 GiB")
        return buffers.ids[:count]

    def count(self, data: bytes) -> int:
        """Return the number of IDs without materializing them in Python."""
        count: int = self._encode_into(self._handle, data, len(data), None, 0)
        if count < 0:
            if count == _CLOSED:
                raise ValueError("unknown or closed tokenizer handle")
            raise ValueError("encode input exceeds 2 GiB")
        return count

    def encode(self, texts: list[bytes]) -> list[list[int]]:
        return encode(self._lib, self._handle, texts)

    def decode(self, ids: list[int]) -> str:
        return call(self._lib, {"op": "decode", "handle": self._handle, "ids": ids})["text"]

    def vocabulary(self) -> tuple[list[str], list[int]]:
        response = call(self._lib, {"op": "pieces", "handle": self._handle})
        return response.get("pieces", []), response.get("space_fusion", [])

    def merges(self) -> list[tuple[int, int]]:
        """BPE merges in rank order as (left, right) piece indices."""
        response = call(self._lib, {"op": "merges", "handle": self._handle})
        return [(merge["left"], merge["right"]) for merge in response.get("merges", [])]

    def close(self) -> None:
        self._finalizer()


def encode(lib: ctypes.CDLL, handle: int, texts: list[bytes]) -> list[list[int]]:
    """Length-prefixed UTF-8 input; native-endian uint32 status, counts, then IDs."""
    size = 4 * len(texts) + sum(map(len, texts))
    if size > _MAX_PAYLOAD:
        raise ValueError("encode input exceeds 2 GiB")
    payload = b"".join([part for data in texts for part in (struct.pack("<I", len(data)), data)])
    length = ctypes.c_size_t()
    pointer = lib.bntok_encode(handle, payload, len(payload), ctypes.byref(length))
    if not pointer:
        raise MemoryError("Go tokenizer returned a null response")
    try:
        response = ctypes.string_at(pointer, length.value)
    finally:
        lib.bntok_free(pointer)
    if len(response) < 4:
        raise RuntimeError("invalid native encode response")
    if struct.unpack_from("=I", response)[0]:
        raise ValueError(response[4:].decode("utf-8"))
    if len(response) % 4 or len(response) < 4 * (1 + len(texts)):
        raise RuntimeError("truncated native encode response")
    words = memoryview(response).cast("I")
    counts = words[1 : 1 + len(texts)].tolist()
    if sum(counts) != len(words) - 1 - len(texts):
        raise RuntimeError("native encode response count mismatch")
    result = []
    offset = 1 + len(texts)
    for count in counts:
        result.append(words[offset : offset + count].tolist())
        offset += count
    return result
