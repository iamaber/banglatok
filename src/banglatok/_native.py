"""Owned-buffer C interfaces; the shared library loads on first use."""

import ctypes
import json
import os
import sys
import threading
import weakref
from array import array
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, TypeVar, cast

T = TypeVar("T")


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


# Status codes returned by bntok_encode_into and bntok_encode_batch instead of an ID count.
_CLOSED = -1
_INVALID = -2
_TOO_LARGE = -4
# Per-thread batch buffers grow to at most this many IDs; larger batches use
# a native allocation for the call.
_MAX_KEPT_IDS = 1 << 20
_MAX_PAYLOAD = 2**31 - 1
# Batch worker limit; more workers than CPUs only add scheduling.
_MAX_THREADS = 1024


def threads(value: int | None) -> int:
    """Resolve num_threads; 0 asks Go for every available CPU."""
    if value is None:
        setting = os.environ.get("BANGLATOK_NUM_THREADS", "").strip()
        if not setting:
            return 0
        try:
            value = int(setting)
        except ValueError:
            raise ValueError("BANGLATOK_NUM_THREADS must be a positive integer") from None
        if value < 1:
            raise ValueError("BANGLATOK_NUM_THREADS must be a positive integer")
    elif type(value) is not int or value < 1:
        raise ValueError("num_threads must be a positive integer or None")
    return min(value, _MAX_THREADS)


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
    lib.bntok_encode_into.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.bntok_encode_into.restype = ctypes.c_longlong
    lib.bntok_encode_batch.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.bntok_encode_batch.restype = ctypes.c_longlong
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


class _IDs:
    """A reusable uint32 output buffer."""

    def __init__(self, size: int) -> None:
        self.ids = (ctypes.c_uint32 * size)()
        self.view = memoryview(self.ids).cast("B").cast("I")
        self.address = ctypes.addressof(self.ids)
        self.size = size


class _Buffers(threading.local):
    """Per-thread output buffers: ctypes releases the GIL during native calls."""

    def __init__(self) -> None:
        self.one = _IDs(1024)
        self.batch: _IDs | None = _IDs(1024)


def _encode_error(status: int) -> ValueError:
    if status == _CLOSED:
        return ValueError("unknown or closed tokenizer handle")
    return ValueError("encode input exceeds 2 GiB")


def _batch_error(status: int) -> Exception:
    if status == _CLOSED:
        return ValueError("unknown or closed tokenizer handle")
    if status == _TOO_LARGE:
        return MemoryError("encoded batch is too large")
    return RuntimeError("invalid native batch request")


class NativeHandle:
    """Own a loaded model and keep its wire formats private to this module."""

    def __init__(self, path: str) -> None:
        self._lib = library()
        response = call(self._lib, {"op": "load", "path": path})
        self._handle = response["handle"]
        self.vocab_size = response["vocabulary"]
        self.algorithm = response.get("algorithm", "")
        self._encode_into = self._lib.bntok_encode_into
        self._encode_batch = self._lib.bntok_encode_batch
        self._buffers = _Buffers()
        self._finalizer = weakref.finalize(self, release, self._lib, self._handle)

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def check_open(self) -> None:
        _check_process()
        if not self._finalizer.alive:
            raise RuntimeError("tokenizer is closed")

    def _payload(self, text: str) -> bytes:
        """UTF-8 for the per-text hot paths, which leave the closed-handle check
        to the native call but raise exactly as check_open plus validation."""
        if _forked:
            raise RuntimeError(_FORKED)
        if not isinstance(text, str):
            self.check_open()
            raise TypeError("text must be a string")
        try:
            return text.encode("utf-8")
        except UnicodeEncodeError:
            if not self._finalizer.alive:
                raise RuntimeError("tokenizer is closed") from None
            raise

    def encode_one(self, text: str) -> list[int]:
        """Encode one text straight into a reusable per-thread buffer."""
        data = self._payload(text)
        ids = self._buffers.one
        count: int = self._encode_into(self._handle, data, len(data), ids.address, ids.size)
        if count > ids.size:
            ids = self._buffers.one = _IDs(max(count, 2 * ids.size))
            count = self._encode_into(self._handle, data, len(data), ids.address, ids.size)
        if count < 0:
            self.check_open()
            raise _encode_error(count)
        return ids.view[:count].tolist()

    def count(self, text: str) -> int:
        """Return the number of IDs without materializing them in Python."""
        data = self._payload(text)
        count: int = self._encode_into(self._handle, data, len(data), None, 0)
        if count < 0:
            self.check_open()
            raise _encode_error(count)
        return count

    def encode_rows(
        self,
        data: bytes,
        runes: "array[int]",
        threads: int,
        build: Callable[[memoryview, "array[int]"], T],
        *,
        max_length: int = 0,
        special: bool = False,
        width: int = 0,
    ) -> T:
        """Encode ``data`` cut into texts of ``runes[i]`` code points.

        ``build`` receives the uint32 IDs and, in ``runes``, each row's unpadded
        length. Rows are truncated to ``max_length`` IDs (0 keeps all), wrapped
        in ``<s>``/``</s>`` with ``special`` and padded to ``width`` (-1: the
        longest row, 0: unpadded).
        """
        if 4 * len(runes) + len(data) > _MAX_PAYLOAD:
            raise ValueError("encode input exceeds 2 GiB")
        buffers = self._buffers
        # Borrow the buffer so a re-entrant call on this thread cannot overwrite it.
        out, buffers.batch = buffers.batch or _IDs(1024), None
        allocated = ctypes.c_void_p()
        size: int = self._encode_batch(
            self._handle,
            data,
            len(data),
            runes.buffer_info()[0],
            len(runes),
            threads,
            max_length,
            special,
            width,
            out.address,
            out.size,
            ctypes.byref(allocated),
        )
        if allocated.value is None:
            try:
                if size < 0:
                    raise _batch_error(size)
                return build(out.view[:size], runes)
            finally:
                buffers.batch = out
        try:
            ids = (ctypes.c_uint32 * size).from_address(allocated.value)
            return build(memoryview(ids).cast("B").cast("I"), runes)
        finally:
            self._lib.bntok_free(allocated)
            if size <= _MAX_KEPT_IDS:
                out = _IDs(min(max(size, 2 * out.size), _MAX_KEPT_IDS))
            buffers.batch = out

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
