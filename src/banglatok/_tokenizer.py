"""Small, typed Python interface; all text processing stays in Go."""

import importlib
import os
from array import array
from collections.abc import Callable, Iterable, Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Literal, cast, overload

from ._hf import tokenizer_config, tokenizer_json
from ._native import _MAX_PAYLOAD, NativeHandle
from ._native import normalize as _normalize
from ._native import threads as _threads

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SPECIAL_TOKENS = {PAD_ID: "<pad>", UNK_ID: "<unk>", BOS_ID: "<s>", EOS_ID: "</s>"}

_BYTE_OFFSET = 4
_PIECE_OFFSET = 260
_SPACE_ID = _BYTE_OFFSET + 0x20

DEFAULT_MODEL = "bn-bpe-58k"


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return value


def _utf8(texts: Iterable[str]) -> tuple[bytes, "array[int]"]:
    """One UTF-8 payload for a batch, with each text's code-point count."""
    items = texts if type(texts) is list else list(texts)
    try:
        data = "".join(items).encode("utf-8")
    except (TypeError, UnicodeEncodeError):
        data = None
    if data is None:
        # Raise for the first invalid text, exactly as encoding one by one does.
        for text in items:
            _text(text).encode("utf-8")
        raise TypeError("texts must be strings")
    if 4 * len(items) + len(data) > _MAX_PAYLOAD:
        raise ValueError("encode input exceeds 2 GiB")
    return data, array("I", map(len, items))


def _padded_lists(ids: memoryview, rows: "array[int]") -> tuple[list[list[int]], list[list[int]]]:
    width = len(ids) // len(rows) if rows else 0
    if not width:
        return [[] for _ in rows], [[] for _ in rows]
    mask = [[1] * n + [0] * (width - n) for n in rows]
    return cast(list[list[int]], ids.cast("B").cast("I", (len(rows), width)).tolist()), mask


def _id_lists(ids: memoryview, rows: "array[int]") -> list[list[int]]:
    batch = []
    offset = 0
    for n in rows:
        batch.append(ids[offset : offset + n].tolist())
        offset += n
    return batch


def _ragged_lists(ids: memoryview, rows: "array[int]") -> tuple[list[list[int]], list[list[int]]]:
    return _id_lists(ids, rows), [[1] * n for n in rows]


def _arrays(np: Any, padded: bool) -> Callable[[memoryview, "array[int]"], tuple[Any, Any]]:
    def build(ids: memoryview, rows: "array[int]") -> tuple[Any, Any]:
        lengths = np.frombuffer(rows, dtype=np.uint32)
        if not padded and len(rows) and lengths.min() != lengths.max():
            raise ValueError("return_tensors='np' needs padding when sequence lengths differ")
        width = len(ids) // len(rows) if rows else 0
        values = np.frombuffer(ids, dtype=np.uint32).reshape(len(rows), width)
        mask = np.arange(width) < lengths[:, None]
        return values.astype(np.int64), mask.astype(np.int64)

    return build


def normalize(text: str) -> str:
    """Apply the model's NFC and whitespace normalization in Go."""
    return _normalize(_text(text))


def available_models() -> list[str]:
    """Names accepted by :meth:`Tokenizer.from_pretrained`, bundled in the wheel."""
    directory = files("banglatok").joinpath("pretrained")
    return sorted(
        entry.name.removesuffix(".json")
        for entry in directory.iterdir()
        if entry.name.endswith(".json")
    )


class Tokenizer:
    """Load a Go JSON model once; reuse it for single or batch encoding.

    Concurrent encode/decode calls are supported. Close only after concurrent
    work has finished. Prefer a context manager for deterministic native cleanup.
    """

    def __init__(self, model_path: str | os.PathLike[str]) -> None:
        path = os.fspath(model_path)
        if not isinstance(path, str) or "\0" in path:
            raise ValueError("model_path must be a string path without NUL characters")
        self._native = NativeHandle(path)
        self._tokens: list[str] | None = None
        self._ids: dict[str, int] | None = None

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_MODEL) -> "Tokenizer":
        """Load a model bundled with the package; no network access is used."""
        names = available_models()
        if name not in names:
            raise ValueError(f"unknown model {name!r}; available: {', '.join(names)}")
        resource = files("banglatok").joinpath(f"pretrained/{name}.json")
        # The Go loader reads the file during construction, so a temporary
        # extraction (for zipped installs) may be removed afterwards.
        with as_file(resource) as path:
            return cls(path)

    @property
    def vocab_size(self) -> int:
        """Total ID space: reserved controls, byte fallback, pieces and fused IDs."""
        return self._native.vocab_size

    @property
    def algorithm(self) -> str:
        """``"bpe"`` or ``"unigram"``."""
        return self._native.algorithm

    @property
    def closed(self) -> bool:
        return self._native.closed

    def encode(self, text: str) -> list[int]:
        """Encode one text into token IDs, without control IDs."""
        return self._native.encode_one(text)

    def encode_batch(
        self, texts: Sequence[str], *, num_threads: int | None = None
    ) -> list[list[int]]:
        """Encode a batch in one native call, preserving input order.

        Large batches are split across ``num_threads`` workers; the default is
        ``BANGLATOK_NUM_THREADS`` or every CPU. IDs never depend on it.
        """
        self._native.check_open()
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not a single string")
        threads = _threads(num_threads)
        data, runes = _utf8(texts)
        return self._native.encode_rows(data, runes, threads, _id_lists)

    def count_tokens(self, text: str) -> int:
        """Return ``len(encode(text))`` without building the ID list."""
        return self._native.count(text)

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        """Decode IDs to normalized text. Control IDs raise unless skipped."""
        self._native.check_open()
        values = list(ids)
        if any(type(value) is not int for value in values):
            raise TypeError("token IDs must be integers")
        if skip_special_tokens:
            values = [value for value in values if value not in SPECIAL_TOKENS]
        return self._native.decode(values)

    def decode_batch(
        self, batch: Sequence[Sequence[int]], *, skip_special_tokens: bool = False
    ) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def tokenize(self, text: str) -> list[str]:
        """Return token strings for ``encode(text)``; see :meth:`id_to_token`."""
        tokens = self._vocabulary()
        return [tokens[i] for i in self.encode(text)]

    def id_to_token(self, token_id: int) -> str:
        """Token text for an ID.

        Learned pieces are their text, fused IDs are ``" "`` plus their piece,
        the space byte is ``" "``, other fallback bytes are ``"<0xNN>"`` and
        control IDs are ``<pad>``, ``<unk>``, ``<s>`` and ``</s>``.
        """
        if type(token_id) is not int:
            raise TypeError("token ID must be an integer")
        tokens = self._vocabulary()
        if not 0 <= token_id < len(tokens):
            raise ValueError(f"token ID {token_id} is outside 0..{len(tokens) - 1}")
        return tokens[token_id]

    def token_to_id(self, token: str) -> int | None:
        """Inverse of :meth:`id_to_token`; learned pieces win over byte names."""
        return self._token_ids().get(_text(token))

    def get_vocab(self) -> dict[str, int]:
        """Mapping from token text to ID for every ID."""
        return dict(self._token_ids())

    def save_huggingface(self, directory: str | os.PathLike[str]) -> None:
        """Write ``tokenizer.json`` and ``tokenizer_config.json`` for Hugging Face.

        Load the directory with ``transformers.AutoTokenizer.from_pretrained`` or
        ``tokenizers.Tokenizer.from_file``; IDs and decoded text match this
        tokenizer. Only BPE models can be exported. With ``tokenizers`` alone, set
        ``encode_special_tokens = True`` so literal ``<s>`` text is not a control
        token; ``transformers`` does this from the saved configuration.
        """
        if self.algorithm != "bpe":
            raise ValueError("Hugging Face export supports BPE models only")
        _, fusion = self._native.vocabulary()
        document = tokenizer_json(self._vocabulary(), self._native.merges(), fusion)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text(document, encoding="utf-8")
        (target / "tokenizer_config.json").write_text(tokenizer_config(), encoding="utf-8")

    # str is itself a Sequence[str]; this overload wins because it is listed first,
    # matching the runtime isinstance(texts, str) check.
    @overload
    def __call__(  # type: ignore[overload-overlap]
        self,
        texts: str,
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool | Literal["longest", "max_length"] = False,
        num_threads: int | None = None,
        return_tensors: None = None,
    ) -> dict[str, list[int]]: ...

    @overload
    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool | Literal["longest", "max_length"] = False,
        num_threads: int | None = None,
        return_tensors: None = None,
    ) -> dict[str, list[list[int]]]: ...

    @overload
    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool | Literal["longest", "max_length"] = False,
        num_threads: int | None = None,
        return_tensors: Literal["np"],
    ) -> dict[str, Any]: ...

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool | Literal["longest", "max_length"] = False,
        num_threads: int | None = None,
        return_tensors: Literal["np"] | None = None,
    ) -> dict[str, list[int]] | dict[str, list[list[int]]] | dict[str, Any]:
        """Prepare encoder inputs: ``input_ids`` and ``attention_mask``.

        ``add_special_tokens`` wraps each sequence in ``<s>``/``</s>``;
        ``max_length`` truncates (keeping both controls); ``padding`` pads with
        ``<pad>`` to the longest sequence or to ``max_length``. ``num_threads``
        is as for :meth:`encode_batch`. ``return_tensors="np"`` returns 2-D
        int64 NumPy arrays, also for a single text, instead of lists.
        """
        single = isinstance(texts, str)
        if padding not in (False, True, "longest", "max_length"):
            raise ValueError("padding must be False, True, 'longest' or 'max_length'")
        if max_length is not None and (
            type(max_length) is not int or max_length < (2 if add_special_tokens else 1)
        ):
            raise ValueError("max_length is too small")
        if padding == "max_length" and max_length is None:
            raise ValueError("padding='max_length' requires max_length")
        if return_tensors not in (None, "np"):
            raise ValueError("return_tensors must be None or 'np'")
        width = 0
        if padding == "max_length":
            width = max_length or 0
        elif padding:
            width = -1
        build: Callable[[memoryview, array[int]], tuple[Any, Any]]
        if return_tensors == "np":
            try:
                numpy: Any = importlib.import_module("numpy")
            except ImportError:
                raise ImportError("return_tensors='np' requires NumPy: pip install numpy") from None
            build = _arrays(numpy, padded=width != 0)
        elif width:
            build = _padded_lists
        else:
            build = _ragged_lists
        self._native.check_open()
        if isinstance(texts, bytes):
            raise TypeError("texts must be a sequence of strings, not a single string")
        threads = _threads(num_threads)
        data, runes = _utf8([texts] if single else texts)  # type: ignore[list-item]
        input_ids, attention_mask = self._native.encode_rows(
            data,
            runes,
            threads,
            build,
            max_length=max_length or 0,
            special=bool(add_special_tokens),
            width=width,
        )
        if single and return_tensors is None:
            return {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _vocabulary(self) -> list[str]:
        self._native.check_open()
        if self._tokens is None:
            pieces, fusion = self._native.vocabulary()
            tokens = [SPECIAL_TOKENS[i] for i in range(_BYTE_OFFSET)]
            tokens += [f"<0x{b:02X}>" for b in range(256)]
            tokens[_SPACE_ID] = " "
            tokens += pieces
            tokens += [" " + tokens[base] for base in fusion]
            if len(tokens) != self.vocab_size:
                raise RuntimeError("native vocabulary size mismatch")
            self._tokens = tokens
        return self._tokens

    def _token_ids(self) -> dict[str, int]:
        if self._ids is None:
            tokens = self._vocabulary()
            ids = {text: i for i, text in enumerate(tokens[:_PIECE_OFFSET])}
            ids.update({text: i for i, text in enumerate(tokens[_PIECE_OFFSET:], _PIECE_OFFSET)})
            self._ids = ids
        return self._ids

    def close(self) -> None:
        """Release the native model; repeated close calls are harmless."""
        self._native.close()

    def __enter__(self) -> "Tokenizer":
        self._native.check_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"{self.algorithm}, vocab_size={self.vocab_size}"
        return f"Tokenizer({state})"
