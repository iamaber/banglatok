"""Fast, lossless subword tokenizers for Bangla, English and Banglish (romanized Bengali)."""

from typing import TYPE_CHECKING

from ._tokenizer import (
    BOS_ID,
    DEFAULT_MODEL,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    Tokenizer,
    available_models,
    normalize,
)

if TYPE_CHECKING:
    __version__: str


def __getattr__(name: str) -> str:
    # importlib.metadata costs more than the rest of the import; load it on demand.
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib.metadata import PackageNotFoundError, version

    global __version__
    try:
        __version__ = version("banglatok")
    except PackageNotFoundError:  # Imported from a source tree without installation.
        __version__ = "0+unknown"
    return __version__


__all__ = [
    "BOS_ID",
    "DEFAULT_MODEL",
    "EOS_ID",
    "PAD_ID",
    "SPECIAL_TOKENS",
    "UNK_ID",
    "Tokenizer",
    "__version__",
    "available_models",
    "normalize",
]
