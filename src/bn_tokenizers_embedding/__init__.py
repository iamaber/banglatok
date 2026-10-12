"""Fast, lossless Bangla, English and Banglish tokenizers with a native Go engine."""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("bn-tokenizers-embedding")
except PackageNotFoundError:  # Imported from a source tree without installation.
    __version__ = "0+unknown"

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
