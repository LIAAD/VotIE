"""Where the FastText binary lives, agreed between the downloader and the loader.

`src/models/bilstm_crf.py` reads the Portuguese FastText vectors from here.
Download them from FASTTEXT_URL and place the unpacked binary at:

    <cache_root>/hub/models--microsoft--mdeberta-v3-base/...   (transformers)
    <cache_root>/fasttext/cc.pt.300.bin                        (this module)

This module stays importable without torch, transformers or fasttext.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

FASTTEXT_FILENAME = "cc.pt.300.bin"
FASTTEXT_SUBDIR = "fasttext"

# Facebook's official Portuguese CommonCrawl vectors (300d).
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz"
FASTTEXT_APPROX_GZ_BYTES = 4_500_000_000
FASTTEXT_APPROX_BIN_BYTES = 7_200_000_000

# Explicit override, checked before anything else.
FASTTEXT_PATH_ENV = "VOTIE_FASTTEXT_PATH"

# Falls back to <repo_parent>/hf_cache.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = _REPO_ROOT.parent / "hf_cache"


def cache_root() -> Path:
    """The shared model cache root: $HF_HOME if set, else <repo_parent>/hf_cache."""
    env = os.environ.get("HF_HOME")
    return Path(env).expanduser() if env else DEFAULT_CACHE_ROOT


def fasttext_path(root: Optional[Union[str, Path]] = None) -> Path:
    """Canonical location of the FastText binary under a given cache root."""
    base = Path(root).expanduser() if root is not None else cache_root()
    return base / FASTTEXT_SUBDIR / FASTTEXT_FILENAME


def fasttext_candidates(explicit: Optional[Union[str, Path]] = None) -> list[Path]:
    """Every location worth checking, in priority order.

    Deliberate choices beat conventions. Note that a *bare* filename in the YAML
    (every config ships `fasttext_path: cc.pt.300.bin`) is a placeholder, not a
    CWD-relative path — resolving it against the working directory would make
    the answer depend on where the job was launched from, so it is tried last.
    """
    candidates: list[Path] = []
    explicit_path = Path(explicit).expanduser() if explicit else None
    # "has a directory component" is what separates a real choice from the
    # placeholder: configs/…yaml say "cc.pt.300.bin", a user says "/data/ft.bin".
    explicit_is_specific = explicit_path is not None and explicit_path.parent != Path(".")

    if explicit_is_specific:
        candidates.append(explicit_path)

    env = os.environ.get(FASTTEXT_PATH_ENV)
    if env:
        candidates.append(Path(env).expanduser())

    candidates.append(fasttext_path())
    if os.environ.get("HF_HOME"):  # also try the default when HF_HOME points elsewhere
        candidates.append(fasttext_path(DEFAULT_CACHE_ROOT))

    # Legacy location, where older checkouts kept the binary.
    candidates.append(_REPO_ROOT / "src" / "models" / FASTTEXT_FILENAME)

    if explicit_path is not None and not explicit_is_specific:
        candidates.append(_REPO_ROOT / "src" / "models" / explicit_path)
        candidates.append(explicit_path)  # CWD-relative, last resort

    return list(dict.fromkeys(candidates))


def resolve_fasttext(explicit: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """First existing candidate, or None."""
    for candidate in fasttext_candidates(explicit):
        if candidate.exists():
            return candidate
    return None
