"""Locate and open the compiled ground-truth libraries exactly once per process.

The prototype called ``ctypes.CDLL('./libfoo.so')`` *inside* every age
evaluation, which paid a dlopen on each of ~500k samples and broke whenever the
working directory was not the project root.  Here each library is opened at
import time and cached.
"""
from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

_BUILD_HINT = "cd native && make"

# <repo>/src/chai/forward/native/_loader.py -> <repo>
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _search_dirs() -> list[Path]:
    dirs = []
    if env := os.environ.get("CHAI_NATIVE_DIR"):
        dirs.append(Path(env))
    dirs.append(_REPO_ROOT / "native" / "build")
    return dirs


@cache
def load(libname: str) -> ctypes.CDLL:
    """Open ``libname`` (e.g. ``"libketcham.so"``), caching the handle."""
    tried = []
    for d in _search_dirs():
        candidate = d / libname
        tried.append(candidate)
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    raise FileNotFoundError(
        f"Could not find {libname}. Looked in:\n"
        + "\n".join(f"  {p}" for p in tried)
        + f"\n\nBuild the native models with:\n  {_BUILD_HINT}\n"
        "or point CHAI_NATIVE_DIR at a directory containing them."
    )
