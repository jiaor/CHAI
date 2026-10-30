"""Studies, audits and validation reports.

The names below are imported on first use: :mod:`.study` runs the compiled
oracles, and :mod:`chai.api` (through :mod:`.calibration`) must import on an
installed wheel, which does not build them.
"""
import importlib
from typing import Any

_LAZY = {
    "REFERENCE": "closure",
    "ClosureReference": "closure",
    "dodson_tc": "closure",
    "effective_tc": "closure",
    "omega_from_diffusion": "closure",
    "build_grid": "study",
    "compare": "study",
    "observables": "study",
    "w1_length": "study",
}

__all__ = [
    "REFERENCE",
    "ClosureReference",
    "build_grid",
    "compare",
    "dodson_tc",
    "effective_tc",
    "observables",
    "omega_from_diffusion",
    "w1_length",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
