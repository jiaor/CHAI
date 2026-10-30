"""(U-Th)/He forward model with radiation-damage accumulation (RDAAM/ZRDAAM).

C++ from David Whipp's Tc1D (https://github.com/HUGG/Tc1D), implementing
Flowers et al. (2009) for apatite and Guenthner et al. (2013) for zircon.

Time convention: ``t[0] == 0`` (present), increasing into the past -- the same
order RDAAM.cpp's path interpolator expects.

NOT thread-safe: ``RDAAM_Init`` / ``RDAAM_FreeCalcArrays`` mutate file-scope
state in the C++ translation unit, so concurrent calls in one process corrupt
each other.  Parallelise with processes only.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from numpy.ctypeslib import ndpointer

from ._loader import load

THREAD_SAFE = False

_APATITE, _ZIRCON = 0, 1

#: Internal precision settings (``HE_PREC_*`` in RDAAM.h).  These set the radial
#: node count (2**7 / 2**8 / 2**9) and the interpolator's maximum temperature
#: step (10 / 2 / 0.5 K).  Comparing BEST against BETTER is how this code's own
#: discretisation error is measured -- see docs/porting-notes.md rule 3.
PREC_GOOD, PREC_BETTER, PREC_BEST = 0, 1, 2
_MINERALS = {"apatite": _APATITE, "zircon": _ZIRCON}

# Opened on first use, not at import: the reference sources are not redistributed (docs/release.md
# §12), so ``chai.forward.native`` must import with ``librdaam.so`` absent.
_lib = None


def _library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = load("librdaam.so")
        _lib.calculate_age_bridge.restype = None
        _lib.calculate_age_bridge.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # time (Ma, present -> past)
            ctypes.POINTER(ctypes.c_double),  # temperature (degC)
            ctypes.c_int,                     # n points
            ctypes.c_double,                  # radius (um)
            ctypes.c_double,                  # U (ppm)
            ctypes.c_double,                  # Th (ppm)
            ctypes.c_double,                  # Sm (ppm)
            ctypes.c_int,                     # 0 = apatite, 1 = zircon
            ctypes.c_int,                     # precision: 0 good, 1 better, 2 best
            ctypes.POINTER(ctypes.c_double),  # out: raw age
            ctypes.POINTER(ctypes.c_double),  # out: alpha-ejection corrected age
        ]
        _lib.calculate_age_debug.restype = None
        _lib.calculate_age_debug.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # time (Ma, present -> past)
            ctypes.POINTER(ctypes.c_double),  # temperature (degC)
            ctypes.c_int,                     # n points
            ctypes.c_double,                  # radius (um)
            ctypes.c_double,                  # U (ppm)
            ctypes.c_double,                  # Th (ppm)
            ctypes.c_double,                  # Sm (ppm)
            ctypes.c_int,                     # precision
            ctypes.POINTER(ctypes.c_double),  # out: raw age
            ctypes.POINTER(ctypes.c_double),  # out: corrected age
            ctypes.POINTER(ctypes.c_double),  # out: total He (nmol/g)
            ndpointer(ctypes.c_double),       # out: ft[4]
            ctypes.POINTER(ctypes.c_int),     # out: interpolated node count
        ]
    return _lib


@dataclass(frozen=True)
class HeResult:
    age: float            # raw (uncorrected) He age, Ma
    corrected_age: float  # alpha-ejection (Ft) corrected age, Ma


def calculate_rdaam(
    time_ma,
    temperature_c,
    radius: float,
    U: float,
    Th: float,
    Sm: float = 0.0,
    mineral: str = "apatite",
    precision: int = PREC_BEST,
) -> HeResult:
    """(U-Th)/He age for one thermal history and one grain.

    Args:
        time_ma: times before present in Ma, ``t[0] == 0``, increasing.
        temperature_c: temperatures in degC.
        radius: grain (equivalent spherical) radius in um.
        U, Th, Sm: parent concentrations in ppm.
        precision: :data:`PREC_GOOD`, :data:`PREC_BETTER` or :data:`PREC_BEST`.
            Lower settings are much faster and coarser; the gap between BEST and
            BETTER bounds this code's own discretisation error.
    """
    if mineral not in _MINERALS:
        raise ValueError(f"mineral must be one of {sorted(_MINERALS)}, got {mineral!r}")

    t = np.ascontiguousarray(time_ma, dtype=np.float64)
    T = np.ascontiguousarray(temperature_c, dtype=np.float64)
    if t.shape != T.shape or t.ndim != 1:
        raise ValueError(f"time/temperature must be matching 1-D arrays, got {t.shape}, {T.shape}")
    if t.size < 2:
        raise ValueError("need at least 2 t-T nodes")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing into the past (t[0] = present = 0)")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if U < 0 or Th < 0 or Sm < 0:
        raise ValueError("U, Th, Sm must be non-negative")
    if precision not in (PREC_GOOD, PREC_BETTER, PREC_BEST):
        raise ValueError(f"precision must be 0, 1 or 2, got {precision}")

    age = ctypes.c_double(0.0)
    corr = ctypes.c_double(0.0)
    _library().calculate_age_bridge(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t.size),
        ctypes.c_double(float(radius)),
        ctypes.c_double(float(U)),
        ctypes.c_double(float(Th)),
        ctypes.c_double(float(Sm)),
        ctypes.c_int(_MINERALS[mineral]),
        ctypes.c_int(int(precision)),
        ctypes.byref(age),
        ctypes.byref(corr),
    )
    return HeResult(age=age.value, corrected_age=corr.value)


@dataclass(frozen=True)
class HeDebugResult:
    """Everything ``calculate_rdaam`` discards, for validating a port."""
    age: float
    corrected_age: float
    total_he: float          # nmol/g
    ft: np.ndarray           # (4,) alpha-ejection factors: 238U, 235U, 232Th, 147Sm
    n_nodes: int             # nodes the internal adaptive interpolator produced


def calculate_rdaam_debug(
    time_ma,
    temperature_c,
    radius: float,
    U: float,
    Th: float,
    Sm: float = 0.0,
    precision: int = PREC_BEST,
) -> HeDebugResult:
    """Apatite only, and apart from the extra outputs identical to
    :func:`calculate_rdaam`.

    ``n_nodes`` is the useful one: the interpolator's grid is chosen from the
    path, so it is the only way to compare this code's discretisation against a
    port's on equal terms.
    """
    t = np.ascontiguousarray(time_ma, dtype=np.float64)
    T = np.ascontiguousarray(temperature_c, dtype=np.float64)
    if t.shape != T.shape or t.ndim != 1 or t.size < 2:
        raise ValueError(f"time/temperature must be matching 1-D arrays, got {t.shape}, {T.shape}")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing into the past (t[0] = present = 0)")

    age, corr, he = (ctypes.c_double(0.0) for _ in range(3))
    ft = np.zeros(4, dtype=np.float64)
    n = ctypes.c_int(0)
    _library().calculate_age_debug(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t.size), ctypes.c_double(float(radius)), ctypes.c_double(float(U)),
        ctypes.c_double(float(Th)), ctypes.c_double(float(Sm)), ctypes.c_int(int(precision)),
        ctypes.byref(age), ctypes.byref(corr), ctypes.byref(he), ft, ctypes.byref(n),
    )
    return HeDebugResult(age=age.value, corrected_age=corr.value, total_he=he.value,
                         ft=ft, n_nodes=n.value)


def eU(U: float, Th: float, Sm: float = 0.0):
    """Effective uranium concentration (ppm), the usual damage proxy."""
    return U + 0.235 * Th + 0.0047 * Sm
