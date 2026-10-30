"""He/Ar production-diffusion ages via Mad_He.f90 (Braun et al. 2006).

No radiation-damage treatment -- a simpler, faster alternative to RDAAM, kept
as an independent cross-check rather than as a primary training target.

Time convention: this module takes the project-canonical present->past ordering
and reverses it internally, because Mad_He.f90 steps ``time(i) - time(i+1)``
forward and therefore requires past->present input.
"""
from __future__ import annotations

import ctypes

import numpy as np

from ._loader import load

THREAD_SAFE = False

METHODS = {"ahe": 1, "zhe": 2, "kfar": 3, "biar": 4, "muar": 5, "hoar": 6}

# Opened on first use, not at import: ``Mad_He.f90`` is not in the sdist (it is not on the product's
# path, docs/release.md §8), so ``chai.forward.native`` must import with ``libmadhe.so`` absent.
_fn = None


def _function():
    global _fn
    if _fn is None:
        _fn = load("libmadhe.so").mad_he_
        _fn.restype = None
        _fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # time (Ma, past -> present)
            ctypes.POINTER(ctypes.c_float),  # temperature (degC)
            ctypes.POINTER(ctypes.c_int),    # ntime
            ctypes.POINTER(ctypes.c_float),  # out: age
            ctypes.POINTER(ctypes.c_int),    # iflag (method)
            ctypes.POINTER(ctypes.c_float),  # grain size (um)
        ]
    return _fn


def calculate_madhe(time_ma, temperature_c, grain_size: float = 100.0, method: str = "AHe") -> float:
    """Diffusion-only He/Ar age in Ma for one thermal history.

    Args:
        time_ma: times before present in Ma, ``t[0] == 0``, increasing (the
            reversal Mad_He.f90 needs is applied internally).
        temperature_c: temperatures in degC.
        grain_size: diffusion radius in um.
        method: one of AHe, ZHe, KfAr, BiAr, MuAr, HoAr.
    """
    key = method.lower()
    if key not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}, got {method!r}")

    t = np.ascontiguousarray(time_ma, dtype=np.float32)
    T = np.ascontiguousarray(temperature_c, dtype=np.float32)
    if t.shape != T.shape or t.ndim != 1:
        raise ValueError(f"time/temperature must be matching 1-D arrays, got {t.shape}, {T.shape}")
    if t.size < 2:
        raise ValueError("need at least 2 t-T nodes")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing into the past (t[0] = present = 0)")

    # Mad_He.f90 wants oldest first.
    t = np.ascontiguousarray(t[::-1])
    T = np.ascontiguousarray(T[::-1])

    ntime = ctypes.c_int(t.size)
    age = ctypes.c_float(0.0)
    iflag = ctypes.c_int(METHODS[key])
    grain = ctypes.c_float(float(grain_size))
    _function()(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(ntime), ctypes.byref(age),
        ctypes.byref(iflag), ctypes.byref(grain),
    )
    return float(age.value)
