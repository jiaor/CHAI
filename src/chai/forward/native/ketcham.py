"""Apatite fission-track forward model (Ketcham et al. annealing), via KETCH.c.

Time convention (canonical for this project): ``t[0] == 0`` is the present day
and ``t`` increases into the past, in Ma.  KETCH.c consumes the array in that
order -- ``InterpolateTTPath`` seeds its internal path from the *last* entry and
marches forward in time -- so no reversal is needed here.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from numpy.ctypeslib import ndpointer

from ._loader import load

# RDAAM-style global state is absent here, but KETCH.c writes into file-scope
# buffers, so treat it as process-local: parallelise with processes, not threads.
THREAD_SAFE = False

N_PDF_BINS = 200
#: Bin centres of the returned track-length distribution, in microns.
LENGTH_BINS = (np.arange(N_PDF_BINS) + 0.5) * 20.0 / N_PDF_BINS

KETCHAM_1999 = 1
KETCHAM_2007 = 2

# Opened on first use, not at import: the reference sources are not redistributed (docs/release.md
# §12), so ``chai.forward.native`` must import with ``libketcham.so`` absent.
_lib = None


def _library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = load("libketcham.so")
        _lib.ketch_main_.restype = None
        _lib.ketch_main_.argtypes = [
            ctypes.POINTER(ctypes.c_int),     # ntime
            ndpointer(ctypes.c_float),        # time (Ma, present -> past)
            ndpointer(ctypes.c_float),        # temperature (degC)
            ctypes.POINTER(ctypes.c_int),     # AnnealingModel
            ctypes.POINTER(ctypes.c_double),  # lenRedStand
            ctypes.POINTER(ctypes.c_int),     # InitTrackLengthMod
            ctypes.POINTER(ctypes.c_double),  # kinetic_par
            ctypes.POINTER(ctypes.c_int),     # kinparID
            ctypes.POINTER(ctypes.c_double),  # alo (initial track length)
            ctypes.POINTER(ctypes.c_double),  # out: final_age
            ctypes.POINTER(ctypes.c_double),  # out: oldest_age
            ctypes.POINTER(ctypes.c_double),  # out: fmean
            ctypes.POINTER(ctypes.c_double),  # out: dfmeanp
            ndpointer(ctypes.c_double),       # out: fdist[201]
        ]
        _lib.ketch_set_pct_.restype = None
        _lib.ketch_set_pct_.argtypes = [ctypes.POINTER(ctypes.c_double)]
        _lib.ketch_set_ket07_published_.restype = None
        _lib.ketch_set_ket07_published_.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return _lib


@dataclass(frozen=True)
class AFTResult:
    age: float                       # pooled fission-track age, Ma
    oldest_age: float                # oldest surviving track, Ma
    mean_length: float               # mean c-axis-projected length, um
    dfmeanp: float
    length_pdf: np.ndarray           # (200,) density over LENGTH_BINS


def set_ket07_published(published: bool = True) -> None:
    """Choose which Ketcham (2007) calibration ``annealing_model=2`` computes.

    ``CalcModelLengthsKet07`` shipped with the 2007 annealing equations but the
    *1999* Dpar -> rmr0 conversion and the 1999 cutoff of 0.55 -- a pairing that
    appears in no publication and that moves ages by a median 3.2% (max 68%).
    The default here is the calibration as published: the 2007 power-law rmr0.
    The total-annealing cutoff is 0.55 either way (``ketch_min_obs_rcmod``;
    KETCH.c explains why the 0.13 of ``legacy/Ketch07.py`` is an error).

    Pass ``False`` to recover the original hybrid.  It is kept only so the
    comparison in docs/inherited.md stays reproducible; nothing in ``chai``
    uses it.  Process-global and not thread-safe.
    """
    _library().ketch_set_ket07_published_(ctypes.byref(ctypes.c_int(int(bool(published)))))


def set_time_step_pct(pct: float = 1.0) -> None:
    """Set the oracle's internal quadrature step, as a percentage of path duration.

    KETCH.c hard-coded this at 1.0; we exposed it (see docs/inherited.md) so the
    oracle's own discretisation error can be measured by refining it. Smaller is
    finer and slower. Process-global and not thread-safe.
    """
    if not 0 < pct <= 100:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    _library().ketch_set_pct_(ctypes.byref(ctypes.c_double(float(pct))))


def calculate_aft(
    time_ma,
    temperature_c,
    kinetic_par: float = 1.8,
    kinpar_id: int = 0,
    annealing_model: int = KETCHAM_2007,
    len_red_stand: float = 0.893,
    init_track_length_mod: int = 1,
    alo: float = 16.0,
) -> AFTResult:
    """Fission-track age and length distribution for one thermal history.

    Args:
        time_ma: times before present in Ma, ``t[0] == 0``, increasing.
        temperature_c: temperatures in degC at those times.
        kinetic_par: kinetic parameter; meaning is set by ``kinpar_id``
            (0 -> Dpar in um, 1 -> Cl apfu, 2 -> OH apfu, 3 -> Cl wt%, 4 -> rmr0).
        annealing_model: 1 = Ketcham et al. (1999), 2 = Ketcham et al. (2007),
            which is the default and matches the PyTorch port's default.  Which
            2007 calibration ``2`` means is set by :func:`set_ket07_published`.
    """
    # KETCH.c leaves its internal `annealmodel` uninitialised for any value
    # other than 1 or 2 (undefined behaviour), so reject those here.
    if annealing_model not in (KETCHAM_1999, KETCHAM_2007):
        raise ValueError(f"annealing_model must be 1 or 2, got {annealing_model}")
    if kinpar_id not in range(5):
        raise ValueError(f"kinpar_id must be in 0..4, got {kinpar_id}")

    t = np.ascontiguousarray(time_ma, dtype=np.float32)
    T = np.ascontiguousarray(temperature_c, dtype=np.float32)
    if t.shape != T.shape or t.ndim != 1:
        raise ValueError(f"time/temperature must be matching 1-D arrays, got {t.shape}, {T.shape}")
    if t.size < 2:
        raise ValueError("need at least 2 t-T nodes")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing into the past (t[0] = present = 0)")

    ntime = ctypes.c_int(t.size)
    am = ctypes.c_int(annealing_model)
    lrs = ctypes.c_double(len_red_stand)
    itlm = ctypes.c_int(init_track_length_mod)
    kp = ctypes.c_double(float(kinetic_par))
    kpid = ctypes.c_int(kinpar_id)
    alo_c = ctypes.c_double(alo)
    age = ctypes.c_double(0.0)
    oldest = ctypes.c_double(0.0)
    fmean = ctypes.c_double(0.0)
    dfmeanp = ctypes.c_double(0.0)
    fdist = np.zeros(N_PDF_BINS + 1, dtype=np.float64)

    _library().ketch_main_(
        ctypes.byref(ntime), t, T, ctypes.byref(am), ctypes.byref(lrs),
        ctypes.byref(itlm), ctypes.byref(kp), ctypes.byref(kpid),
        ctypes.byref(alo_c), ctypes.byref(age), ctypes.byref(oldest),
        ctypes.byref(fmean), ctypes.byref(dfmeanp), fdist,
    )
    return AFTResult(
        age=age.value,
        oldest_age=oldest.value,
        mean_length=fmean.value,
        dfmeanp=dfmeanp.value,
        length_pdf=fdist[1:].copy(),
    )
