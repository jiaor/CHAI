"""Exact, batched, differentiable apatite fission-track model in PyTorch.

A direct port of ``native/ketcham/KETCH.c``
(``InterpolateTTPathKet`` -> ``CalcModelLengths*`` -> ``SumPopulationsKet`` ->
``CalcModelAgesKet``), not a learned approximation.  Validated against that C
code in ``tests/test_torch_ketcham.py``.

Both calibrations are available as :class:`AnnealingModel` objects;
:data:`KETCHAM_2007`, the model as published, is the default.

Two deliberate departures from the C, both documented in PLAN.md:

1. **Internal grid.** ``InterpolateTTPathKet`` chooses its time step adaptively
   from the cooling rate and proximity to the total-annealing temperature.
   That is data-dependent control flow, so each path in a batch would want a
   different node count.  Since the internal grid is a quadrature choice rather
   than part of the model, we use a uniformly refined grid instead
   (:func:`chai.paths.refine`) and verify convergence in ``n_sub``.

2. **No early exit.** The C returns as soon as a population anneals totally,
   leaving older entries uncomputed.  Here every node is evaluated under an
   ``alive`` mask that latches off.  This is equivalent for the age, because
   totally-annealed populations form a prefix of zeros and
   ``AgeCorrectionKet(0) == 0``, so they contribute nothing to the sum.  It is
   equivalent for the length distribution too: the population weights telescope,
   so the C's ``firstTTNode`` start index only skips terms that are gated out by
   ``rLen > 0`` anyway.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ...paths import KELVIN_AT_0C, SECS_PER_MA, refine

# Ketcham et al. (1999) fanning curvilinear model -- `modKetchamEtAl` in KETCH.c
C0, C1, C2, C3 = -19.844, 0.38951, -51.253, -7.6423
A, B = -0.12327, -11.988

# Ketcham et al. (2007) -- `modKetchamEtAl07` in KETCH.c.  `b` is unused here:
# the 2007 form inverts through 1/(1 + x^(1/a)) rather than the 1999 power law.
C0_07, C1_07, C2_07, C3_07 = 0.39528, 0.01073, -65.12969, -7.91715
A_07 = 0.04672

#: Reduced c-axis-projected length below which a population is totally annealed
#: (``MIN_OBS_RCMOD`` in KETCH.c).  This is the 1999 value; the 2007
#: calibration uses 0.13, carried on :data:`KETCHAM_2007`.
MIN_OBS_RCMOD = 0.55

#: Length reduction in the age standard (``lenRedStand``).
STD_LENGTH_REDUCTION = 0.893

#: 238U decay constant per second (``U238SEC``), used for the population weights.
U238SEC = 4.91575e-18

#: Shortest observable track (``MIN_LENGTH``); shorter bins are held at zero.
MIN_LENGTH = 2.15

#: Hard truncation of each Gaussian population, in standard deviations
#: (``PDF_NUMSD``).  One-sided: it clips bins *shorter* than the population mean.
PDF_NUMSD = 4.0

#: Bin geometry of the distribution the C oracle returns.  Set in ``ketch_main_``;
#: the ``NUM_PDF_PTS`` / ``PDF_BIN_SIZE`` in ketch.h are dead defaults.
N_PDF_BINS = 200
PDF_MAX_LENGTH = 20.0

_TINY = 1e-30


def rmr0_from_dpar(dpar: Tensor) -> Tensor:
    """Apatite-apatite conversion parameter from Dpar (``ETCH_PIT_LENGTH``).

    Note this is the 1999 exponential form used by ``CalcModelLengthsKet``;
    Ketcham et al. (2007) uses the power law in :func:`rmr0_from_dpar_ket07`.
    """
    mid = 1.0 - torch.exp(0.647 * (dpar - 1.75) - 1.834)
    return torch.where(
        dpar <= 1.75,
        torch.full_like(dpar, 0.84),
        torch.where(dpar >= 4.58, torch.zeros_like(dpar), mid),
    )


def rmr0_from_dpar_ket07(dpar: Tensor) -> Tensor:
    """Ketcham et al. (2007) power-law conversion from Dpar to ``rmr0``.

    This is what the 2007 paper (and ``legacy/Ketch07.py``) specifies.  It is
    *not* what ``CalcModelLengthsKet07`` computed before 2026-09-19, which kept
    the 1999 exponential below -- see docs/inherited.md.
    """
    mid = 0.84 * (((4.58 - dpar) / 2.98).clamp_min(0.0) ** 0.21)
    return torch.where(
        dpar <= 1.75,
        torch.full_like(dpar, 0.84),
        torch.where(dpar >= 4.58, torch.zeros_like(dpar), mid),
    )


def age_correction(c: Tensor, min_obs_rcmod: float = MIN_OBS_RCMOD) -> Tensor:
    """``AgeCorrectionKet``: track density reduction implied by length reduction."""
    return torch.where(
        c > 0.765,
        1.600 * c - 0.599,
        torch.where(
            c >= min_obs_rcmod,
            9.205 * c * c - 9.157 * c + 2.269,
            torch.zeros_like(c),
        ),
    )


# ---------------------------------------------------------------------------
# Annealing calibrations
# ---------------------------------------------------------------------------

def _reduced_length_ket99(log_time_int: Tensor, temp_calc: Tensor) -> Tensor:
    x1 = (log_time_int - C2) / (temp_calc - C3)
    # x2 <= 0 is the deep-annealing limit; clamping sends r -> 0, which is
    # the correct limit.  The C produces NaN there instead.
    x2 = (1.0 + A * (C0 + C1 * x1)).clamp_min(_TINY)
    r = x2 ** (1.0 / A)
    x3 = 1.0 - B * r
    return torch.where(x3 < 0, torch.zeros_like(x3), x3.clamp_min(_TINY) ** (1.0 / B))


def _equivalent_time_ket99(r: Tensor, temp_calc: Tensor) -> Tensor:
    et = ((1.0 - r.clamp_min(1e-12) ** B) / B) ** A
    et = ((et - 1.0) / A - C0) / C1
    return torch.exp(et * (temp_calc - C3) + C2)


def _reduced_length_ket07(log_time_int: Tensor, temp_calc: Tensor) -> Tensor:
    x1 = (log_time_int - C2_07) / (temp_calc - C3_07)
    # x2 < 0 is the no-annealing limit (r -> 1); the C takes a fractional power
    # of it and produces NaN.  Clamping to 0 gives that limit exactly.  At the
    # other end x2 ** (1/a) overflows to inf, and 1/inf = 0 is again correct.
    x2 = (C0_07 + C1_07 * x1).clamp_min(0.0)
    return 1.0 / (1.0 + x2 ** (1.0 / A_07))


def _equivalent_time_ket07(r: Tensor, temp_calc: Tensor) -> Tensor:
    et = (1.0 / r.clamp_min(1e-12) - 1.0) ** A_07
    et = (et - C0_07) / C1_07
    return torch.exp(et * (temp_calc - C3_07) + C2_07)


@dataclass(frozen=True)
class AnnealingModel:
    """A fission-track annealing calibration.

    Attributes:
        k_offset: ``k = k_offset - rmr0``; 1.0 in Ketcham et al. (1999), 1.04 in
            Ketcham et al. (2007).
        min_obs_rcmod: reduced length below which a population is treated as
            totally annealed and invisible.
        rmr0_from_dpar: Dpar -> rmr0 conversion.
        reduced_length: ``(log(time_int), -log(mean T_K)) -> r``.
        equivalent_time: ``(r, -log(mean T_K)) -> equivalent time``, the time the
            *next* interval's temperature would need to produce the same ``r``.
    """
    name: str
    k_offset: float
    min_obs_rcmod: float
    rmr0_from_dpar: Callable[[Tensor], Tensor]
    reduced_length: Callable[[Tensor, Tensor], Tensor]
    equivalent_time: Callable[[Tensor, Tensor], Tensor]


#: Ketcham et al. (1999), ``annealing_model=1`` in the native binding.
KETCHAM_1999 = AnnealingModel(
    name="ketcham1999", k_offset=1.0, min_obs_rcmod=MIN_OBS_RCMOD,
    rmr0_from_dpar=rmr0_from_dpar,
    reduced_length=_reduced_length_ket99, equivalent_time=_equivalent_time_ket99,
)

#: Ketcham et al. (2007), and the default throughout ``chai``: the 2007
#: annealing equations, ``k = 1.04 - rmr0``, and the power-law Dpar -> rmr0
#: conversion.
#:
#: ``CalcModelLengthsKet07`` paired those equations with the *1999* exponential
#: rmr0, which is in no publication; that hybrid is not reproduced here.  The
#: oracle now computes this calibration too, so the two defaults agree --- see
#: :func:`chai.forward.native.set_ket07_published`.
#:
#: ``min_obs_rcmod`` stays at 0.55, *not* the 0.13 that ``legacy/Ketch07.py``
#: uses.  0.13 is an error there: the density polynomial in
#: :func:`age_correction` has roots at 0.4426 and 0.5515, so below 0.55 it goes
#: negative and then climbs to 1.234 at ``c = 0.13``, which would make a totally
#: annealed track count for more than an unannealed one and gives d(age)/dT > 0.
#: ``RDAAM.cpp`` --- Ketcham's own code --- pairs these same 2007 constants with
#: a total-annealing length of 0.55.
KETCHAM_2007 = AnnealingModel(
    name="ketcham2007", k_offset=1.04, min_obs_rcmod=MIN_OBS_RCMOD,
    rmr0_from_dpar=rmr0_from_dpar_ket07,
    reduced_length=_reduced_length_ket07, equivalent_time=_equivalent_time_ket07,
)


def reduced_lengths(t: Tensor, T: Tensor, dpar: Tensor, n_sub: int = 64,
                    model: AnnealingModel = KETCHAM_2007
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """Reduced c-axis-projected lengths of every track population along a path.

    ``CalcModelLengthsKet`` / ``CalcModelLengthsKet07``, batched.

    Args:
        t: ``(B, N)`` control-point times in Ma, ``t[:, 0] == 0``, increasing.
        T: ``(B, N)`` control-point temperatures in degC.
        dpar: ``(B,)`` etch-pit length in um.
        n_sub: sub-intervals per control-point segment.
        model: annealing calibration; defaults to :data:`KETCHAM_2007`.

    Returns:
        ``(r, tau, first_node)`` where ``r`` is ``(B, M-1)`` reduced lengths
        after the kinetic conversion (index 0 = oldest population), ``tau`` is
        ``(B, M)`` node times in seconds ordered oldest-first, and
        ``first_node`` is ``(B,)`` the index of the youngest totally-annealed
        population.
    """
    t_fine, T_fine = refine(t, T, n_sub)
    # KETCH.c works oldest-first, in seconds and Kelvin.
    tau = t_fine.flip(-1) * SECS_PER_MA
    TK = T_fine.flip(-1) + KELVIN_AT_0C

    Bsz, M = tau.shape
    rmr0 = model.rmr0_from_dpar(dpar)
    k = model.k_offset - rmr0
    # equivTotAnnLen: total annealing for this apatite, expressed in the
    # reduced length of the more resistant one the equations are written for.
    equiv_tot = model.min_obs_rcmod ** (1.0 / k) * (1.0 - rmr0) + rmr0
    zero = torch.zeros_like(rmr0)

    equiv_time = torch.zeros_like(rmr0)
    temp_calc = -torch.log((TK[:, M - 2] + TK[:, M - 1]) / 2.0)
    alive = torch.ones(Bsz, dtype=torch.bool, device=tau.device)

    r_rev = []
    for j in range(M - 2, -1, -1):
        time_int = (tau[:, j] - tau[:, j + 1]) + equiv_time
        r = model.reduced_length(torch.log(time_int), temp_calc)
        r = torch.where(r < equiv_tot, zero, r)
        r = torch.where(alive, r, zero)
        r_rev.append(r)

        still_alive = alive & (r > 0)
        if j >= 1:
            # C updates temp_calc first, then uses the NEW value for equiv_time.
            update = still_alive & (r < 0.999)
            tc_new = -torch.log((TK[:, j - 1] + TK[:, j]) / 2.0)
            # Evaluated on a harmless r where it is not used: at a dead
            # population the exponential overflows float32, and the masked
            # inf still sends NaN back through the where().
            r_safe = torch.where(update, r, torch.full_like(r, 0.5))
            equiv_time = torch.where(update, model.equivalent_time(r_safe, tc_new),
                                     equiv_time)
            temp_calc = torch.where(update, tc_new, temp_calc)
        alive = still_alive

    r = torch.stack(r_rev[::-1], dim=1)                       # (B, M-1), oldest first

    # Kinetic conversion to the modelled apatite composition.
    rmr0_b, k_b = rmr0[:, None], k[:, None]
    conv = ((r - rmr0_b) / (1.0 - rmr0_b)).clamp_min(0.0) ** k_b
    conv = torch.where(r <= rmr0_b, torch.zeros_like(conv), conv)
    conv = torch.where(conv < model.min_obs_rcmod, torch.zeros_like(conv), conv)

    # Youngest totally-annealed population (zeros form a prefix).
    idx = torch.arange(conv.shape[1], device=conv.device)
    first_node = torch.where(conv == 0, idx, torch.full_like(idx, -1)).max(dim=1).values
    return conv, tau, first_node.clamp_min(0)


def _age_from_populations(conv: Tensor, tau: Tensor, std_length_reduction: float,
                          model: AnnealingModel) -> Tensor:
    """``CalcModelAgesKet``, given reduced lengths that are already computed."""
    dt = tau[:, :-1] - tau[:, 1:]                             # (B, M-1)
    cutoff = model.min_obs_rcmod

    mid = (conv[:, :-1] + conv[:, 1:]) / 2.0                  # (B, M-2)
    age = (age_correction(mid, cutoff) * dt[:, :-1]).sum(dim=1)
    # KETCH.c uses the endpoint, not the midpoint, for the final interval.
    age = age + age_correction(conv[:, -1], cutoff) * dt[:, -1]
    return age / (std_length_reduction * SECS_PER_MA)


def aft_age(t: Tensor, T: Tensor, dpar: Tensor, n_sub: int = 64,
            std_length_reduction: float = STD_LENGTH_REDUCTION,
            model: AnnealingModel = KETCHAM_2007) -> Tensor:
    """Pooled apatite fission-track age in Ma.

    Differentiable with respect to ``t``, ``T`` and ``dpar``.

    Args:
        t: ``(B, N)`` control-point times in Ma, ``t[:, 0] == 0``, increasing.
        T: ``(B, N)`` control-point temperatures in degC.
        dpar: ``(B,)`` etch-pit length in um.
        n_sub: sub-intervals per segment; raise until the age stops moving.
        model: annealing calibration; defaults to :data:`KETCHAM_2007`.

    Returns:
        ``(B,)`` fission-track ages in Ma.
    """
    conv, tau, _ = reduced_lengths(t, T, dpar, n_sub=n_sub, model=model)
    return _age_from_populations(conv, tau, std_length_reduction, model)


# ---------------------------------------------------------------------------
# Track-length distribution:  SumPopulationsKet -> CalcModelStats
# ---------------------------------------------------------------------------

def length_bin_centres(n_bins: int = N_PDF_BINS, max_length: float = PDF_MAX_LENGTH,
                       dtype: torch.dtype = torch.float64,
                       device: torch.device | str | None = None) -> Tensor:
    """Bin centres of the length distribution, in um -- the oracle's ``pdfAxis``."""
    i = torch.arange(n_bins, dtype=dtype, device=device)
    return (i + 0.5) * max_length / n_bins


def initial_track_length(dpar: Tensor, l0_user: float = 16.0) -> Tensor:
    """Unannealed c-axis-projected track length ``l0`` from Dpar, in um.

    ``InitialTrackLength`` with ``l0model = L0_FROM_KPAR``, ``kinParType =
    ETCH_PIT_LENGTH`` and ``doProject = 1``, i.e. ``l0vsKparDefault[1]``.  The
    ``l0_user`` fallback mirrors ``ForwardModel``'s ``if (initLength < 1)``
    guard, which no physical Dpar reaches.
    """
    l0 = 0.205 * dpar + 16.10
    return torch.where(l0 < 1.0, torch.full_like(l0, l0_user), l0)


def reduced_stdev(r: Tensor) -> Tensor:
    """``ReducedStdev`` with ``doProject = 1``: population width vs reduced length.

    The quadratic has no real roots, so this is bounded below by ~0.044 and
    never threatens the division in :func:`track_length_pdf`.
    """
    return 0.1081 - 0.1642 * r + 0.1052 * r * r


def _pdf_from_populations(conv: Tensor, tau: Tensor, dpar: Tensor, bins: Tensor,
                          model: AnnealingModel, l0_user: float = 16.0,
                          bin_chunk: int | None = None) -> Tensor:
    """``SumPopulationsKet``: sum the per-population Gaussians onto ``bins``.

    Args:
        conv: ``(B, M-1)`` kinetically converted reduced lengths, oldest first.
        tau: ``(B, M)`` node times in seconds, oldest first.
        dpar: ``(B,)`` etch-pit length in um.
        bins: ``(n_bins,)`` bin centres in um.
        bin_chunk: bins evaluated per pass.  The full ``(B, M-1, n_bins)``
            intermediate is far too large to materialise (1.2e9 elements at
            ``B=1e4``), so bins are chunked; ``None`` sizes the chunk to keep
            each pass near 8M elements.

    Returns:
        ``(B, n_bins)`` normalised density.
    """
    Bsz, n_pop = conv.shape
    n_bins = bins.shape[0]

    # Population weight: the 238U decay integral over the interval, telescoped.
    # Written with expm1 rather than the C's difference of two ~1e17 exponentials,
    # which is algebraically identical but loses no precision in float32.
    dtau = tau[:, :-1] - tau[:, 1:]                                # (B, M-1) > 0
    weight = torch.exp(U238SEC * tau[:, 1:]) * torch.expm1(U238SEC * dtau) / U238SEC

    r_len = conv                                                    # doProject = 1
    r_stdev = reduced_stdev(r_len)
    obs_bias = age_correction(conv, model.min_obs_rcmod)
    calc = weight * obs_bias / (r_stdev * (2.0 * torch.pi) ** 0.5)   # (B, M-1)
    live = conv > 0                                                  # `if (rLen > 0)`

    x = bins / initial_track_length(dpar, l0_user)[:, None]          # (B, n_bins)
    observable = bins >= MIN_LENGTH                                  # (n_bins,)

    if bin_chunk is None:
        bin_chunk = max(1, min(n_bins, 2 ** 23 // max(1, Bsz * n_pop)))

    out = []
    for lo in range(0, n_bins, bin_chunk):
        hi = min(lo + bin_chunk, n_bins)
        # (B, M-1, C): population j's Gaussian evaluated at each bin in the chunk
        z = (r_len[:, :, None] - x[:, None, lo:hi]) / r_stdev[:, :, None]
        contrib = calc[:, :, None] * torch.exp(-(z * z) / 2.0)
        keep = live[:, :, None] & (z <= PDF_NUMSD) & observable[None, None, lo:hi]
        out.append(torch.where(keep, contrib, torch.zeros_like(contrib)).sum(dim=1))
    pdf = torch.cat(out, dim=1)                                      # (B, n_bins)

    # Normalise by the trapezoidal cdf, exactly as the C does (note its cdf[0]
    # is pdf[0], not 0; that bin is below MIN_LENGTH and so always zero anyway).
    total = pdf[:, 0] + _trapezoid_areas(pdf, bins).sum(dim=1)
    ok = total > 0
    safe = torch.where(ok, total, torch.ones_like(total))
    return torch.where(ok[:, None], pdf / safe[:, None], pdf)


def _trapezoid_areas(pdf: Tensor, bins: Tensor) -> Tensor:
    """``(B, n_bins-1)`` areas of the trapezia between consecutive bins."""
    return (pdf[:, 1:] + pdf[:, :-1]) / 2.0 * (bins[1:] - bins[:-1])


def length_stats(pdf: Tensor, bins: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation of a length distribution, in um.

    The mean is the rectangle-rule sum ``ketch_main_`` reports as ``fmean``;
    the standard deviation is ``CalcModelStats``' ``StatsVal.stDev``, which uses
    trapezoidal areas instead, so the two are not quite consistent with each
    other.  Both are reproduced as-is, because both are what the oracle returns.
    """
    bin_width = (bins[-1] - bins[0]) / (bins.shape[0] - 1)
    mean = (pdf * bins).sum(dim=1) * bin_width

    # CalcModelStats: areas[0] = 0, areas[i] spans bins[i-1]..bins[i], and the
    # variance weights area i by its *right* edge bins[i].
    areas = torch.cat([torch.zeros_like(pdf[:, :1]), _trapezoid_areas(pdf, bins)], dim=1)
    total = areas.sum(dim=1)
    ok = total > 0
    safe = torch.where(ok, total, torch.ones_like(total))
    stats_mean = (areas[:, 1:] * (bins[1:] + bins[:-1]) / 2.0).sum(dim=1) / safe
    var = (areas / safe[:, None] * (bins - stats_mean[:, None]) ** 2).sum(dim=1)
    stdev = torch.sqrt(var.clamp_min(_TINY))
    return mean, torch.where(ok, stdev, torch.zeros_like(stdev))


def track_length_pdf(t: Tensor, T: Tensor, dpar: Tensor, n_sub: int = 64,
                     bins: Tensor | None = None, l0_user: float = 16.0,
                     bin_chunk: int | None = None,
                     model: AnnealingModel = KETCHAM_2007) -> tuple[Tensor, Tensor]:
    """Modelled c-axis-projected track-length distribution.

    Args:
        t: ``(B, N)`` control-point times in Ma, ``t[:, 0] == 0``, increasing.
        T: ``(B, N)`` control-point temperatures in degC.
        dpar: ``(B,)`` etch-pit length in um.
        n_sub: sub-intervals per segment.
        bins: ``(n_bins,)`` bin centres in um; defaults to the oracle's 200 bins.
        model: annealing calibration; defaults to :data:`KETCHAM_2007`.

    Returns:
        ``(pdf, bins)`` with ``pdf`` of shape ``(B, n_bins)``, normalised so its
        trapezoidal integral over ``bins`` is 1 (or identically zero where every
        population is totally annealed).

    Gradients flow through the bin weights, but the ``z <= PDF_NUMSD`` and
    ``bins >= MIN_LENGTH`` truncations are hard cutoffs inherited from the C:
    populations are excluded discontinuously, so the derivative is that of the
    surviving terms only.
    """
    if bins is None:
        bins = length_bin_centres(dtype=t.dtype, device=t.device)
    conv, tau, _ = reduced_lengths(t, T, dpar, n_sub=n_sub, model=model)
    return _pdf_from_populations(conv, tau, dpar, bins, model, l0_user, bin_chunk), bins


def aft_forward(t: Tensor, T: Tensor, dpar: Tensor, n_sub: int = 64,
                bins: Tensor | None = None,
                std_length_reduction: float = STD_LENGTH_REDUCTION,
                l0_user: float = 16.0, bin_chunk: int | None = None,
                model: AnnealingModel = KETCHAM_2007) -> dict[str, Tensor]:
    """Every AFT observable at once, sharing a single annealing pass.

    Prefer this to calling :func:`aft_age` and :func:`track_length_pdf`
    separately -- the annealing recursion is the expensive part and is shared.

    Returns:
        ``{"age", "length_pdf", "bins", "mean_length", "std_length"}``, matching
        ``AFTResult.age``, ``.length_pdf``, ``.mean_length`` and ``.dfmeanp``
        from the native binding.
    """
    if bins is None:
        bins = length_bin_centres(dtype=t.dtype, device=t.device)
    conv, tau, _ = reduced_lengths(t, T, dpar, n_sub=n_sub, model=model)
    pdf = _pdf_from_populations(conv, tau, dpar, bins, model, l0_user, bin_chunk)
    mean, stdev = length_stats(pdf, bins)
    return {
        "age": _age_from_populations(conv, tau, std_length_reduction, model),
        "length_pdf": pdf,
        "bins": bins,
        "mean_length": mean,
        "std_length": stdev,
    }
