"""The observation model: from modelled observables to measured data.

Stages 2 and 3 give the *modelled* observables -- a pooled fission-track age, a
track-length distribution, an alpha-ejection-corrected helium age per grain.
None of those is data.  Data is what a laboratory reports, which differs from
the model in two ways that matter to every inference built on top:

1. **It is noisy, and the noise is not a fixed percentage.**  A fission-track
   age is a count of etched tracks, so its precision is set by how many tracks
   there were to count -- which is the age times the uranium content times the
   area the analyst looked at.  A 5 Ma, 3 ppm apatite and a 150 Ma, 80 ppm
   apatite are not the same measurement with the same error bar, and the Stage 1
   audit's stand-in tolerance (a flat 5%) treats them as if they were.
2. **It is a set, not a vector.**  One rock gives one AFT age and *some* helium
   grains, each with its own radius and eU.  The Stage 1 audit sent Stage 5
   after the spread of ages *across* grains, on the grounds that a single
   helium age adds almost nothing once the length distribution is in hand.
   ``docs/observation.md`` Sec 6 finds that reading overstated -- it averages
   over a deep past nothing constrains, and in the window where helium is
   sensitive one age is worth 20% -- but the recommendation right, and neither
   the test nor the answer exists unless the container can hold several.

This module supplies the first; :mod:`chai.data.aliquot` supplies the second.

Two directions, on purpose
--------------------------
:meth:`ObservationModel.observe` **draws** data given the truth, in NumPy,
seeded, batched over samples.  That is the direction simulation-based inference
needs, and it is the direction the corpus is read in.

:meth:`ObservationModel.log_prob` **scores** data given a prediction, in
PyTorch, differentiably.  That is the direction HMC/NUTS needs -- PLAN.md R2
exists for it -- and it composes with the Stage 2/3 ports to give
``d(log likelihood)/dT_i`` in one backward pass.

They describe the same distribution, with one documented exception: the
sampler draws the track count from a Poisson, and ``log_prob`` uses the
log-normal that the Poisson converges to.  ``docs/observation.md`` measures the
gap (it matters below ~20 counted tracks, and the corpus fraction there is
reported); the alternative -- marginalising the unobservable calibration
factors against a Poisson -- has no closed form and no practitioner uses it.

What is *not* modelled, and why
-------------------------------
* **Grain-property errors are conditioned on, not marginalised.**  Radius, eU
  and Dpar are perturbed by :meth:`observe`, because a real analysis reports
  measured values, and a network trained on true ones would be trained on
  information it will never be given.  But ``log_prob`` conditions on the
  reported values rather than integrating over the truth behind them.  That is
  what every code in this field does, it is right to O(sigma^2), and the
  alternative is a nested integral per grain.
* **Compositional zoning.**  U-Th zonation changes both the alpha-ejection
  correction and the damage distribution, and is the single largest contributor
  to single-grain scatter.  It is folded into the dispersion term here rather
  than forward-modelled, because forward-modelling it needs a prior over
  radial zoning profiles that this project does not have and RDAAM as bound
  cannot consume.  :attr:`AHeNoise.dispersion_rel_sd` is where it lives.
* **Confined-length yield.**  How many lengths an analyst can measure falls off
  sharply in low-density samples, because a confined track must be intersected
  by another track to be etched at all.  The mechanism is understood and the
  constant is not: see :attr:`AFTNoise.length_yield_ref`, which is off by
  default and which ``docs/observation.md`` quantifies instead of asserting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from chai.data.aliquot import Aliquot

__all__ = [
    "EFFORT_KEYS",
    "AFTNoise",
    "AHeNoise",
    "ObservationModel",
    "age_from_density",
    "length_w1_sigma",
    "mean_length_sigma",
    "observe_corpus",
    "smear_matrix",
    "spontaneous_density",
    "track_density_per_ma_ppm",
    "truth_from_corpus",
]

# ---------------------------------------------------------------------------
# Decay and counting constants
# ---------------------------------------------------------------------------

#: Total decay constant of 238U, /yr (Jaffey et al. 1971).
LAMBDA_D = 1.55125e-10

#: Spontaneous-fission decay constant of 238U, /yr.  The literature spans
#: 7.0-8.5e-17 and the zeta calibration exists precisely because of it.  Here
#: it only sets the *scale* of the track count, so a 10% error in it is
#: indistinguishable from a 10% error in the area an analyst counted -- which
#: is why this module never claims to predict a track density to better than a
#: factor, and why :attr:`AFTNoise.area_per_grain_um2` is the knob to turn if
#: the realised counts look unlike a particular laboratory's.
LAMBDA_F = 8.46e-17

AVOGADRO = 6.02214076e23
U_MOLAR_MASS = 238.029        # g/mol, natural U
U238_ABUNDANCE = 0.992742     # atom fraction
APATITE_DENSITY = 3.20        # g/cm3

#: Etchable range of a fission track in apatite, um.  Equal to the oracle's
#: default unannealed length ``alo`` (``calculate_aft(..., alo=16.0)``), so the
#: counting model and the annealing model describe the same track.
ETCHABLE_RANGE_UM = 16.0

#: Fraction of tracks intersecting the polished surface that etch and are
#: counted.  0.93 is the usual apatite figure; like ``LAMBDA_F`` it is absorbed
#: into the effective counted area.
ETCH_EFFICIENCY = 0.93


def track_density_per_ma_ppm(etchable_range_um: float = ETCHABLE_RANGE_UM,
                             efficiency: float = ETCH_EFFICIENCY) -> float:
    """Spontaneous tracks per cm^2 per (Ma of age x ppm of U), young-age limit.

    The derivation, since the number is otherwise a magic constant:

    * 1 ppm U in apatite is
      ``rho_ap * 1e-6 / M_U * N_A * f238`` = 8.04e15 atoms of 238U per cm^3.
    * Each decays by spontaneous fission at ``LAMBDA_F``, so a Ma of
      accumulation gives ``8.04e15 * 8.46e-17 * 1e6`` = 6.80e5 fissions/cm^3.
    * A polished internal surface cuts every track whose centre lies within
      ``R |cos theta|`` of it; averaged over orientation that is ``R/2`` of
      depth, so the areal density is ``6.80e5 * R/2 * q``.

    With ``R`` = 16 um and ``q`` = 0.93 that is 506 tracks/cm^2 per Ma per ppm:
    a 50 Ma, 20 ppm apatite carries 5.1e5 tracks/cm^2, which is squarely inside
    the 1e5-1e6 band analysts work in.
    """
    n238 = APATITE_DENSITY * 1e-6 / U_MOLAR_MASS * AVOGADRO * U238_ABUNDANCE
    fissions = n238 * LAMBDA_F * 1e6                      # per cm3 per Ma
    return fissions * (etchable_range_um * 1e-4) / 2.0 * efficiency


#: Evaluated once; ``spontaneous_density`` is called per corpus block.
_K_RHO = track_density_per_ma_ppm()


def spontaneous_density(age_ma: np.ndarray, U_ppm: np.ndarray) -> np.ndarray:
    """Spontaneous track density, tracks/cm^2, for an apparent age and a U content.

    ``rho = K U (exp(lambda_d t) - 1) / (lambda_d * 1e6)``, the standard
    accumulation law with the full exponential rather than its linearisation
    (worth 0.8% at 100 Ma, 4% at 500 Ma -- small, but it is one ``expm1``).

    The *apparent* age is the right argument, not the stratigraphic one: a
    fission-track age is defined as the age whose undisturbed track density
    equals the observed one, so annealing has already been absorbed into it by
    the time the oracle returns it.  That is what makes this function the
    inverse of :func:`age_from_density` and not an approximation to one.
    """
    age_ma = np.asarray(age_ma, dtype=float)
    return _K_RHO * np.asarray(U_ppm, dtype=float) * np.expm1(LAMBDA_D * age_ma * 1e6) \
        / (LAMBDA_D * 1e6)


def age_from_density(rho: np.ndarray, U_ppm: np.ndarray) -> np.ndarray:
    """Apparent age in Ma from a track density and a measured U content."""
    rho = np.asarray(rho, dtype=float)
    U_ppm = np.maximum(np.asarray(U_ppm, dtype=float), 1e-12)
    return np.log1p(LAMBDA_D * 1e6 * rho / (_K_RHO * U_ppm)) / (LAMBDA_D * 1e6)


# ---------------------------------------------------------------------------
# Track lengths
# ---------------------------------------------------------------------------

def smear_matrix(bins: np.ndarray, sigma_um: float) -> np.ndarray:
    """``(B, B)`` transfer matrix for length measurement error.

    Row ``i`` is the distribution of the *measured* length of a track whose
    true length falls in bin ``i``: a Gaussian of width ``sigma_um``, integrated
    over each bin and renormalised.  Renormalising matters -- a 16 um track
    smeared by 0.15 um loses nothing, but the truncation at 0 and 20 um would
    otherwise quietly delete probability and leave the multinomial's ``pvals``
    summing to less than one.  Conditioning on "the track was measured, and its
    measurement landed in range" is both the honest reading and the one that
    keeps the likelihood normalised.

    A matrix rather than a convolution because the same object is needed in
    NumPy (to draw) and in torch (to score), and a 200x200 matmul is free.
    """
    if sigma_um <= 0:
        return np.eye(bins.size)
    from scipy.special import erf

    width = float(bins[1] - bins[0])
    edges = np.concatenate([bins - width / 2, bins[-1:] + width / 2])
    z = (edges[None, :] - bins[:, None]) / (sigma_um * np.sqrt(2.0))
    cdf = 0.5 * (1.0 + erf(z))
    M = np.diff(cdf, axis=1)
    return M / M.sum(axis=1, keepdims=True)


def length_w1_sigma(pdf: np.ndarray, n_lengths: np.ndarray | int,
                    bin_width: float) -> np.ndarray:
    """Expected transport distance between ``n`` measured lengths and their truth, um.

    The Stage 1 audit compared two length distributions with a stand-in
    tolerance of 0.15 um and said so.  This is that number, derived: the
    Wasserstein-1 distance between two distributions is ``int |F - G| dx``, and
    for an empirical CDF from ``n`` draws ``F_n(x) - F(x)`` is asymptotically
    normal with sd ``sqrt(F(1-F)/n)``, so

        ``E[W1] = sqrt(2 / (pi n)) * int sqrt(F (1 - F)) dx``.

    It is a *scale*, not a standard deviation -- the mean absolute deviation of
    the sampling noise -- which is exactly what the audit's kernel needs, and
    it depends on the distribution: a narrow, fully-reset length distribution
    is pinned down by 100 tracks far better than a broad, mixed-age one.
    """
    pdf = np.atleast_2d(np.asarray(pdf, dtype=float))
    p = pdf / np.maximum(pdf.sum(axis=1, keepdims=True), 1e-300)
    F = np.cumsum(p, axis=1)
    integral = np.sqrt(np.clip(F * (1.0 - F), 0.0, None)).sum(axis=1) * bin_width
    n = np.maximum(np.asarray(n_lengths, dtype=float), 1.0)
    return np.sqrt(2.0 / (np.pi * n)) * integral


def mean_length_sigma(pdf: np.ndarray, n_lengths: np.ndarray | int,
                      bins: np.ndarray, meas_sd_um: float = 0.0) -> np.ndarray:
    """Standard error of the measured mean track length, um.

    The other stand-in the Stage 1 audit flagged: it used a flat 0.10 um.  The
    real number is ``sd(length) / sqrt(n)``, and the spread of a length
    distribution is itself a strong function of the history -- a fully reset
    sample is narrow and a partially annealed one is broad, by more than a
    factor of two -- so the precision of the mean is not a constant either.

    ``meas_sd_um`` adds the per-track measurement error in quadrature, which
    is the only place it enters: smearing widens the distribution, it does not
    bias its mean.
    """
    pdf = np.atleast_2d(np.asarray(pdf, dtype=float))
    p = pdf / np.maximum(pdf.sum(axis=1, keepdims=True), 1e-300)
    mean = p @ bins
    var = p @ (bins ** 2) - mean ** 2
    n = np.maximum(np.asarray(n_lengths, dtype=float), 1.0)
    return np.sqrt(np.clip(var, 0.0, None) + meas_sd_um ** 2) / np.sqrt(n)


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AFTNoise:
    """Fission-track measurement: counting statistics, calibration, lengths.

    The defaults describe a competent, unremarkable single-sample analysis:
    20 grains counted, a few thousand square microns each, 100 confined lengths.
    They are lab effort, not physics, and every one of them is meant to be
    changed to match a dataset.

    Args:
        n_grains_counted: grains counted for the pooled age.
        area_per_grain_um2: area counted per grain.  With the other defaults a
            50 Ma, 20 ppm apatite yields 203 spontaneous tracks and an age
            good to 7.9%; a 5 Ma, 3 ppm one yields 3 tracks and an age good to
            58%.  That spread *is* the point of this class.
        zeta_rel_sd: relative sd of the calibration (zeta, or the reference
            material in a LA-ICP-MS workflow).  Systematic within a sample.
        u_rel_sd: relative sd of the uranium determination.  Also systematic:
            it scales the whole age, not each track.
        n_lengths: confined lengths the analyst sets out to measure.
        length_meas_sd_um: repeatability of one c-axis-projected length.  The
            oracle's distribution is already projected and already carries
            Ketcham's observational bias, so this is the measurement on top of
            it, not the projection itself.
        dpar_pit_sd_um: scatter between individual etch pits.
        n_dpar: pits averaged, so the reported Dpar has sd
            ``dpar_pit_sd_um / sqrt(n_dpar)``.
        length_yield_ref: track density, tracks/cm^2, at which ``n_lengths``
            confined tracks are routinely found.  Confined tracks need a host
            track to etch through, so their density goes as ``rho^2`` and the
            realised count falls off quadratically below this.  ``None``
            (the default) disables it: the mechanism is textbook but the
            constant is not calibrated anywhere this project can cite, and
            asserting one would put an invented number inside every posterior.
            ``docs/observation.md`` reports how much of the corpus it would
            bite on instead.
    """

    n_grains_counted: int = 20
    area_per_grain_um2: float = 2000.0
    zeta_rel_sd: float = 0.02
    u_rel_sd: float = 0.03
    n_lengths: int = 100
    length_meas_sd_um: float = 0.15
    dpar_pit_sd_um: float = 0.20
    n_dpar: int = 20
    length_yield_ref: float | None = None

    def __post_init__(self) -> None:
        if self.n_grains_counted < 1 or self.n_dpar < 1:
            raise ValueError("counted grains and etch pits must be >= 1")
        if self.area_per_grain_um2 <= 0:
            raise ValueError(f"area must be positive, got {self.area_per_grain_um2}")
        if self.n_lengths < 0:
            raise ValueError(f"n_lengths must be >= 0, got {self.n_lengths}")
        for name in ("zeta_rel_sd", "u_rel_sd", "length_meas_sd_um", "dpar_pit_sd_um"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def area_cm2(self) -> float:
        """Total counted area."""
        return self.n_grains_counted * self.area_per_grain_um2 * 1e-8

    @property
    def systematic_rel_var(self) -> float:
        """Relative variance shared by the whole sample -- calibration and U."""
        return self.zeta_rel_sd ** 2 + self.u_rel_sd ** 2

    @property
    def dpar_sd(self) -> float:
        return self.dpar_pit_sd_um / np.sqrt(self.n_dpar)

    def expected_counts(self, age_ma: np.ndarray, U_ppm: np.ndarray) -> np.ndarray:
        """Spontaneous tracks this effort expects to count."""
        return spontaneous_density(age_ma, U_ppm) * self.area_cm2

    def age_rel_sd(self, age_ma: np.ndarray, U_ppm: np.ndarray) -> np.ndarray:
        """Relative one-sigma on the age, before drawing anything.

        The forecast, as opposed to :meth:`ObservationModel.observe`'s realised
        value: ``sqrt(1/Ns + zeta^2 + U^2)`` at the *expected* count.  This is
        the function that replaces the audit's flat 5%.
        """
        ns = np.maximum(self.expected_counts(age_ma, U_ppm), 1e-9)
        return np.sqrt(1.0 / ns + self.systematic_rel_var)


@dataclass(frozen=True)
class AHeNoise:
    """Single-grain (U-Th)/He: analytical error, excess dispersion, outliers.

    Helium dating's defining difficulty is that replicate grains from one rock
    disagree by far more than their analytical uncertainties (Flowers et al.
    2022 and the literature it reviews; Durango aliquots reproduce to ~5-10%
    against analytical errors of 1-3%).  The excess is real and physical --
    U-Th zonation, undetected inclusions, implantation from neighbouring
    grains, broken crystals whose Ft correction is wrong -- and a model that
    reports only the analytical error will drive any inference to
    over-confident, over-structured thermal histories.

    Args:
        analytical_rel_sd: gas and parent measurement, propagated through the
            age equation.
        dispersion_rel_sd: the excess, as a relative sd in log-age.  Everything
            in the paragraph above lives here.
        outlier_fraction: probability that a grain is not a member of that
            population at all.  **Zero by default.**  Real datasets do contain
            such grains, but any non-zero value here is a claim about a
            laboratory's practice that nothing in this project measures -- the
            same objection that retired the prototype's hand-tuned mixture
            weights (``docs/prior.md``).  Set it, with a dataset behind it, to
            train a posterior that is robust to bad grains.
        outlier_rel_sd: width of that component, in log-age.  0.7 spans roughly
            a factor of two either way, which is what an inclusion or a
            mis-estimated Ft does.  Symmetric, because both signs occur.
        radius_rel_sd: equivalent-spherical-radius estimate from a 2-D grain
            image.
        eU_rel_sd: relative sd on the reported eU.
        n_grains_range: how many grains of those available were dated,
            inclusive.  ``None`` dates all of them.  A range makes the set size
            vary, which is what a Deep Sets consumer must handle and what real
            aliquots do.
    """

    analytical_rel_sd: float = 0.025
    dispersion_rel_sd: float = 0.06
    outlier_fraction: float = 0.0
    outlier_rel_sd: float = 0.7
    radius_rel_sd: float = 0.05
    eU_rel_sd: float = 0.03
    n_grains_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for name in ("analytical_rel_sd", "dispersion_rel_sd", "outlier_rel_sd",
                     "radius_rel_sd", "eU_rel_sd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.outlier_fraction <= 1.0:
            raise ValueError(f"outlier_fraction must be in [0, 1], got {self.outlier_fraction}")
        if self.n_grains_range is not None:
            lo, hi = self.n_grains_range
            if not 1 <= lo <= hi:
                raise ValueError(f"n_grains_range must be ordered and >= 1, got {(lo, hi)}")

    @property
    def rel_sd(self) -> float:
        """Reported one-sigma: analytical and dispersion, in quadrature."""
        return float(np.hypot(self.analytical_rel_sd, self.dispersion_rel_sd))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

#: What :meth:`ObservationModel.observe` needs to be given.  ``U`` is the
#: aliquot's uranium content in ppm -- what sets the track count -- and is a
#: separate input from the per-grain ``he_eU`` because the fission-track age
#: and the helium ages are measured on different grains of the same rock.
TRUTH_KEYS = ("aft_age", "length_pdf", "dpar", "U", "he_age", "he_radius", "he_eU")

#: What :meth:`ObservationModel.observe` accepts as per-sample ``effort``.
EFFORT_KEYS = ("n_grains_counted", "area_per_grain_um2", "n_lengths", "n_he_grains",
               "aft_present")


@dataclass(frozen=True)
class ObservationModel:
    """Truth -> data, and data -> log-likelihood.

    Args:
        aft: fission-track noise.
        he: helium noise.
        length_bins: bin centres of the track-length distribution, um.
            Defaults to the oracle's 200 bins, which is what both the corpus
            and the torch port produce.
    """

    aft: AFTNoise = AFTNoise()
    he: AHeNoise = AHeNoise()
    length_bins: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.length_bins is None:
            # the oracle's axis, from the port so that the compiled oracle is not needed
            from chai.forward.torchimpl.ketcham import length_bin_centres
            object.__setattr__(self, "length_bins", length_bin_centres().numpy())
        object.__setattr__(self, "_smear",
                           smear_matrix(self.length_bins, self.aft.length_meas_sd_um))

    @property
    def bin_width(self) -> float:
        return float(self.length_bins[1] - self.length_bins[0])

    def smeared(self, pdf: np.ndarray) -> np.ndarray:
        """Bin probabilities of the *measured* length distribution.

        Accepts either convention -- the oracle's normalised density or the
        corpus's bin probabilities -- because it renormalises first.  Rows that
        are identically zero (a totally annealed history has no tracks and no
        distribution) stay zero and are handled by the caller.
        """
        pdf = np.atleast_2d(np.asarray(pdf, dtype=float))
        total = pdf.sum(axis=1, keepdims=True)
        p = np.divide(pdf, total, out=np.zeros_like(pdf), where=total > 0)
        return p @ self._smear

    # -- draw ---------------------------------------------------------------

    def observe(self, truth: dict[str, np.ndarray], rng: np.random.Generator,
                effort: dict[str, np.ndarray] | None = None) -> Aliquot:
        """Draw one measurement of every sample in ``truth``.

        Noise is drawn fresh on every call rather than stored with the corpus.
        That is deliberate: a simulation-based-inference training set gets more
        from re-drawing the observation each epoch (the network then sees the
        true ``p(x | theta)`` rather than one sample of it), and a corpus that
        stored noisy data would have baked one laboratory's effort into a file
        that takes an hour to regenerate.

        Args:
            truth: arrays keyed by :data:`TRUTH_KEYS`.  ``aft_age`` ``(n,)`` Ma,
                ``length_pdf`` ``(n, n_bins)``, ``dpar`` ``(n,)`` um, ``U``
                ``(n,)`` ppm, and ``he_age`` / ``he_radius`` / ``he_eU``
                ``(n, G)``.
            rng: seeded generator; nothing else is consulted.
            effort: **per-sample** laboratory effort, overriding the scalar
                fields of :class:`AFTNoise` and :class:`AHeNoise` row by row
                (PLAN.md Stage 11).  Any subset of ``n_grains_counted``,
                ``area_per_grain_um2``, ``n_lengths`` -- each ``(n,)`` -- and
                ``n_he_grains`` ``(n,)``, how many of the available helium
                grains were dated (``0`` dates none), and ``aft_present``
                ``(n,)`` bool, False for a sample with no fission-track
                analysis at all.  Keys not given fall back to the
                dataclass fields.  This is how one training batch carries
                twenty laboratories' worth of practice instead of one, and how
                a sample missing an observable gets into the training set:
                *missing* is an effort of zero.

        Returns:
            An :class:`~chai.data.aliquot.Aliquot` of the same length, with
            its helium grains in random order.
        """
        missing = [k for k in TRUTH_KEYS if k not in truth]
        if missing:
            raise ValueError(f"truth is missing {missing}")
        age = np.asarray(truth["aft_age"], dtype=float)
        U = np.asarray(truth["U"], dtype=float)
        n = age.size
        effort = dict(effort or {})
        unknown = set(effort) - set(EFFORT_KEYS)
        if unknown:
            raise ValueError(f"effort has unknown keys {sorted(unknown)}; allowed {EFFORT_KEYS}")
        def per_sample(key: str, default: Any, dtype: Any = float) -> np.ndarray:
            v = np.asarray(effort.get(key, default), dtype=dtype)
            if v.ndim == 0:
                v = np.full(n, v, dtype=dtype)
            if v.shape != (n,):
                raise ValueError(f"effort[{key!r}] must be ({n},), got {v.shape}")
            return v
        n_counted = per_sample("n_grains_counted", self.aft.n_grains_counted)
        area_um2 = per_sample("area_per_grain_um2", self.aft.area_per_grain_um2)
        n_lengths = per_sample("n_lengths", self.aft.n_lengths, dtype=np.int64)
        aft_present = per_sample("aft_present", True, dtype=bool)
        n_he = effort.get("n_he_grains")
        if n_he is not None:
            n_he = per_sample("n_he_grains", n_he, dtype=np.int64)
        if (n_counted < 1).any() or (area_um2 <= 0).any() or (n_lengths < 0).any():
            raise ValueError("effort must have >= 1 counted grain, positive area, >= 0 lengths")
        area_cm2 = n_counted * area_um2 * 1e-8
        # A one-grain aliquot must still arrive as (n, 1): ``atleast_2d`` on an
        # (n,) array would read n grains of one sample, silently and wrongly.
        for k in ("he_age", "he_radius", "he_eU"):
            if np.ndim(truth[k]) != 2:
                raise ValueError(f"truth[{k!r}] must be (n, n_grains), got "
                                 f"shape {np.shape(truth[k])}")
        for k in TRUTH_KEYS:
            if np.shape(truth[k])[0] != n:
                raise ValueError(f"truth[{k!r}] has {np.shape(truth[k])[0]} rows, "
                                 f"expected {n}")

        # -- fission-track age: a Poisson count, then two systematic factors.
        lam = spontaneous_density(age, U) * area_cm2
        ns = rng.poisson(lam)
        zeta = np.exp(rng.normal(0.0, self.aft.zeta_rel_sd, n)) if self.aft.zeta_rel_sd else 1.0
        u_fac = np.exp(rng.normal(0.0, self.aft.u_rel_sd, n)) if self.aft.u_rel_sd else 1.0
        U_obs = U * u_fac
        age_obs = age_from_density(ns / area_cm2 * zeta, U_obs)

        # The age carried by a single track, which is what converts a count
        # uncertainty into an age uncertainty.  Using it rather than
        # ``age_obs / sqrt(ns)`` costs nothing and stays finite at ns = 0,
        # where the sample has no age but still has an upper bound.
        unit = age_from_density(zeta / area_cm2, U_obs)
        age_sd_count = np.sqrt(np.maximum(ns, 1.0)) * unit
        age_sd = np.sqrt(age_sd_count ** 2 + (age_obs ** 2) * self.aft.systematic_rel_var)

        # -- confined lengths: multinomial over the smeared distribution.
        p = self.smeared(truth["length_pdf"])
        have = p.sum(axis=1) > 0
        n_l = np.where(have, n_lengths, 0)
        if self.aft.length_yield_ref is not None:
            yield_frac = np.minimum(1.0, (spontaneous_density(age, U)
                                          / self.aft.length_yield_ref) ** 2)
            n_l = rng.poisson(n_l * yield_frac)
        counts = self._multinomial(n_l, p, rng)

        # -- Dpar: the mean of ``n_dpar`` etch pits.
        dpar = np.asarray(truth["dpar"], dtype=float) + rng.normal(0.0, self.aft.dpar_sd, n)

        # -- a sample with no fission-track analysis: zero-fill, and flag it.
        # Drawn and then discarded rather than skipped, so that the random
        # stream a seeded caller sees does not depend on which rows are absent.
        if not aft_present.all():
            keep = aft_present
            ns = np.where(keep, ns, 0)
            age_obs = np.where(keep, age_obs, 0.0)
            age_sd = np.where(keep, age_sd, 0.0)
            age_sd_count = np.where(keep, age_sd_count, 0.0)
            counts = counts * keep[:, None]
            dpar = np.where(keep, dpar, 0.0)

        # -- helium grains.
        he_age, he_sd, mask = self._observe_he(truth["he_age"], rng, n_he)
        radius = np.asarray(truth["he_radius"], dtype=float) * np.exp(
            rng.normal(0.0, self.he.radius_rel_sd, he_age.shape))
        eU = np.asarray(truth["he_eU"], dtype=float) * np.exp(
            rng.normal(0.0, self.he.eU_rel_sd, he_age.shape))

        # Shuffled here rather than through :meth:`Aliquot.permute_grains` on
        # the way out, so that the per-grain arrays in ``meta`` -- which that
        # method knows nothing about -- stay aligned with the grains they
        # describe.  Grain order carries no information and randomising it
        # turns a consumer that is not permutation-invariant from quietly
        # wrong into visibly unstable.
        order = np.argsort(rng.random(he_age.shape), axis=1)
        he_age, he_sd, mask, radius, eU = (
            np.take_along_axis(v, order, axis=1)
            for v in (he_age, he_sd, mask, radius, eU))

        return Aliquot(
            aft_age=age_obs, aft_age_sd=age_sd, n_spontaneous=ns,
            length_counts=counts, length_bins=np.asarray(self.length_bins),
            dpar=dpar, dpar_sd=np.where(aft_present, self.aft.dpar_sd, 0.0),
            he_age=he_age * mask, he_age_sd=he_sd * mask,
            he_radius=radius * mask, he_eU=eU * mask, he_mask=mask,
            aft_mask=aft_present,
            meta={
                "config": self.to_config(),
                "effort": {k: np.asarray(v) for k, v in effort.items()},
                "aft_age_sd_counting": age_sd_count,
                "aft_age_sd_systematic": age_obs * np.sqrt(self.aft.systematic_rel_var),
                "he_age_sd_analytical": he_age * self.he.analytical_rel_sd * mask,
                "expected_counts": lam,
                # Reported grain g is stored grain ``grain_order[:, g]``: the
                # one thing a consumer needs to look a reported grain up in the
                # corpus (Stage 8 conditions on its measured Th/U).
                "grain_order": order,
            },
        )

    def _multinomial(self, n_l: np.ndarray, p: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        """Per-row multinomial draw, with ``pvals`` forced onto the simplex.

        ``Generator.multinomial`` rejects rows whose leading probabilities sum
        past one by a single ulp, which float32 corpus pdfs renormalised in
        float64 do often enough to matter over two million rows.
        """
        counts = np.zeros(p.shape, dtype=np.int64)
        total = p.sum(axis=1, keepdims=True)
        q = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
        q[:, -1] = np.maximum(0.0, 1.0 - q[:, :-1].sum(axis=1))
        rows = np.flatnonzero((n_l > 0) & (total[:, 0] > 0))
        if rows.size:
            counts[rows] = rng.multinomial(np.asarray(n_l)[rows], q[rows])
        return counts

    def _observe_he(self, age: np.ndarray, rng: np.random.Generator,
                    n_he: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log-normal dispersion, optional outliers, and the dated subset.

        ``n_he`` ``(n,)`` fixes how many grains each sample dated; it takes
        precedence over :attr:`AHeNoise.n_grains_range`, and may be zero.
        """
        age = np.atleast_2d(np.asarray(age, dtype=float))
        n, g = age.shape
        sd = self.he.rel_sd
        eps = rng.normal(0.0, sd, (n, g)) if sd else np.zeros((n, g))
        if self.he.outlier_fraction > 0.0 and self.he.outlier_rel_sd > 0.0:
            is_out = rng.random((n, g)) < self.he.outlier_fraction
            eps = eps + is_out * rng.normal(0.0, self.he.outlier_rel_sd, (n, g))
        obs = age * np.exp(eps)

        mask = np.ones((n, g), dtype=bool)
        if n_he is not None:
            mask = np.arange(g)[None, :] < np.minimum(n_he, g)[:, None]
        elif self.he.n_grains_range is not None:
            lo, hi = self.he.n_grains_range
            k = rng.integers(lo, min(hi, g) + 1, size=n)
            # Grain order is randomised on the way out, so masking the tail is
            # a uniformly random subset and needs no permutation of its own.
            mask = np.arange(g)[None, :] < k[:, None]
        return obs, obs * sd, mask

    # -- score --------------------------------------------------------------

    def log_prob(self, obs: Aliquot, pred: dict[str, Any], *,
                 components: bool = False) -> Any:
        """``log p(data | predicted observables)``, differentiable, in torch.

        Every term uses the uncertainties **reported in** ``obs`` rather than
        recomputing them from this config, so the same call scores real
        laboratory data with real error bars.  For data this model drew, the
        two are the same number by construction.

        The AFT age term is log-normal, which is where sampler and likelihood
        differ: ``observe`` draws a Poisson count.  The two agree to O(1/sqrt
        Ns) and ``docs/observation.md`` measures the disagreement.

        Grain properties (Dpar, radius, eU) are conditioned on, not scored --
        they are inputs to the forward model, and the prediction was made at
        the reported values.

        Args:
            obs: the data.
            pred: ``{"aft_age": (n,), "length_pdf": (n, n_bins),
                "he_age": (n, G)}`` from the Stage 2/3 ports, on any device.
            components: return the per-term breakdown as well.

        Returns:
            ``(n,)`` log-density, or ``(total, {"aft_age", "lengths", "he"})``.
        """
        import torch

        dev = pred["aft_age"].device
        dt = pred["aft_age"].dtype
        def as_t(x):
            return torch.as_tensor(np.asarray(x), dtype=dt, device=dev)

        tiny = torch.finfo(dt).tiny

        # -- AFT age, log-normal on the reported sigma.  A sample with no
        # tracks has no age, and scoring one would be scoring a clamp -- so it
        # is substituted out *before* the division rather than masked after,
        # which is what keeps an infinity from reaching the backward pass.
        # A sample with no AFT analysis at all (``aft_mask`` False) has no
        # length counts either, so the multinomial term below is zero for it
        # by itself.
        has_age = torch.as_tensor(obs.has_age, device=dev)
        age_obs = torch.where(has_age, as_t(obs.aft_age), torch.ones((), dtype=dt, device=dev))
        age_sd = torch.where(has_age, as_t(obs.aft_age_sd), torch.ones((), dtype=dt, device=dev))
        age_pred = pred["aft_age"].clamp_min(tiny)
        s = (age_sd / age_obs).clamp_min(1e-6)
        z = (torch.log(age_obs) - torch.log(age_pred)) / s
        lp_age = torch.where(
            has_age,
            -0.5 * z ** 2 - torch.log(s * np.sqrt(2 * np.pi)) - torch.log(age_obs),
            torch.zeros((), dtype=dt, device=dev))

        # -- confined lengths, multinomial over the smeared prediction.
        counts = as_t(obs.length_counts)
        p = pred["length_pdf"]
        p = p / p.sum(dim=1, keepdim=True).clamp_min(tiny)
        p = (p @ as_t(self._smear)).clamp_min(tiny)
        n_l = counts.sum(dim=1)
        lp_len = (counts * torch.log(p)).sum(dim=1) \
            + torch.lgamma(n_l + 1) - torch.lgamma(counts + 1).sum(dim=1)

        # -- helium grains, log-normal (plus the outlier component if any).
        # Absent grains are substituted out on the same argument as above.
        mask = torch.as_tensor(obs.he_mask, device=dev)
        one = torch.ones((), dtype=dt, device=dev)
        he_obs = torch.where(mask, as_t(obs.he_age), one)
        he_sd = torch.where(mask, as_t(obs.he_age_sd), one)
        he_pred = pred["he_age"].clamp_min(tiny)
        s_he = (he_sd / he_obs).clamp_min(1e-6)
        d = torch.log(he_obs) - torch.log(he_pred)
        lp_grain = self._lognormal_mixture(d, s_he) - torch.log(he_obs)
        lp_he = torch.where(mask, lp_grain, torch.zeros((), dtype=dt, device=dev)).sum(dim=1)

        total = lp_age + lp_len + lp_he
        if components:
            return total, {"aft_age": lp_age, "lengths": lp_len, "he": lp_he}
        return total

    def _lognormal_mixture(self, d: Any, s: Any) -> Any:
        """Log-density of ``d = log(obs) - log(pred)``, before the Jacobian.

        One Gaussian of width ``s``, or -- when outliers are switched on -- that
        component with weight ``1 - f`` against a broad one of width
        ``sqrt(s^2 + outlier^2)``.  The outlier width adds in quadrature
        because :meth:`_observe_he` draws it as an additive perturbation on top
        of the ordinary dispersion, so the mixture here is exactly the mixture
        that was sampled.
        """
        import torch

        def normal(width: Any) -> Any:
            return -0.5 * (d / width) ** 2 - torch.log(width * np.sqrt(2 * np.pi))

        f = self.he.outlier_fraction
        if f <= 0.0 or self.he.outlier_rel_sd <= 0.0:
            return normal(s)
        wide = torch.sqrt(s ** 2 + self.he.outlier_rel_sd ** 2)
        return torch.logaddexp(normal(s) + np.log1p(-f), normal(wide) + np.log(f))

    # -- config -------------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        """Plain data, for provenance attributes and YAML."""
        return {"aft": asdict(self.aft), "he": asdict(self.he)}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ObservationModel:
        he = dict(cfg.get("he", {}))
        if he.get("n_grains_range") is not None:
            he["n_grains_range"] = tuple(he["n_grains_range"])
        return cls(aft=AFTNoise(**cfg.get("aft", {})), he=AHeNoise(**he))

    def replace(self, **kwargs: Any) -> ObservationModel:
        """A copy with ``aft=`` / ``he=`` fields overridden by keyword."""
        aft = replace(self.aft, **kwargs.pop("aft", {}))
        he = replace(self.he, **kwargs.pop("he", {}))
        if kwargs:
            raise ValueError(f"unexpected keys {list(kwargs)}")
        return ObservationModel(aft=aft, he=he, length_bins=self.length_bins)

    def describe(self) -> str:
        """The model with this instance's numbers substituted in."""
        a, h = self.aft, self.he
        return "\n".join([
            "Observation model",
            (f"  AFT  {a.n_grains_counted} grains x {a.area_per_grain_um2:g} um^2"
             f" = {a.area_cm2 * 1e6:.0f}e-6 cm^2 counted"),
            f"       Ns ~ Poisson({_K_RHO:.0f} * age[Ma] * U[ppm] * area)",
            (f"       age scale x LogNormal(0, {a.zeta_rel_sd:g}) calibration"
             f" x LogNormal(0, {a.u_rel_sd:g}) uranium"),
            (f"       {a.n_lengths} confined lengths, multinomial, smeared by"
             f" {a.length_meas_sd_um:g} um"),
            f"       Dpar +- {a.dpar_sd:.3f} um ({a.n_dpar} pits)",
            (f"  AHe  age x LogNormal(0, {h.rel_sd:.3f})"
             f" = analytical {h.analytical_rel_sd:g} + dispersion {h.dispersion_rel_sd:g}"),
            f"       outliers {h.outlier_fraction:g} at width {h.outlier_rel_sd:g}",
            f"       radius +-{h.radius_rel_sd:g}, eU +-{h.eU_rel_sd:g} (relative)",
            f"       grains dated: {h.n_grains_range or 'all available'}",
        ])


# ---------------------------------------------------------------------------
# Reading the corpus
# ---------------------------------------------------------------------------

def truth_from_corpus(path: Any, index: Any = None) -> dict[str, np.ndarray]:
    """The modelled observables of some corpus rows, keyed by :data:`TRUTH_KEYS`.

    Read with ``h5py`` directly rather than through
    :class:`chai.eval.audit.Corpus`, to keep :mod:`chai.data` free of a
    dependency on the torch ports that the audit module pulls in.

    The aliquot's uranium content -- which sets the fission-track count, and
    with it the age's precision -- is taken as the **mean U of the stored
    helium grains**.  The corpus does not simulate a separate population of
    counted apatite grains, but it does not need to: those grains and these
    are draws from the same :class:`~chai.data.priors.GrainPrior`, so their
    mean is the right estimate of the rock's uranium content, and using one
    grain's U instead would add a spurious factor-of-several scatter to the
    counting precision.

    Args:
        path: the corpus HDF5 file.
        index: rows to read; ``None`` reads all of them.
    """
    import h5py

    sel = slice(None) if index is None else index
    with h5py.File(path, "r") as f:
        return {
            "aft_age": f["aft/age"][sel].astype(float),
            "length_pdf": f["aft/length_pdf"][sel].astype(float),
            "dpar": f["aft/dpar"][sel].astype(float),
            "U": f["he/U"][sel].astype(float).mean(axis=1),
            "he_age": f["he/corrected_age"][sel].astype(float),
            "he_radius": f["he/radius"][sel].astype(float),
            "he_eU": f["he/eU"][sel].astype(float),
        }


def observe_corpus(path: Any, index: Any, model: ObservationModel,
                   rng: np.random.Generator) -> Aliquot:
    """One noisy realisation of some corpus rows.  Convenience over
    :func:`truth_from_corpus` and :meth:`ObservationModel.observe`."""
    return model.observe(truth_from_corpus(path, index), rng)
