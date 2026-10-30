"""Does the answer fit the data: the posterior predictive, datum by datum.

The head always returns a posterior, even for data no history under the model
can produce.  This says when it has: for each datum (the AFT age, the mean
projected length, each helium age), where the observed value falls among what
the posterior's paths predict *with the model's own observation noise added*,
as a two-sided tail probability.  A datum past the threshold is listed, with
the interval it should have fallen in; the history is still returned, because
hiding it would hide what the model tried.

The predictive distribution is the paths' mixture of the observation model's
noise, so its CDF at the observed value is the mean over paths of each noise
distribution's CDF.  That is smooth in the observed value and not limited to
``1/n_paths`` resolution, which counting paths past the datum would be:

- a helium age: log-normal, the model's 12.3% (analytical and dispersion);
- the AFT age: a Poisson count, whose mean follows the path's age at the
  sample's count rate (tracks per Ma), times the model's systematic factor
  (zeta and U, 3.6%) integrated by Gauss–Hermite.  The Poisson CDF is the
  regularized incomplete gamma with a half-count continuity correction, so a
  datum exactly at the mean scores near 0.5;
- the mean projected length: normal, each path's own spread plus the 0.15 µm
  measurement error, over the root of the number of lengths.

The per-datum level depends on how many data the sample has, ``1 - (1 - c)^(1/k^b)``,
with ``c`` and ``b`` calibrated so that a sample the model fits by construction
warns about one time in twenty whatever its number of data
(``scripts/misfit_study.py``).  ``b = 1`` would be Šidák's correction for
independent data; the posterior predictive has seen the data, so its tail
probabilities pile up in the middle, and the more so the fewer the data.  Pure NumPy/SciPy:
the browser port (``web/js/fit.js``) computes the same numbers from the same
predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import gammaincc, ndtr

__all__ = ["Datum", "Fit", "FIT_BETA", "FIT_C", "FIT_MIN_LENGTHS", "check", "datum_level"]

#: The sample-level false-alarm rate the level is set for, and the constants of :func:`datum_level`
#: that give it, calibrated on the members' prior predictive (docs/misfit.md §3).
FIT_TARGET = 0.05
FIT_C, FIT_BETA = 0.46, 1.5
#: Fewer lengths than this and the mean length is not tested (its normal approximation is poor).
FIT_MIN_LENGTHS = 5
#: Probabilists' Gauss–Hermite nodes for the AFT age's systematic factor.
GH_NODES = 16
#: Bisection steps for the predicted intervals.
_BISECT = 60
#: 238U total decay constant, /Ma (chai.data.observe.LAMBDA_D).
LAMBDA_D_MA = 1.55125e-10 * 1e6
CAUTION = "The histories do not fit data within error; treat with caution"


def datum_level(k: int, c: float = FIT_C, beta: float = FIT_BETA) -> float:
    """The per-datum two-sided level for a sample with ``k`` tested data: ``1 - (1 - c)^(1/k^beta)``."""
    return 1.0 - (1.0 - c) ** (1.0 / max(k, 1) ** beta)


@dataclass(frozen=True)
class Datum:
    """One datum's place in the posterior predictive.

    Attributes:
        label: ``"AFT"``, ``"Mean length"``, ``"He 1"``, ...
        observed: the value, Ma or µm.
        p: two-sided tail probability, ``2 min(F, 1 - F)``.
        F: the predictive CDF at the observed value.
        interval: the predictive interval at the level the sample is tested at.
        outside: ``p`` below that level.
        unit: ``"Ma"`` or ``"µm"``.
    """

    label: str
    observed: float
    p: float
    F: float
    interval: tuple[float, float]
    outside: bool
    unit: str


@dataclass
class Fit:
    """The posterior predictive check for one sample.

    Attributes:
        data: one :class:`Datum` per tested datum.
        level: the per-datum level (:func:`datum_level` of their number).
        n_paths: paths the predictions came from.
        warnings: one line per datum outside, and :data:`CAUTION` first when
            several are, or the AFT age is.
    """

    data: list[Datum]
    level: float
    n_paths: int
    warnings: list[str] = field(default_factory=list)

    @property
    def misfit(self) -> list[Datum]:
        return [d for d in self.data if d.outside]

    @property
    def caution(self) -> bool:
        m = self.misfit
        return len(m) >= 2 or any(d.label == "AFT" for d in m)


# ---------------------------------------------------------------------------
# The predictive CDFs, one per kind of datum, as a function of a candidate value
# ---------------------------------------------------------------------------

def _gh(sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes (as log factors) and weights for E[f(exp(sd Z))], Z standard normal."""
    x, w = np.polynomial.hermite_e.hermegauss(GH_NODES)
    return sd * x, w / w.sum()


def _he_cdf(pred: np.ndarray, rel_sd: float):
    logp = np.log(np.maximum(pred, 1e-12))
    return lambda x: float(ndtr((np.log(x) - logp) / rel_sd).mean())


def _aft_cdf(pred: np.ndarray, age: float, ns: int, sys_var: float):
    """The observed age's CDF: count ~ Poisson(rate x g(path age)), times a log-normal systematic factor."""
    g = lambda t: np.expm1(LAMBDA_D_MA * np.maximum(t, 0.0))  # noqa: E731
    rate = max(ns, 1) / g(age)                                   # tracks per unit of g, from the sample's own count
    lam = rate * g(pred)                                         # (n,) the count each path predicts
    z, w = _gh(float(np.sqrt(sys_var)))
    # age_obs <= x  <=>  count x factor <= rate g(x); the factor multiplies the whole age.
    def F(x: float) -> float:
        m = rate * g(x) * np.exp(-z)                             # (nodes,) the count threshold per factor
        q = gammaincc(m[:, None] + 0.5, lam[None, :])            # P(N <= m), continuity-corrected
        return float((w @ q).mean())
    return F


def _length_cdf(pdf: np.ndarray, bins: np.ndarray, n: int, meas_sd: float):
    p = pdf / np.maximum(pdf.sum(axis=1, keepdims=True), 1e-300)
    mean = p @ bins
    var = np.clip(p @ bins ** 2 - mean ** 2, 0.0, None) + meas_sd ** 2
    se = np.sqrt(var / n)
    return lambda x: float(ndtr((x - mean) / se).mean())


def _invert(F, lo: float, hi: float, target: float, log: bool) -> float:
    """``x`` with ``F(x) = target`` by bisection, in log space for ages."""
    a, b = (np.log(lo), np.log(hi)) if log else (lo, hi)
    for _ in range(_BISECT):
        m = 0.5 * (a + b)
        if F(np.exp(m) if log else m) < target:
            a = m
        else:
            b = m
    m = 0.5 * (a + b)
    return float(np.exp(m) if log else m)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def aft_count(aft: Any, sys_var: float) -> int:
    """The spontaneous count the model reads: as given, or backed out of ``age_sd`` (``Model.to_aliquot``)."""
    if aft.n_spontaneous is not None:
        return int(aft.n_spontaneous)
    return max(round(1.0 / ((aft.age_sd / aft.age) ** 2 - sys_var)), 1)


def _fmt(x: float) -> str:
    """Three significant figures for the usual ranges, keeping a trailing zero (33.0, not 33)."""
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def check(sample: Any, prediction: Any, observation: Any, *, c: float = FIT_C) -> Fit:
    """The posterior predictive check of ``sample`` against ``prediction`` (:func:`chai.api.predict`).

    Args:
        sample: a :class:`chai.api.Sample`.
        prediction: what the posterior's paths predict for it.
        observation: the :class:`~chai.data.observe.ObservationModel` the head was trained on.
        c: the sample-level constant of :func:`datum_level`.
    """
    sys_var = float(observation.aft.systematic_rel_var)
    he_rel = float(observation.he.rel_sd)
    meas_sd = float(observation.aft.length_meas_sd_um)
    items: list[tuple[str, float, Any, bool, str]] = []   # label, observed, F, log-space, unit
    a = sample.aft
    if a is not None:
        items.append(("AFT", float(a.age), _aft_cdf(prediction.aft_age, a.age, aft_count(a, sys_var), sys_var), True, "Ma"))
        L = np.asarray(a.projected_lengths, dtype=float)
        if L.size >= FIT_MIN_LENGTHS:
            items.append(("Mean length", float(L.mean()), _length_cdf(prediction.length_pdf, prediction.bins, L.size, meas_sd),
                          False, "µm"))
    for k, g in enumerate(sample.grains):
        items.append((f"He {k + 1}", float(g.corrected_age), _he_cdf(prediction.he_age[:, k], he_rel), True, "Ma"))

    level = datum_level(len(items), c)
    data, lines = [], []
    for label, x, F, log, unit in items:
        f = F(x)
        p = min(1.0, 2.0 * min(f, 1.0 - f))
        lo_b, hi_b = (x * 1e-3 + 1e-6, x * 1e3 + 1e3) if log else (0.0, 20.0)
        interval = (_invert(F, lo_b, hi_b, level / 2, log), _invert(F, lo_b, hi_b, 1 - level / 2, log))
        d = Datum(label=label, observed=x, p=p, F=f, interval=interval, outside=p < level, unit=unit)
        data.append(d)
        if d.outside:
            lines.append(f"{label} ({_fmt(x)} {unit}) is outside what the histories predict, "
                         f"{100 * (1 - level):.3g}%: {_fmt(interval[0])}–{_fmt(interval[1])} {unit}")
    fit = Fit(data=data, level=level, n_paths=len(prediction.paths), warnings=lines)
    if fit.caution:
        fit.warnings.insert(0, CAUTION)
    return fit
