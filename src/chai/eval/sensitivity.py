"""How much of a posterior is the prior?

PLAN.md's highest-severity risk is that a Goal 2 posterior is only ever as
defensible as the prior it was amortized over.  Stage 1 answered half of it by
replacing the prototype's hand-tuned family mixture with a generative model
whose every knob is written down (:mod:`chai.data.priors`).  This module
answers the other half: **if that model's knobs were set differently, how
differently would the same data be interpreted?**

The method, and why it needs no new corpus
------------------------------------------
The prior factorises.  A history is drawn by first drawing four
hyperparameters ``h = (tau, a, dT_drift, T_0)`` from a box, and then drawing a
path from ``p(T | h)`` -- a fixed recipe: a squared-exponential GP of length
``tau``, capped at ``rate_max``, integrated inside the temperature box.  Change
only the *hyperprior* and ``p(T | h)`` is untouched, so every history already in
the corpus is a valid draw from the new prior, merely at the wrong frequency.
Reweighting each row by

    ``w_i = p_alt(h_i) / p_base(h_i)``

therefore turns the shipped corpus into an exact sample from the alternative
prior -- the corpus stores ``h`` per row for exactly this reason -- and the
expensive part of any comparison, the forward models over 200 000 paths, is run
**once** and reused by every variant.  Two priors then differ only in a vector
of weights, which removes Monte Carlo noise from the comparison as well: the
variants share their pool, their queries and their forward evaluations, so a
difference between them is a difference in the prior and nothing else.

What is *not* reweightable is a change to ``p(T | h)`` itself -- the time span
``t_max``, the walls, the rate limit, the GP's stationarity.  Those priors put
mass where this corpus has no samples at all, no weight can manufacture them,
and :func:`log_weights` refuses rather than silently returning an estimate that
rests on nothing.  They need a fresh corpus, and that is the second half of the
study (``docs/prior-sensitivity.md``).

What is measured
----------------
Not "does the prior-predictive distribution move" -- of course it does, that is
what changing a prior means -- but **does the answer to a fixed question move**.
A query is one aliquot's worth of observables from a held-out history; the
posterior is the kernel-weighted pool of :mod:`chai.eval.audit`, now carrying
prior weights as well; and the reported quantity is the shift of the posterior
mean between two priors, in kelvin and in units of the baseline posterior's own
sd.  A shift well inside one posterior sd is a prior the data has overruled; a
shift of several is a prior the data cannot see past, and any credible interval
there is a statement about the prior.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from chai.data.priors import ThermalHistoryPrior
from chai.eval.audit import Tolerance, _pool_tensors, mismatch_to, tolerance_at

__all__ = [
    "STRUCTURAL_FIELDS",
    "PriorVariant",
    "hyper_log_density",
    "log_weights",
    "mismatch_matrix",
    "standard_variants",
    "variant_log_density",
    "variant_table",
    "weighted_moments",
    "weighted_posterior",
]

#: Fields of :class:`~chai.data.priors.ThermalHistoryPrior` that change
#: ``p(T | h)`` rather than ``p(h)``.  A variant that moves any of them cannot
#: be reached by reweighting an existing corpus, because the corpus holds no
#: draws from the conditional it describes -- not even badly-weighted ones.
STRUCTURAL_FIELDS = ("t_max", "T_min", "T_max", "rate_max", "n_features", "n_steps")

#: Relative slack on a support test.  The corpus stores hyperparameters as
#: float32, so a draw from ``LogUniform(8, 80)`` can come back as 80.000004 and
#: fall outside the very range it was drawn from.  A part in a million is far
#: below any distinction the study makes and far above float32's rounding.
_SUPPORT_RTOL = 1e-6


def _in_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    slack = _SUPPORT_RTOL * max(abs(lo), abs(hi), 1.0)
    return (x >= lo - slack) & (x <= hi + slack)


def _log_uniform_logpdf(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``log p(x)`` for ``x ~ LogUniform(lo, hi)``; ``-inf`` outside."""
    out = np.full(np.shape(x), -np.inf)
    ok = _in_range(x, lo, hi)
    out[ok] = -np.log(np.clip(x[ok], 1e-300, None)) - np.log(np.log(hi / lo))
    return out


def _uniform_logpdf(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``log p(x)`` for ``x ~ Uniform(lo, hi)``; ``-inf`` outside."""
    out = np.full(np.shape(x), -np.inf)
    out[_in_range(x, lo, hi)] = -np.log(hi - lo)
    return out


def hyper_log_density(prior: ThermalHistoryPrior,
                      hyper: dict[str, np.ndarray]) -> np.ndarray:
    """``log p(h)`` under ``prior``, for the hyperparameters a corpus stores.

    The four are independent by construction, so this is a sum of four
    marginals.  ``drift`` is the one that needs care: the prior draws a *total*
    cooling over the time span and stores it divided by ``t_max``
    (:meth:`~chai.data.priors.ThermalHistoryPrior.sample`), so its density
    carries the corresponding Jacobian.

    Args:
        prior: the prior whose density is wanted.
        hyper: ``tau``, ``rate_sd``, ``drift``, ``T_0``, each ``(n,)``, as
            stored in ``hyper/*`` in the corpus.
    """
    missing = {"tau", "rate_sd", "drift", "T_0"} - set(hyper)
    if missing:
        raise KeyError(f"hyper is missing {sorted(missing)}")
    drift_lo, drift_hi = prior.drift_total_range
    return (
        _log_uniform_logpdf(np.asarray(hyper["tau"], float), *prior.tau_range)
        + _log_uniform_logpdf(np.asarray(hyper["rate_sd"], float), *prior.rate_sd_range)
        + _uniform_logpdf(np.asarray(hyper["drift"], float) * prior.t_max, drift_lo, drift_hi)
        + np.log(prior.t_max)
        + _uniform_logpdf(np.asarray(hyper["T_0"], float), *prior.T_present_range)
    )


@dataclass(frozen=True)
class PriorVariant:
    """One alternative prior, and the geologist it belongs to.

    Args:
        name: short key, used in result files and tables.
        why: the one-line case for taking this prior seriously.  A variant
            nobody would write is not evidence about anything.
        prior: the alternative.  Its structural fields must match the corpus's
            prior; only the hyperprior box may move.
        tilt: an optional extra ``log`` factor on ``h``, so a variant is not
            restricted to sub-boxes of the base ranges.  It need only be
            correct up to a constant, since weights are normalised.
    """

    name: str
    why: str
    prior: ThermalHistoryPrior
    tilt: Callable[[dict[str, np.ndarray]], np.ndarray] | None = field(
        default=None, repr=False)


def variant_log_density(variant: PriorVariant,
                        hyper: dict[str, np.ndarray]) -> np.ndarray:
    """``log p_variant(h)``, up to an additive constant."""
    lp = hyper_log_density(variant.prior, hyper)
    if variant.tilt is not None:
        lp = lp + np.asarray(variant.tilt(hyper), dtype=float)
    return lp


def log_weights(hyper: dict[str, np.ndarray], base: ThermalHistoryPrior,
                variant: PriorVariant) -> np.ndarray:
    """``log w_i = log p_variant(h_i) - log p_base(h_i)``, centred on its max.

    Centring keeps the weights representable; every consumer normalises, so
    only differences matter.  Rows outside the variant's support get ``-inf``
    and drop out of every sum they appear in.

    Raises:
        ValueError: if the variant changes ``p(T | h)`` rather than ``p(h)``.
            Reweighting cannot reach such a prior from this corpus: the corpus
            holds no draws from the conditional it describes, so an importance
            estimate would not be imprecise, it would be undefined.  Those
            variants need their own corpus.
    """
    moved = [f for f in STRUCTURAL_FIELDS
             if getattr(base, f) != getattr(variant.prior, f)]
    if moved:
        raise ValueError(
            f"variant {variant.name!r} moves {moved}, which changes p(T | h) and not "
            "just the hyperprior: no reweighting of this corpus can represent it. "
            "Simulate a corpus under it instead (scripts/simulate.py).")

    lw = variant_log_density(variant, hyper) - hyper_log_density(base, hyper)
    finite = lw[np.isfinite(lw)]
    if finite.size == 0:
        raise ValueError(f"variant {variant.name!r} has no support in this corpus")
    return lw - finite.max()


def weighted_moments(x: np.ndarray, log_w: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted mean and sd down axis 0, plus the effective sample size.

    The ESS ``(sum w)^2 / sum w^2`` is the number that decides whether a
    variant's numbers may be read at all: a prior that keeps a twentieth of the
    corpus is estimated from a twentieth of the corpus, however many rows the
    file has.
    """
    x = np.asarray(x, dtype=float)
    if log_w is None:
        return x.mean(axis=0), x.std(axis=0), float(x.shape[0])
    w = np.exp(log_w - log_w.max())
    sw = w.sum()
    ess = float(sw * sw / (w * w).sum())
    mean = (w[:, None] * x).sum(axis=0) / sw if x.ndim == 2 else (w * x).sum() / sw
    var = ((w[:, None] * (x - mean) ** 2).sum(axis=0) / sw if x.ndim == 2
           else (w * (x - mean) ** 2).sum() / sw)
    return mean, np.sqrt(var), ess


def mismatch_matrix(pool: dict[str, np.ndarray], queries: dict[str, np.ndarray],
                    keys: tuple[str, ...], tol: Tolerance, bin_width: float,
                    device: str = "cuda") -> torch.Tensor:
    """``(n_query, n_pool)`` squared standardised mismatch, computed once.

    The mismatch depends on the data and the pool, and **not** on the prior --
    that enters only as a weight on the pool.  So every prior variant in a
    study shares this matrix, and the cost of a variant after the first is a
    handful of elementwise operations rather than another pass over the pool.

    ``tol`` fields may be arrays over *queries* rather than over pool rows: the
    analytical uncertainty belongs to the measurement being scored, and there
    is exactly one measurement per row of the returned matrix.
    """
    dev = torch.device(device)
    cols = _pool_tensors(pool, bin_width, dev)
    qcols = _pool_tensors(queries, bin_width, dev)
    n_q = next(iter(qcols.values())).shape[0]

    rows = []
    for j in range(n_q):
        q = {k: qcols[k][j] for k in keys if k != "length_pdf"}
        if "length_pdf" in keys:
            q["length_cdf"] = qcols["length_cdf"][j]
        rows.append(mismatch_to(cols, q, keys, tolerance_at(tol, j), bin_width))
    return torch.stack(rows)


def weighted_posterior(T_pool: torch.Tensor, z2: torch.Tensor,
                       log_prior_w: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """The kernel posterior over ``T(t)`` for every query, under one prior.

    Pool member ``i`` carries weight ``p_prior(h_i) * exp(-z_i^2 / 2)``: the
    prior's own weight times the same soft acceptance
    :func:`~chai.eval.audit.conditional_spread` uses.  With uniform prior
    weights the two agree exactly, which is what makes the baseline column of
    every table in this study comparable with the Stage 1 audit.

    Args:
        T_pool: ``(n_pool, n_grid)`` paths, on ``z2``'s device.
        z2: ``(n_query, n_pool)`` from :func:`mismatch_matrix`.
        log_prior_w: ``(n_pool,)`` log prior weights, or ``None`` for the
            corpus's own prior.

    Returns:
        ``mean`` and ``sd`` over the nodes, ``(n_query, n_grid)``, and ``ess``
        per query.
    """
    logw = -0.5 * z2 if log_prior_w is None else log_prior_w[None, :] - 0.5 * z2
    logw = logw - logw.max(dim=1, keepdim=True).values
    w = torch.exp(logw)
    sw = w.sum(dim=1)
    ess = sw * sw / (w * w).sum(dim=1)

    # Centred before the matmul: E[T^2] - E[T]^2 on raw temperatures cancels
    # four significant figures of a float32 sum when the posterior is tight,
    # and the posterior being tight is the case this study is about.
    centre = T_pool.mean(dim=0)
    Tc = T_pool - centre
    mean = (w @ Tc) / sw[:, None]
    var = ((w @ (Tc * Tc)) / sw[:, None] - mean * mean).clamp_min(0.0)
    return {
        "mean": (mean + centre).cpu().numpy(),
        "sd": var.sqrt().cpu().numpy(),
        "ess": ess.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------

def _log_mid(lo: float, hi: float) -> float:
    """The point that halves a log-uniform range, so two bands are equiprobable."""
    return float(np.sqrt(lo * hi))


def standard_variants(base: ThermalHistoryPrior | None = None) -> tuple[PriorVariant, ...]:
    """The priors this study compares against the shipped one.

    Each is a prior somebody would actually write down.  The bands split a base
    range at its median so that the two halves are equiprobable under the base
    prior -- each keeps half the corpus, and neither is estimated from a corner
    of it.  ``quiet`` is the conjunction a geologist working a stable craton
    would write, and it is the most informative prior here (a quarter of the
    corpus); ``tau_peaked`` is not a sub-box at all, and is here to show that
    the method covers any hyperprior with support inside the base, not only
    rectangles.
    """
    base = base or ThermalHistoryPrior()
    tau_mid = _log_mid(*base.tau_range)
    rate_mid = _log_mid(*base.rate_sd_range)
    drift_mid = 0.5 * sum(base.drift_total_range)
    T0_mid = 0.5 * sum(base.T_present_range)

    def peaked(hyper: dict[str, np.ndarray], centre: float = 25.0,
               width: float = 0.35) -> np.ndarray:
        z = (np.log(np.asarray(hyper["tau"], float)) - np.log(centre)) / width
        return -0.5 * z * z

    return (
        PriorVariant("tau_short", "many short thermal episodes: a restless margin",
                     replace(base, tau_range=(base.tau_range[0], tau_mid))),
        PriorVariant("tau_long", "one or two slow episodes: a decaying orogen",
                     replace(base, tau_range=(tau_mid, base.tau_range[1]))),
        PriorVariant("tau_peaked", "a geologist confident the timescale is ~25 Ma",
                     base, tilt=peaked),
        PriorVariant("rate_calm", "the cooling rate barely fluctuates",
                     replace(base, rate_sd_range=(base.rate_sd_range[0], rate_mid))),
        PriorVariant("rate_wild", "large fluctuations, which the walls make episodic",
                     replace(base, rate_sd_range=(rate_mid, base.rate_sd_range[1]))),
        PriorVariant("drift_small", "little net exhumation over the time span",
                     replace(base, drift_total_range=(base.drift_total_range[0], drift_mid))),
        PriorVariant("drift_large", "deep burial exhumed over the time span",
                     replace(base, drift_total_range=(drift_mid, base.drift_total_range[1]))),
        PriorVariant("quiet", "the stable-craton prior: calm rate and little drift",
                     replace(base, rate_sd_range=(base.rate_sd_range[0], rate_mid),
                             drift_total_range=(base.drift_total_range[0], drift_mid))),
        PriorVariant("surface_cold", "a control: present-day temperature is measured, "
                                     "not inferred, so the deep past must not move",
                     replace(base, T_present_range=(base.T_present_range[0], T0_mid))),
    )


def variant_table(variants: tuple[PriorVariant, ...], hyper: dict[str, np.ndarray],
                  base: ThermalHistoryPrior) -> list[dict[str, Any]]:
    """Name, rationale, ESS and retained fraction for each variant.  For reports."""
    rows = []
    for v in variants:
        lw = log_weights(hyper, base, v)
        w = np.exp(lw)
        rows.append({
            "name": v.name,
            "why": v.why,
            "ess": float(w.sum() ** 2 / (w * w).sum()),
            "ess_fraction": float(w.sum() ** 2 / (w * w).sum() / w.size),
            "in_support": float(np.isfinite(lw).mean()),
        })
    return rows
