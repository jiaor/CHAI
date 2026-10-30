"""Prior sensitivity: the reweighting identity, and the posterior it feeds.

The study in ``docs/prior-sensitivity.md`` rests on one claim -- that a corpus
drawn under one hyperprior, reweighted, *is* a sample from another -- and on
the kernel posterior of :mod:`chai.eval.audit` continuing to mean what it
meant in Stage 1 once weights are attached to it.  Both are pinned here.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from chai.data.priors import ThermalHistoryPrior
from chai.eval.audit import Tolerance, conditional_spread
from chai.eval.sensitivity import (
    PriorVariant,
    hyper_log_density,
    log_weights,
    mismatch_matrix,
    standard_variants,
    variant_table,
    weighted_moments,
    weighted_posterior,
)

#: Cheap but structurally identical to the shipped prior: only the hyperprior
#: may differ between two priors that are compared by reweighting, and these
#: settings are shared by every prior in this file.
FAST = ThermalHistoryPrior(n_steps=256, n_features=64)


def _hyper(n: int, prior: ThermalHistoryPrior = FAST, seed: int = 0) -> dict[str, np.ndarray]:
    return {k: v.astype(float) for k, v in prior.sample(n, np.random.default_rng(seed)).hyper.items()}


# ---------------------------------------------------------------------------
# The density and the weights
# ---------------------------------------------------------------------------

def test_the_hyper_density_is_the_density_the_prior_samples_from():
    """A histogram of draws against the density they are supposed to have.

    ``drift`` is the one worth checking rather than asserting: the prior draws
    a total cooling over the horizon and stores it divided by ``t_max``, so a
    density written for the undivided quantity would be off by a factor of 100
    -- invisibly, because weights normalise it away until the day a variant
    moves ``t_max``.
    """
    h = _hyper(40_000)
    lp = hyper_log_density(FAST, h)
    assert np.isfinite(lp).all()

    for name, lo, hi, log_axis in (("tau", *FAST.tau_range, True),
                                   ("rate_sd", *FAST.rate_sd_range, True),
                                   ("drift", 0.0, 2.0, False),
                                   ("T_0", *FAST.T_present_range, False)):
        x = h[name]
        edges = (np.geomspace(lo, hi, 9) if log_axis else np.linspace(lo, hi, 9))
        counts, _ = np.histogram(x, bins=edges)
        # The marginal density is the joint divided by the other three, which
        # are constant across this variable: compare shapes, not levels.
        centre = np.sqrt(edges[:-1] * edges[1:]) if log_axis else 0.5 * (edges[:-1] + edges[1:])
        expect = (1.0 / centre if log_axis else np.ones_like(centre)) * np.diff(edges)
        expect = expect / expect.sum() * counts.sum()
        assert counts == pytest.approx(expect, rel=0.06), name


def test_reweighting_a_corpus_is_the_same_as_drawing_from_the_variant():
    """The study's central claim, checked against the only ground truth there is.

    Draw 8000 histories under the base prior and reweight them to a variant;
    draw 8000 more under the variant itself.  If the identity holds, the two
    describe the same distribution of ``T(t)`` -- and they must, because
    ``p(T | h)`` never moved.  The tolerance is Monte Carlo noise at the
    realised effective sample size, which for a half-box variant is half the
    draws.
    """
    base_batch = FAST.sample(8000, np.random.default_rng(0))
    T_base = base_batch.T
    hyper = {k: v.astype(float) for k, v in base_batch.hyper.items()}

    for v in (standard_variants(FAST)[0], standard_variants(FAST)[4]):   # tau_short, rate_wild
        direct = v.prior.sample(8000, np.random.default_rng(1)).T
        lw = log_weights(hyper, FAST, v)
        mean_w, sd_w, ess = weighted_moments(T_base, lw)
        mean_d, sd_d, _ = weighted_moments(direct, None)

        # Two independent estimates of the same mean, each with standard error
        # sd/sqrt(n_eff); 4 sigma of the difference is the honest bar.
        se = np.sqrt(sd_w ** 2 / ess + sd_d ** 2 / direct.shape[0])
        assert np.abs(mean_w - mean_d).max() < 4 * se.max(), v.name
        assert sd_w == pytest.approx(sd_d, rel=0.08), v.name


def test_a_tilted_variant_is_not_restricted_to_sub_boxes():
    """The same identity for a variant that is not a rectangle of the base."""
    tilt = PriorVariant("peaked", "tau known to ~25 Ma", FAST,
                        tilt=lambda h: -0.5 * ((np.log(h["tau"]) - np.log(25.0)) / 0.3) ** 2)
    hyper = _hyper(20_000)
    lw = log_weights(hyper, FAST, tilt)
    w = np.exp(lw)

    # The reweighted tau marginal must be the tilted one: log tau is normal
    # about log 25 with sd 0.3, truncated to the base range.
    logtau = np.log(hyper["tau"])
    mean = (w * logtau).sum() / w.sum()
    sd = np.sqrt((w * (logtau - mean) ** 2).sum() / w.sum())
    assert mean == pytest.approx(np.log(25.0), abs=0.02)
    assert sd == pytest.approx(0.3, rel=0.1)
    assert np.isfinite(lw).all(), "a tilt narrows the prior, it does not truncate it"


def test_a_sub_box_weight_is_exactly_a_stratification():
    """Half the box means half the corpus, at equal weight: no variance inflation."""
    hyper = _hyper(20_000)
    v = standard_variants(FAST)[0]                      # tau_short
    lw = log_weights(hyper, FAST, v)
    inside = hyper["tau"] <= v.prior.tau_range[1] * (1 + 1e-6)

    assert np.isfinite(lw[inside]).all() and not np.isfinite(lw[~inside]).any()
    assert np.allclose(lw[inside], 0.0), "every retained row must carry the same weight"
    ess = variant_table((v,), hyper, FAST)[0]["ess"]
    assert ess == pytest.approx(inside.sum(), rel=1e-6)


def test_a_structural_change_is_refused_rather_than_estimated():
    """A prior that moves the horizon is a different corpus, not a reweighting."""
    for field, value in (("t_max", 200.0), ("rate_max", 25.0), ("T_max", 300.0)):
        v = PriorVariant("structural", "needs its own corpus", replace(FAST, **{field: value}))
        with pytest.raises(ValueError, match="p\\(T \\| h\\)"):
            log_weights(_hyper(64), FAST, v)


def test_a_variant_with_no_overlap_is_refused_rather_than_returning_noise():
    hyper = _hyper(256)
    v = PriorVariant("impossible", "outside every draw",
                     replace(FAST, tau_range=(79.99, 80.0)))
    hyper["tau"] = np.full(hyper["tau"].shape, 10.0)
    with pytest.raises(ValueError, match="no support"):
        log_weights(hyper, FAST, v)


# ---------------------------------------------------------------------------
# The weighted posterior
# ---------------------------------------------------------------------------

def _synthetic_pool(n=4000, n_grid=16, seed=0):
    """The Stage 1 audit's synthetic pool: the age depends on node 3 alone."""
    rng = np.random.default_rng(seed)
    T = rng.normal(80.0, 20.0, (n, n_grid))
    return T, {
        "aft_age": 50.0 + 0.2 * T[:, 3],
        "mean_length": np.full(n, 15.0),
        "he_age": 30.0 + 0.2 * T[:, 11],
        "length_pdf": np.full((n, 8), 1.0 / 8 / 0.1),
    }


def test_the_unweighted_posterior_is_the_audits_conditional_spread():
    """Stage 1's numbers must survive the machinery that generalises them.

    With uniform prior weights and the query taken from the pool, this is the
    same estimator :func:`conditional_spread` runs, so the two must agree to
    float32 -- otherwise the baseline column of every table in the sensitivity
    study is quietly a different quantity from the audit it is compared with.
    """
    T, pool = _synthetic_pool()
    keys = ("aft_age", "he_age")
    tol = Tolerance(aft_age_rel=1e-3, he_age_rel=1e-3)
    rows = np.arange(16)

    ref = conditional_spread(T, pool, rows, tol, 0.1, ladder=(("x", keys),), device="cpu")
    queries = {k: v[rows] for k, v in pool.items()}
    z2 = mismatch_matrix(pool, queries, keys, tol, 0.1, device="cpu")
    got = weighted_posterior(torch.as_tensor(T, dtype=torch.float32), z2)

    assert got["ess"] == pytest.approx(ref["rungs"][0]["ess"], rel=1e-4)
    ratio = np.median(got["sd"] / ref["prior_sd"], axis=0)
    assert ratio == pytest.approx(ref["rungs"][0]["ratio"], abs=2e-3)


def test_a_zero_one_prior_weight_is_the_same_as_conditioning_on_the_subset():
    T, pool = _synthetic_pool()
    keys = ("aft_age",)
    tol = Tolerance(aft_age_rel=1e-2)
    keep = np.zeros(T.shape[0], dtype=bool)
    keep[::2] = True
    queries = {k: v[:8] for k, v in pool.items()}

    z2 = mismatch_matrix(pool, queries, keys, tol, 0.1, device="cpu")
    lw = torch.where(torch.as_tensor(keep), 0.0, -np.inf).float()
    weighted = weighted_posterior(torch.as_tensor(T, dtype=torch.float32), z2, lw)

    sub = {k: v[keep] for k, v in pool.items()}
    z2_sub = mismatch_matrix(sub, queries, keys, tol, 0.1, device="cpu")
    direct = weighted_posterior(torch.as_tensor(T[keep], dtype=torch.float32), z2_sub)

    for k in ("mean", "sd", "ess"):
        assert weighted[k] == pytest.approx(direct[k], rel=1e-4, abs=1e-4), k


def test_the_posterior_moments_are_the_weighted_moments():
    """Guards the centred-matmul form against the textbook one it replaced."""
    T, pool = _synthetic_pool(n=500)
    keys = ("aft_age",)
    tol = Tolerance(aft_age_rel=2e-2)
    queries = {k: v[:4] for k, v in pool.items()}
    rng = np.random.default_rng(3)
    lw = rng.normal(0.0, 1.0, T.shape[0])

    z2 = mismatch_matrix(pool, queries, keys, tol, 0.1, device="cpu")
    got = weighted_posterior(torch.as_tensor(T, dtype=torch.float32), z2,
                             torch.as_tensor(lw, dtype=torch.float32))
    for j in range(4):
        w = np.exp(lw - 0.5 * z2[j].numpy().astype(float))
        mean, sd, ess = weighted_moments(T, np.log(w))
        assert got["mean"][j] == pytest.approx(mean, abs=1e-3)
        assert got["sd"][j] == pytest.approx(sd, rel=1e-3)
        assert got["ess"][j] == pytest.approx(ess, rel=1e-3)


def test_a_query_from_outside_the_pool_is_scored_against_its_own_sigma():
    """The tolerance indexes queries, not pool rows -- sigma belongs to the data."""
    T, pool = _synthetic_pool(n=1000)
    queries = {k: v[[3, 7]] for k, v in pool.items()}
    tol = Tolerance(aft_age_rel=np.array([1e-4, 1e-1]))

    z2 = mismatch_matrix(pool, queries, ("aft_age",), tol, 0.1, device="cpu")
    got = weighted_posterior(torch.as_tensor(T, dtype=torch.float32), z2)
    assert got["sd"][0, 3] < 0.2 * got["sd"][1, 3], "a sharper sigma must bind harder"
    assert float(z2[0, 3]) == pytest.approx(0.0, abs=1e-6)
