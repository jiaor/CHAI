"""Stage 8: the corpus as a reference posterior, by importance sampling with the exact likelihood.

Mechanics, on the tiny corpus and on CPU: the weighted summaries, the grain
bookkeeping between the corpus and a reported aliquot, and -- the claim the
study rests on -- that the helium screen drops nothing that matters and says
how much it dropped.  The forward models are the real ports on the tiny
corpus's 16-node paths; the *data* here are made by those same ports from a
true path, so the posterior has a truth to be near.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chai.data.observe import ObservationModel
from chai.eval.reference import (
    QUANTILES,
    Query,
    ReferenceConfig,
    ReferencePool,
    aft_log_likelihood,
    he_log_likelihood,
    he_log_likelihood_max,
    marginal_w1,
    posterior_summary_from_weights,
    reference_posterior,
    weighted_quantiles,
)
from chai.forward.torchimpl.ketcham import aft_forward
from chai.forward.torchimpl.rdaam import rdaam_age

torch.set_default_dtype(torch.float32)
N_POOL = 240
CFG = ReferenceConfig(aft_chunk=128, he_batch=256, he_block=48, max_he_rows=N_POOL, tail_sample=0)


# ---------------------------------------------------------------------------
# Weighted summaries
# ---------------------------------------------------------------------------

def test_weighted_quantiles_agree_with_numpy_on_uniform_weights():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal((5001, 3)))
    w = torch.ones(5001, dtype=torch.float64)
    q = weighted_quantiles(x, w).numpy()
    ref = np.quantile(x.numpy(), QUANTILES, axis=0, method="inverted_cdf")
    np.testing.assert_allclose(q, ref, atol=2e-3)


def test_weighted_quantiles_follow_the_weights():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    w = torch.tensor([0.0, 0.0, 1.0, 3.0])
    q = weighted_quantiles(x, w, (0.1, 0.5, 0.9))
    assert q[:, 0].tolist() == [2.0, 3.0, 3.0]


def test_marginal_w1_is_the_mean_shift_for_a_shifted_copy():
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.normal(80.0, 5.0, (20_000, 2)))
    w = torch.ones(20_000, dtype=torch.float64)
    d = marginal_w1(x, w, x + torch.tensor([3.0, 0.0]))
    assert abs(float(d[0]) - 3.0) < 0.1 and float(d[1]) < 1e-9


@pytest.fixture(scope="module")
def pool(tiny_corpus) -> ReferencePool:
    return ReferencePool(tiny_corpus, device="cpu", max_rows=N_POOL)


def test_summary_with_uniform_weights_is_the_plain_moments(pool):
    lw = torch.zeros(len(pool))
    s = posterior_summary_from_weights(pool, lw)
    T = pool.T.double().numpy()
    np.testing.assert_allclose(s["mean"], T.mean(0), rtol=1e-6)
    np.testing.assert_allclose(s["sd"], T.std(0), rtol=1e-6)
    assert abs(s["ess"] - len(pool)) < 1e-3 and s["n_kept"] == len(pool)
    assert s["quantiles"].shape == (len(QUANTILES), pool.n_grid)
    assert s["sd_halves"].shape == (2, pool.n_grid)
    np.testing.assert_allclose(s["sd_halves"], T[0::2].std(0)[None].repeat(2, 0), rtol=0.3)
    assert s["hyper_mean"].shape == (3,) and s["fn_quantiles"].shape[0] == len(QUANTILES)


def test_summary_ignores_minus_inf_rows_and_reports_the_ess(pool):
    lw = torch.full((len(pool),), -np.inf)
    lw[:10] = 0.0
    s = posterior_summary_from_weights(pool, lw)
    np.testing.assert_allclose(s["mean"], pool.T[:10].double().mean(0).numpy(), rtol=1e-6)
    assert abs(s["ess"] - 10.0) < 1e-6 and s["n_kept"] == 10
    with pytest.raises(ValueError):
        posterior_summary_from_weights(pool, torch.full((len(pool),), -np.inf))


# ---------------------------------------------------------------------------
# A query made by the forward models themselves
# ---------------------------------------------------------------------------

def _consistent_query(pool: ReferencePool, i: int, seed: int, *, n_he: int = 2, aft: bool = True
                      ) -> tuple[Query, np.ndarray]:
    """Row ``i`` of the pool observed as one aliquot whose modelled observables come from the
    ports at its own grains -- so the exact likelihood is exactly right for it."""
    rng = np.random.default_rng(seed)
    T = pool.T[i:i + 1]
    t = pool.t.expand(1, -1)
    radius = np.array([[45.0, 80.0]])
    U = np.array([[20.0, 60.0]])
    Th = np.array([[30.0, 90.0]])
    dpar = torch.tensor([2.0])
    with torch.no_grad():
        a = aft_forward(t, T, dpar, n_sub=CFG.n_sub_aft)
        he = torch.stack([rdaam_age(t, T, torch.tensor([radius[0, g]]), torch.tensor([U[0, g]]),
                                    torch.tensor([Th[0, g]]), n_sub=CFG.n_sub_he, rdim=CFG.rdim,
                                    use_checkpoint=False)["corrected_age"] for g in range(2)], 1)
    truth = {"aft_age": a["age"].numpy(), "length_pdf": a["length_pdf"].double().numpy(),
             "dpar": dpar.numpy(), "U": np.array([30.0]), "he_age": he.double().numpy(),
             "he_radius": radius, "he_eU": U + 0.235 * Th}
    model = ObservationModel()
    obs = model.observe(truth, rng, effort={"n_he_grains": np.array([n_he]),
                                            "aft_present": np.array([aft]),
                                            "n_lengths": np.array([60])})
    q = Query.from_observation(obs, 0, U_true=U[0], Th_true=Th[0], T_true=pool.T[i].numpy(), row=i)
    return q, obs.meta["grain_order"][0]


def test_query_conditions_on_the_reported_eu_split_by_the_measured_th_u(pool):
    q, order = _consistent_query(pool, 3, seed=0)
    U_true, Th_true = np.array([20.0, 60.0]), np.array([30.0, 90.0])
    # the reported grains are a permutation of the stored ones; Th/U follows the grain
    np.testing.assert_allclose(q.he_Th / q.he_U, (Th_true / U_true)[order])
    # and the parents conditioned on reproduce the reported eU exactly
    np.testing.assert_allclose(q.he_U + 0.235 * q.he_Th, q.obs.he_eU[0])
    assert q.n_he == 2 and q.has_aft


def test_helium_bound_is_a_bound_and_is_attained(pool):
    q, _ = _consistent_query(pool, 5, seed=1)
    model = ObservationModel()
    top = he_log_likelihood_max(q, model)
    lpH = he_log_likelihood(pool, torch.arange(0, 60), q, model, CFG)
    assert lpH.shape == (60,) and torch.isfinite(lpH).all()
    assert float(lpH.max()) <= top + 1e-4
    # a prediction equal to the data reaches the bound
    from chai.eval.reference import _he_view, _repeat
    view = _repeat(_he_view(q.obs), 1)
    pred = {"aft_age": torch.ones(1), "length_pdf": torch.ones((1, 200)),
            "he_age": torch.as_tensor(np.where(view.he_mask, view.he_age, 1.0), dtype=torch.float32)}
    assert abs(float(model.log_prob(view, pred)[0]) - top) < 1e-6


def test_modality_switches_off_the_missing_half(pool):
    model = ObservationModel()
    q_he, _ = _consistent_query(pool, 7, seed=2, aft=False)
    assert not q_he.has_aft
    assert torch.equal(aft_log_likelihood(pool, q_he, model, CFG), torch.zeros(len(pool)))
    q_aft, _ = _consistent_query(pool, 7, seed=3, n_he=0)
    assert q_aft.n_he == 0
    r = reference_posterior(pool, q_aft, model, CFG)
    assert r["n_he_rows"] == 0 and r["screen_closed"] and r["log_dropped_bound"] == -np.inf
    assert torch.isfinite(r["log_w"]).all()


def test_screen_drops_nothing_that_matters_and_says_how_much_it_dropped(pool):
    """The whole point.  Brute force evaluates every row's helium; the screen stops early.
    Where both evaluated a row they agree exactly; the rows only brute force saw carry
    less mass than the screen's own bound promised; the summaries agree."""
    model = ObservationModel()
    q, _ = _consistent_query(pool, 11, seed=4)
    full = reference_posterior(pool, q, model, ReferenceConfig(**{**CFG.to_config(), "screen_margin": 1e9}))
    assert full["n_he_rows"] == len(pool) and torch.isfinite(full["log_w"]).all()
    screened = reference_posterior(pool, q, model, ReferenceConfig(**{**CFG.to_config(), "screen_margin": 4.0}))
    seen = torch.isfinite(screened["log_w"])
    assert 0 < int(seen.sum()) < len(pool), "the screen should stop before the end on this query"
    assert torch.allclose(screened["log_w"][seen], full["log_w"][seen])
    w = torch.exp(full["log_w"] - full["log_w"].max()).double()
    dropped = float(torch.log(w[~seen].sum()) - torch.log(w[seen].sum()))
    assert dropped <= screened["log_dropped_bound"] + 1e-6
    assert screened["log_dropped_bound"] < -4.0
    # e^-4 of the mass cannot move a summary by much
    np.testing.assert_allclose(screened["mean"], full["mean"], atol=0.05 * full["sd"].max() + 0.5)
    assert abs(screened["ess"] - full["ess"]) / full["ess"] < 0.05


def test_the_posterior_finds_the_truth(pool):
    """The truth is a pool member observed with the ports' own predictions: it should carry a
    large share of the weight, and the recent nodes should be pinned far below the prior."""
    model = ObservationModel()
    q, _ = _consistent_query(pool, 11, seed=5)
    r = reference_posterior(pool, q, model, CFG)
    w = torch.exp(r["log_w"] - r["log_w"][torch.isfinite(r["log_w"])].max())
    w = torch.where(torch.isfinite(r["log_w"]), w, torch.zeros_like(w))
    assert int(torch.argmax(w)) == 11 or float(w[11] / w.sum()) > 0.05
    prior_sd = pool.T.double().std(0).numpy()
    recent = slice(0, pool.n_grid // 3)
    assert (r["sd"][recent] < 0.7 * prior_sd[recent]).all()
    assert np.abs(r["mean"] - q.T_true)[recent].max() < 2.0 * r["sd"][recent].max() + 5.0
    assert r["time_aft"] > 0 and r["time_he"] > 0 and r["n_pool"] == len(pool)


def test_an_open_screen_samples_its_tail_and_weights_it_up(pool):
    """Capped before it can close, the ordered pass is completed by a random sample of the
    rest, each standing for ``n_rest / n_sample`` rows: the estimator stays unbiased and the
    tail's share of the mass is measured."""
    model = ObservationModel()
    q, _ = _consistent_query(pool, 11, seed=6)
    full = reference_posterior(pool, q, model, ReferenceConfig(**{**CFG.to_config(), "screen_margin": 1e9}))
    capped = ReferenceConfig(**{**CFG.to_config(), "max_he_rows": 96, "tail_sample": 48, "screen_margin": 1e9})
    r = reference_posterior(pool, q, model, capped)
    assert not r["screen_closed"] and r["n_he_rows"] == 96 and r["n_tail"] == 48
    seen = torch.isfinite(r["log_w"])
    assert int(seen.sum()) == 96 + 48
    ordered = torch.argsort(aft_log_likelihood(pool, q, model, CFG), descending=True)[:96]
    head = torch.zeros(len(pool), dtype=torch.bool)
    head[ordered] = True
    tail = seen & ~head
    assert int(tail.sum()) == 48
    assert torch.allclose(r["log_w"][head], full["log_w"][head])
    offset = float(np.log((len(pool) - 96) / 48))
    assert torch.allclose(r["log_w"][tail] - offset, full["log_w"][tail], atol=1e-4)
    assert 0.0 <= r["tail_mass"] <= 1.0
    # the tail estimate of the dropped mass sits under the bound
    w = torch.exp(full["log_w"] - full["log_w"].max()).double()
    true_dropped = float(w[~head].sum() / w.sum())
    assert np.log(true_dropped / (1 - true_dropped)) <= r["log_dropped_bound"] + 1e-6
    # a query with no fission-track data has nothing to order by and takes no tail
    q_he, _ = _consistent_query(pool, 7, seed=7, aft=False)
    r_he = reference_posterior(pool, q_he, model, capped)
    assert r_he["n_tail"] == 0 and np.isnan(r_he["tail_mass"]) and not r_he["screen_closed"]
    assert int(torch.isfinite(r_he["log_w"]).sum()) == 96
