"""Stage 13: the exact likelihood as a correction -- the weights, their diagnostics, and the screened corpus.

On the tiny corpus and on CPU, with the likelihood replaced by a known
function of the path where the claim is about the estimator rather than the
physics, so every number here has an exact answer to be checked against.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chai.data.aliquot import Aliquot
from chai.data.observe import ObservationModel
from chai.eval import refine as R
from chai.eval.reference import Query, ReferencePool
from chai.models.flow import FlowConfig
from chai.models.gaussian import AmortizedPosterior
from chai.train.data import AFT_FLAGS, AFT_INPUTS, GRAIN_INPUTS

N_POOL = 3_000


@pytest.fixture(scope="module")
def pool(tiny_corpus) -> ReferencePool:
    return ReferencePool(tiny_corpus, device="cpu", max_rows=N_POOL)


def _net(latent, *, unconditional: bool = False, seed: int = 0) -> AmortizedPosterior:
    torch.manual_seed(seed)
    net = AmortizedPosterior(latent, n_aft=len(AFT_INPUTS), n_bins=200, n_grain=len(GRAIN_INPUTS),
                             aft_flags=AFT_FLAGS, head=FlowConfig(n_layers=2, d_hidden=32),
                             meta={"config": {"unconditional": unconditional,
                                              "dequantise": {"width": 10.0, "jitter": 1.0, "noise_sd": 0.5}}})
    return net.eval()


def _query(n_he: int = 2) -> Query:
    G = max(n_he, 1)
    obs = Aliquot(aft_age=np.array([40.0]), aft_age_sd=np.array([3.0]), n_spontaneous=np.array([150]),
                  length_counts=np.zeros((1, 200)), length_bins=(np.arange(200) + 0.5) * 0.1, dpar=np.array([2.0]),
                  dpar_sd=np.array([0.1]), he_age=np.full((1, G), 30.0), he_age_sd=np.full((1, G), 2.0),
                  he_radius=np.full((1, G), 60.0), he_eU=np.full((1, G), 40.0), he_mask=np.ones((1, G), bool),
                  aft_mask=np.array([True]))
    return Query(obs, np.full(G, 30.0), np.full(G, 40.0))


def _fake_likelihood(node: int, centre: float, sd: float):
    def f(T, t, query, observation, cfg):
        return -0.5 * ((T[:, node].double() - centre) / sd) ** 2
    return f


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_ess_is_n_for_equal_weights_and_one_for_a_single_weight():
    assert R.ess(np.zeros(100)) == pytest.approx(100.0)
    lw = np.full(100, -np.inf)
    lw[7] = 3.0
    assert R.ess(lw) == pytest.approx(1.0)
    assert R.ess(np.log([1.0, 1.0, 2.0])) == pytest.approx(16.0 / 6.0)


def test_khat_reads_the_tail_index_of_known_weights():
    rng = np.random.default_rng(0)
    # a Pareto tail with shape k = 1 / alpha
    assert R.pareto_khat(np.log(rng.pareto(2.0, 40_000) + 1)) == pytest.approx(0.5, abs=0.15)
    assert R.pareto_khat(np.log(rng.pareto(1.0, 40_000) + 1)) == pytest.approx(1.0, abs=0.2)
    # light-tailed weights are comfortably below the 0.7 line
    assert R.pareto_khat(rng.normal(0.0, 0.3, 4_000)) < 0.3
    assert R.pareto_khat(np.zeros(4_000)) == -np.inf


# ---------------------------------------------------------------------------
# The weights
# ---------------------------------------------------------------------------

def test_a_head_that_is_the_prior_under_a_flat_likelihood_gets_equal_weights(tiny_latent, monkeypatch):
    net = _net(tiny_latent, unconditional=True)
    prior = R.PriorDensity(net)
    monkeypatch.setattr(R, "path_log_likelihood", lambda T, *a, **k: torch.zeros(T.shape[0]))
    # The head's embedding is not the prior's (it reads data), so pin it to the prior's own.
    pinned = _PinnedHead(prior)
    r = R.refine(pinned, prior, _query(), ObservationModel(), R.RefineConfig(n=512, epsilon=0.1),
                 generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(r.log_w, torch.zeros_like(r.log_w), atol=2e-4, rtol=0)
    assert r.ess == pytest.approx(512.0, rel=1e-4)
    assert int(r.from_prior.sum()) == round(0.1 * 512)
    assert r.T.shape == (512, tiny_latent.n_grid) and r.prior_mass == pytest.approx(0.1, abs=1e-3)


class _PinnedHead(torch.nn.Module):
    """A head whose answer is the prior density whatever the data: ``q = p``."""

    def __init__(self, prior: R.PriorDensity):
        super().__init__()
        self.prior = prior
        self.latent = prior.latent

    def forward(self, batch):
        return self.prior._q


def test_refinement_moves_the_head_toward_the_likelihood_and_the_mixture_bounds_the_weights(tiny_latent, monkeypatch):
    prior = R.PriorDensity(_net(tiny_latent, unconditional=True))
    head = _PinnedHead(prior)
    # A likelihood on node 5 far off the prior's centre: the refined mean must move to it.
    T0 = prior.latent.inverse(prior.sample(20_000, torch.Generator().manual_seed(1)), clamp=True)[0][:, 5]
    centre = float(T0.quantile(0.8))
    monkeypatch.setattr(R, "path_log_likelihood", _fake_likelihood(5, centre, 4.0))
    r = R.refine(head, prior, _query(), ObservationModel(), R.RefineConfig(n=20_000, epsilon=0.05),
                 generator=torch.Generator().manual_seed(2))
    s = r.summary()
    assert abs(s["mean"][5] - centre) < 0.5 * abs(float(T0.mean()) - centre)
    # q_mix >= eps p, so w <= L / eps: every log-weight is below -log(eps) + max log L = log 20
    assert float(r.log_w.max()) <= np.log(1 / 0.05) + 1e-4
    idx = r.resample(1_000, torch.Generator().manual_seed(3))
    assert idx.shape == (1_000,) and int(idx.max()) < r.n


def test_a_head_in_other_coordinates_is_refused(tiny_latent, tiny_corpus):
    from chai.models.latent import Latent
    other = Latent.fit(tiny_corpus, split="val")
    with pytest.raises(ValueError, match="different coordinates"):
        R.refine(_net(other), R.PriorDensity(_net(tiny_latent, unconditional=True)), _query(), ObservationModel())
    with pytest.raises(ValueError, match="not trained unconditionally or with empty samples"):
        R.PriorDensity(_net(tiny_latent))


# ---------------------------------------------------------------------------
# The screened corpus
# ---------------------------------------------------------------------------

def _exact(pool: ReferencePool, node: int, centre: float, sd: float) -> np.ndarray:
    lw = -0.5 * ((pool.T[:, node].double() - centre) / sd) ** 2
    w = torch.exp(lw - lw.max())
    return ((w[:, None] * pool.T.double()).sum(0) / w.sum()).numpy()


def test_the_screen_is_the_brute_force_posterior_when_it_reaches_every_row(pool, tiny_latent, monkeypatch):
    monkeypatch.setattr(R, "path_log_likelihood", _fake_likelihood(4, 60.0, 5.0))
    spool = R.ScreenPool(pool, _net(tiny_latent))
    cfg = R.ScreenConfig(block=256, stop_share=0.0, max_rows=N_POOL, tail=0)
    r = R.screened_reference(_net(tiny_latent), spool, _query(), ObservationModel(), cfg)
    assert r["n_ordered"] == N_POOL and r["n_tail"] == 0
    np.testing.assert_allclose(r["mean"], _exact(pool, 4, 60.0, 5.0), rtol=1e-5)


def test_an_untrained_order_is_still_unbiased_through_the_tail(pool, tiny_latent, monkeypatch):
    """A random head orders the pool badly; the uniform tail sample must carry what it missed."""
    monkeypatch.setattr(R, "path_log_likelihood", _fake_likelihood(4, 60.0, 8.0))
    net = _net(tiny_latent, seed=5)
    spool = R.ScreenPool(pool, net)
    exact = _exact(pool, 4, 60.0, 8.0)[4]
    cfg = R.ScreenConfig(block=128, stop_share=0.5, max_rows=512, tail=1_000)
    means, tails = [], []
    for seed in range(30):
        r = R.screened_reference(net, spool, _query(), ObservationModel(), cfg, seed=seed)
        means.append(r["mean"][4])
        tails.append(r["tail_mass"])
    assert r["n_ordered"] <= 512 and r["n_tail"] == 1_000
    assert np.mean(tails) > 0.2, "a bad order leaves real mass in the tail, and says so"
    assert abs(np.mean(means) - exact) < 3 * np.std(means) / np.sqrt(len(means)) + 0.3
