"""Stage 12: SBC ranks, their summary, and the deep-past conditional test."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from chai.data.observe import ObservationModel
from chai.eval.calibration import (
    FUNCTIONALS,
    LEVELS,
    _pit_from_draws,
    coverage_from_pit,
    functional_values,
    pit_summary,
    sbc,
    strata_of,
)
from chai.eval.deep_past import (
    _TrainPaths,
    boundary_nodes,
    conditional_pits,
    deep_past_test,
    head_conditional_pits,
    oldest_observed_age,
)
from chai.models.gaussian import AmortizedPosterior, sample_mvn
from chai.train.data import CorpusSplit, EffortPrior, NoisyBatches, fixed_batches

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------------------
# PIT values and their summary
# ---------------------------------------------------------------------------

def test_pit_summary_on_uniform_samples_reads_as_calibrated():
    u = np.random.default_rng(0).random((40_000, 3))
    s = pit_summary(u)
    np.testing.assert_allclose(s["coverage"], np.broadcast_to(np.array(LEVELS)[:, None], (5, 3)), atol=0.01)
    assert (s["ecdf_dev"] < 0.012).all()
    assert (np.abs(s["centre_excess"]) < 0.01).all()
    assert (np.abs(s["tail_excess"]) < 0.005).all()
    assert (np.abs(s["bias"]) < 0.005).all()
    assert (s["chi2_p"] > 1e-3).all()
    assert s["histogram"].shape == (20, 3) and s["histogram"].sum(0).tolist() == [40_000] * 3


def test_pit_summary_names_the_direction_of_a_miscalibration():
    rng = np.random.default_rng(1)
    from scipy.stats import norm
    truth = rng.standard_normal(40_000)
    too_narrow = norm.cdf(truth / 0.5)        # a head with half the sd: truth in the tails
    too_wide = norm.cdf(truth / 2.0)          # twice the sd: truth in the centre
    cold = norm.cdf(truth + 0.5)              # head's median below the truth
    s = pit_summary(np.stack([too_narrow, too_wide, cold], 1))
    assert s["tail_excess"][0] > 0.2 and s["centre_excess"][0] < -0.1
    assert s["centre_excess"][1] > 0.1 and s["tail_excess"][1] < 0
    assert s["bias"][2] > 0.1
    c68 = s["coverage"][LEVELS.index(0.68)]
    assert c68[0] < 0.4 and c68[1] > 0.68


def test_coverage_from_pit_is_the_central_interval():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100_000)
    from scipy.stats import norm
    u = norm.cdf(x)
    cov = coverage_from_pit(u, (0.5, 0.95))
    assert abs(cov[0] - 0.5) < 0.01 and abs(cov[1] - 0.95) < 0.005
    # the same thing computed as an interval
    inside = (x > norm.ppf(0.025)) & (x < norm.ppf(0.975))
    assert abs(inside.mean() - cov[1]) < 1e-9


def test_randomised_pit_is_uniform_with_atoms_and_would_not_be_without():
    """Truth and draws from the same mixture with a point mass at the wall."""
    gen = torch.Generator().manual_seed(0)
    n, S = 20_000, 60

    def mixture(*shape):
        x = torch.randn(*shape, generator=gen) * 20 + 10
        return x.clamp_min(0.0)                         # ~30% of the mass exactly at 0

    truth, draws = mixture(n, 1), mixture(S, n, 1)
    u = _pit_from_draws(draws, truth, gen).numpy().ravel()
    assert kstest(u, "uniform").pvalue > 0.01
    # Counting only the strictly-below draws piles every wall truth onto u = 0.
    naive = ((draws < truth).sum(0).float() / S).numpy().ravel()
    assert kstest(naive, "uniform").pvalue < 1e-6
    assert (naive == 0).mean() > 0.2


def test_functionals_have_the_documented_meaning():
    t = np.linspace(0, 100, 64)
    T = torch.zeros(2, 64)
    T[0] = torch.linspace(0, 126, 64)                   # 2 K per node, monotonic
    T[1, 10] = 150.0                                    # one spike
    f = functional_values(T, t)
    assert f.shape == (2, len(FUNCTIONALS))
    k10, k30 = np.argmin(np.abs(t - 10)), np.argmin(np.abs(t - 30))
    assert torch.isclose(f[0, 0], T[0, k30] - T[0, k10])
    dt = float(t[1] - t[0])
    assert torch.isclose(f[0, 4], torch.tensor(2.0 / dt))                  # K/Ma, not K/node
    assert torch.isclose(f[1, 2], torch.tensor(150.0)) and torch.isclose(f[1, 4], torch.tensor(150.0 / dt))


def test_largest_rate_reads_each_segment_at_its_own_spacing():
    """On the 250 Ma member's grid a 4.7 K step past 100 Ma is 1 K/Ma, not 4.7 K/node."""
    from chai.paths import fixed_grid

    t = fixed_grid(96, 250.0, "piecewise:64@100,32@250").numpy()
    T = torch.zeros(1, 96, dtype=torch.float64)
    T[0, 64:] = torch.arange(1, 33, dtype=torch.float64) * 150.0 / 32      # 1 K/Ma past 100 Ma
    assert torch.isclose(functional_values(T, t)[0, 4], torch.tensor(1.0, dtype=torch.float64))


def test_the_horizon_stratum_follows_the_grid_and_late_exhumation_needs_both_ages():
    t = np.concatenate([np.linspace(0, 100, 64), 100 + np.arange(1, 33) * 150 / 32])
    batch = {"T": np.full((4, 96), 50.0)}
    box = (np.zeros(96), np.full(96, 200.0))
    aft = np.array([99.0, 243.0, 150.0, 150.0])
    s = strata_of(batch, t, box, aft_true=aft, he_true=np.array([10.0, 10.0, 30.0, 80.0]))
    assert s["near_horizon"].tolist() == [False, True, False, False]            # 97% of 250 Ma
    assert s["late_exhumation"].tolist() == [False, True, True, False]
    assert "late_exhumation" not in strata_of(batch, t, box, aft_true=aft)


def test_strata_are_named_masks_over_the_rows():
    t = np.linspace(0, 100, 64)
    T = torch.full((5, 64), 50.0)
    T[0, 3] = 0.0                                       # on the floor inside the window
    T[1, 60] = 200.0                                    # on the ceiling, but past 67 Ma
    batch = {"T": T, "effort": {"aft_present": torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool),
                                "n_lengths": torch.tensor([50, 0, 0, 0, 30]),
                                "n_he_grains": torch.tensor([3, 3, 0, 4, 0])}}
    box = (np.zeros(64), np.full(64, 200.0))
    s = strata_of(batch, t, box, aft_true=np.array([10.0, 99.0, 50.0, 98.0, 5.0]),
                  p_fast=np.array([0.0, 0.2, 0.5, 0.9, 0.09]))
    assert s["full"].tolist() == [True, False, False, False, False]
    assert s["no_lengths"].tolist() == [False, True, False, False, False]
    assert s["aft_only"].tolist() == [False, False, True, False, True]
    assert s["he_only"].tolist() == [False, False, False, True, False]
    assert s["touches_wall"].tolist() == [True, False, False, False, False]
    assert s["near_horizon"].tolist() == [False, True, False, True, False]
    assert s["fast_true"].tolist() == [False, True, False, False, False]  # 31 and 94 K/Ma steepest segments
    assert s["fast"].tolist() == [False, True, True, True, False]        # from the head's draws, not the truth
    assert "fast" not in strata_of(batch, t, box)
    assert s["all"].all()


# ---------------------------------------------------------------------------
# The deep past
# ---------------------------------------------------------------------------

def test_oldest_age_and_boundary_follow_the_observables():
    aft = torch.zeros(4, 9)
    aft[0, 0], aft[0, 3] = np.log(30.0), 1.0            # AFT age 30 Ma
    aft[1, 0], aft[1, 3] = np.log(80.0), 1.0            # 80 Ma
    grains = torch.zeros(4, 3, 4)
    mask = torch.zeros(4, 3, dtype=torch.bool)
    grains[0, 0, 0], mask[0, 0] = np.log(45.0), True    # a He grain older than the AFT age
    grains[2, 1, 0], mask[2, 1] = np.log(12.0), True    # He only
    old = oldest_observed_age({"aft": aft, "grains": grains, "grain_mask": mask})
    np.testing.assert_allclose(old[:3], [45.0, 80.0, 12.0])
    assert np.isnan(old[3])
    t = np.linspace(0, 100, 64)
    b = boundary_nodes(old, t)
    assert t[b[0] - 1] <= 45.0 < t[b[0]] and t[b[1] - 1] <= 80.0 < t[b[1]]
    assert b[3] == 64, "no observable, no deep past to test"
    b2 = boundary_nodes(old, t, margin=1.5)
    assert (b2[:3] > b[:3]).all() and b2[1] == 64


class _GaussianLatent:
    """A stand-in with the methods :func:`conditional_pits` uses, for a
    prior that really is Gaussian, so the regression reference is exact."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.cov = torch.as_tensor(cov, dtype=torch.float64)
        self.mean = torch.as_tensor(mean, dtype=torch.float64)
        # a piecewise grid, 2 Ma to the sixth node then 5 Ma, so offsets in Ma are not j * dt
        n = len(mean)
        self.t_grid = torch.as_tensor(np.concatenate([2.0 * np.arange(6), 10.0 + 5.0 * np.arange(1, n - 5)]))

    def node_index(self, k: int) -> int:
        return k

    def conditional(self, target, given):
        t, g = torch.as_tensor(target), torch.as_tensor(given)
        S_tg, S_gg = self.cov[t][:, g], self.cov[g][:, g]
        A = torch.linalg.solve(S_gg, S_tg.T).T
        return A, self.mean[t] - A @ self.mean[g], self.cov[t][:, t] - A @ S_tg.T

    path_conditional = conditional      # the path is the whole vector here


def _ar1_paths(n: int, n_grid: int, rng: np.random.Generator, rho: float = 0.9) -> np.ndarray:
    x = np.empty((n, n_grid))
    x[:, 0] = rng.standard_normal(n)
    for k in range(1, n_grid):
        x[:, k] = rho * x[:, k - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return (50.0 + 20.0 * x).astype(np.float32)


def test_conditional_pits_are_uniform_when_the_paths_come_from_the_prior():
    """The control on a Gaussian AR(1) prior: the regression PIT is exactly
    uniform, the kNN PIT uniform up to the neighbours' slack, and both see a
    head whose deep past is too narrow."""
    rng = np.random.default_rng(0)
    n_grid, N, m = 12, 200_000, 4000
    train = _ar1_paths(N, n_grid, rng)
    lat = _GaussianLatent(train.mean(0), np.cov(train.astype(np.float64).T))
    paths = _TrainPaths(train, "cpu")
    gen = torch.Generator().manual_seed(0)
    truth = torch.as_tensor(_ar1_paths(m, n_grid, rng))[:, None, :]
    bounds = np.full(m, 4)
    r = conditional_pits(truth, bounds, paths, lat, K=50, gen=gen)
    assert r["offsets"].tolist() == list(range(n_grid - 4))
    t = lat.t_grid.numpy()
    assert np.allclose(r["offset_ma"], t[4:] - t[4])                   # one boundary: exact
    for j in (0, 3, 7):
        u_reg, u_knn = r["pit_reg"][j], r["pit_knn"][j]
        assert u_reg.size == m
        # exact up to the covariance being estimated from 2e5 paths (checked by hand in float64)
        assert kstest(u_reg, "uniform").statistic < 0.04, "Gaussian prior: the regression is exact"
        assert abs(u_reg.mean() - 0.5) < 0.02
        cov_knn = coverage_from_pit(u_knn, (0.68, 0.95))
        assert abs(cov_knn[0] - 0.68) < 0.06 and abs(cov_knn[1] - 0.95) < 0.03
    # Neighbours far past the boundary spread like the prior's conditional.
    assert abs(r["sd_knn"][7] / r["sd_reg"][7] - 1.0) < 0.1
    # A "head" that halves the conditional spread is caught by both references.
    A, b, S = lat.conditional(list(range(4, n_grid)), list(range(4)))
    rec = truth[:, 0, :4].double()
    mu = rec @ A.T + b
    L = torch.linalg.cholesky(S)
    narrow = (mu[:, None, :] + 0.5 * torch.randn(m, 8, n_grid - 4, dtype=torch.float64,
                                                 generator=gen) @ L.T).float()
    T_head = torch.cat([truth.expand(m, 8, n_grid)[:, :, :4], narrow], dim=2)
    h = conditional_pits(T_head, bounds, paths, lat, K=50, gen=gen)
    for j in (1, 5):
        assert pit_summary(h["pit_reg"][j])["tail_excess"][0] < -0.03
        assert pit_summary(h["pit_knn"][j])["centre_excess"][0] > 0.15
        assert h["sd_net"][j] < 0.6 * h["sd_knn"][j]


# ---------------------------------------------------------------------------
# End to end on the tiny corpus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model(tiny_corpus, tiny_latent):
    from chai.models.encoder import EncoderConfig
    from chai.models.gaussian import HeadConfig
    from chai.train.data import AFT_FLAGS, AFT_INPUTS, GRAIN_INPUTS
    c = CorpusSplit(tiny_corpus, "val", latent=tiny_latent, max_rows=1024)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(p_no_aft=0.2, p_no_he=0.2), seed=0)
    batches = fixed_batches(ds, 512, seed=5)
    torch.manual_seed(0)
    m = AmortizedPosterior(tiny_latent, n_aft=len(AFT_INPUTS), n_bins=c.length_bins.size,
                           n_grain=len(GRAIN_INPUTS), aft_flags=AFT_FLAGS,
                           encoder=EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8),
                           head=HeadConfig(d_hidden=32))
    m.fit_normalisation(batches[0])
    return m, batches, c


def test_sample_mvn_matches_the_distribution():
    torch.manual_seed(0)
    from torch.distributions import MultivariateNormal
    L = torch.tril(torch.randn(3, 5, 5)) * 0.3 + 2 * torch.eye(5)
    q = MultivariateNormal(torch.randn(3, 5), scale_tril=L)
    z = sample_mvn(q, 50_000)
    assert z.shape == (50_000, 3, 5)
    torch.testing.assert_close(z.mean(0), q.loc, atol=0.05, rtol=0)
    emp = torch.einsum("sni,snj->nij", z - q.loc, z - q.loc) / 50_000
    torch.testing.assert_close(emp, q.covariance_matrix, atol=0.15, rtol=0.05)


def test_sbc_returns_pits_of_the_right_shape_and_the_untrained_head_is_the_prior_in_z(tiny_model):
    m, batches, c = tiny_model
    r = sbc(m, batches, device="cpu", n_draws=40, seed=0, aft_true=c.aft_age[:1024])
    assert r["pit_T"].shape == (1024, 16) and r["pit_z"].shape == (1024, 19)
    assert r["pit_T_exact"].shape == (1024, 16) and r["pit_fn"].shape == (1024, len(FUNCTIONALS))
    assert set(r["strata"]) >= {"full", "aft_only", "he_only", "touches_wall", "near_horizon", "all"}
    assert ((r["pit_T"] >= 0) & (r["pit_T"] <= 1)).all()
    # An untrained head is N(0, I) in z, which is the prior's second moments:
    # the hyperparameters are exactly N(0, 1) under the prior, so their PITs are uniform.
    for i in range(3):
        assert kstest(r["pit_z"][:, i], "uniform").pvalue > 1e-3
    # Where no draw touched a wall the clamped and the exact PIT agree to Monte Carlo noise.
    s_clamped, s_exact = pit_summary(r["pit_T"]), pit_summary(r["pit_T_exact"])
    assert np.abs(s_clamped["coverage"][1] - s_exact["coverage"][1]).max() < 0.25


def test_deep_past_test_runs_end_to_end_with_its_control(tiny_model, tiny_corpus):
    m, batches, _ = tiny_model
    train = CorpusSplit(tiny_corpus, "train")
    r = deep_past_test(m, batches, train.T, device="cpu", n_draws=8, K=30, max_rows=600)
    assert r["boundary"].shape == (600,) and r["usable"].sum() > 100
    assert len(r["pit_knn"]) == len(r["offsets"]) == len(r["control"]["pit_knn"])
    assert r["n_at_offset"][0] == r["usable"].sum() * 8
    assert r["control"]["n_at_offset"][0] == r["usable"].sum()
    h = r["head_given_truth"]
    assert len(h["pit_head"]) == len(r["offsets"]) and h["pit_head"][0].size == r["usable"].sum()
    assert np.isfinite(h["sd_head"]).all() and (h["sd_head"] > 0).all()
    fixed = deep_past_test(m, batches, train.T, device="cpu", n_draws=4, K=30, max_rows=300,
                           fixed_boundary=8, control=False)
    assert (fixed["boundary"] == 8).all() and len(fixed["offsets"]) == 8 and "control" not in fixed


def test_head_conditional_matches_a_hand_computation(tiny_model):
    m, batches, _ = tiny_model
    b = batches[0]
    r = head_conditional_pits(m, [b], np.full(512, 6), device="cpu", max_rows=64)
    with torch.no_grad():
        q = m({k: v for k, v in b.items() if torch.is_tensor(v)})
        lat = m.latent
        ix = [lat.node_index(k) for k in range(16)]
        W = lat.W_inv[ix].double()
        mean = lat.mean[ix] + q.loc[0].double() @ W.T
        C = W @ q.covariance_matrix[0].double() @ W.T
        G = C[6:, :6] @ torch.linalg.inv(C[:6, :6])
        T = b["T"][0].double()
        mu = mean[6:] + G @ (T[:6] - mean[:6])
        sd = torch.sqrt(torch.diagonal(C[6:, 6:] - G @ C[:6, 6:]))
        u = torch.special.ndtr((T[6:] - mu) / sd).numpy()
    for j in range(10):
        assert abs(r["pit_head"][j][0] - u[j]) < 1e-4


# ---------------------------------------------------------------------------
# A head given the present day (Stage 19c): scored as the user gives it, a range
# ---------------------------------------------------------------------------

def test_a_present_range_holds_the_truth_uniformly_even_where_it_is_cut():
    from chai.eval.calibration import present_ranges
    gen = torch.Generator().manual_seed(0)
    T0 = 25.0 * torch.rand(200_000, generator=gen, dtype=torch.float64)      # the prior's U(0, 25)
    r = present_ranges(T0, 5.0, 0.0, 25.0, gen)
    assert (r[:, 0] >= 0).all() and (r[:, 1] <= 25).all() and ((r[:, 1] - r[:, 0]) <= 5.0 + 1e-12).all()
    assert ((T0 >= r[:, 0]) & (T0 <= r[:, 1])).all()
    # Given its range, the truth is uniform on it -- the cut ones at the edges included.
    u = ((T0 - r[:, 0]) / (r[:, 1] - r[:, 0])).numpy()
    cut = ((r[:, 1] - r[:, 0]) < 5.0 - 1e-9).numpy()
    assert 0.1 < cut.mean() < 0.3
    for sel in (np.ones_like(cut), cut):
        assert kstest(u[sel], "uniform").pvalue > 1e-3


@pytest.fixture(scope="module")
def present_model(tiny_corpus):
    from chai.models.encoder import EncoderConfig
    from chai.models.flow import FlowConfig
    from chai.models.latent import Latent, LatentConfig
    from chai.train.data import AFT_FLAGS, AFT_INPUTS, GRAIN_INPUTS
    latent = Latent.fit(tiny_corpus, config=LatentConfig(present_input=True))
    c = CorpusSplit(tiny_corpus, "val", latent=latent, max_rows=512)
    batches = fixed_batches(NoisyBatches(c, ObservationModel(), EffortPrior(), seed=0), 256, seed=5)
    torch.manual_seed(0)
    m = AmortizedPosterior(latent, n_aft=len(AFT_INPUTS), n_bins=c.length_bins.size,
                           n_grain=len(GRAIN_INPUTS), aft_flags=AFT_FLAGS,
                           encoder=EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8),
                           head=FlowConfig(n_layers=2, n_bins=4, d_hidden=16, n_hidden=1))
    m.fit_normalisation(batches[0])
    return m, batches, c


def test_draws_given_a_present_range_each_take_their_own_present_day(present_model):
    from chai.eval.calibration import draw_given_present, present_ranges
    m, batches, _ = present_model
    b = batches[0]
    gen = torch.Generator().manual_seed(1)
    ranges = present_ranges(b["T0"], 5.0, 0.0, 25.0, gen)
    z, T0 = draw_given_present(m, b, 7, ranges, gen, max_rows=600)       # 2 draws per pass: four passes
    assert z.shape == (7, 256, m.dim) and T0.shape == (7, 256)
    assert ((T0 >= ranges[:, 0]) & (T0 <= ranges[:, 1])).all()
    assert T0.std(0).min() > 0                                          # not one T_0 per row
    T, _ = m.latent.inverse(z, clamp=True, T0=T0)
    assert torch.equal(T[..., 0], T0.to(T.dtype)) and bool((T >= T0[..., None].to(T.dtype) - 1e-5).all())


def test_sbc_of_a_head_given_the_present_day_scores_node_0_only_on_a_range(present_model):
    m, batches, c = present_model
    exact = sbc(m, batches, device="cpu", n_draws=20, seed=0)
    ranged = sbc(m, batches, device="cpu", n_draws=20, seed=0, present_width=5.0)
    n_grid = m.latent.n_grid
    # Given T_0 exactly, node 0 is the input: nodes 1 on are scored.
    assert exact["first_node"] == 1 and exact["pit_T"].shape == (512, n_grid - 1)
    assert exact["t_grid"].size == n_grid - 1 and "present_range" not in exact
    # On a range, every node is a draw; z (the excess over the true T_0) keeps its shape.
    assert ranged["first_node"] == 0 and ranged["pit_T"].shape == (512, n_grid)
    assert ranged["pit_z"].shape == exact["pit_z"].shape == (512, m.dim)
    np.testing.assert_array_equal(ranged["nll"], exact["nll"])
    r, T0 = ranged["present_range"], c.T[:512, 0]
    assert ((T0 >= r[:, 0] - 1e-5) & (T0 <= r[:, 1] + 1e-5)).all()
    # Node 0's draws are uniform on the range the truth is uniform on: calibrated by construction.
    assert kstest(ranged["pit_T"][:, 0], "uniform").pvalue > 1e-3
    with pytest.raises(ValueError, match="positive"):
        sbc(m, batches[:1], device="cpu", n_draws=4, present_width=-1.0)


def test_present_width_is_refused_by_a_head_that_draws_the_present_day(tiny_model):
    m, batches, _ = tiny_model
    with pytest.raises(ValueError, match="draws it"):
        sbc(m, batches[:1], device="cpu", n_draws=4, present_width=5.0)
