"""The flow head: the spline, the identity start, exact densities, the base, and the dequantiser."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from chai.data.observe import ObservationModel
from chai.eval.calibration import sbc
from chai.eval.deep_past import deep_past_test
from chai.models.encoder import EncoderConfig
from chai.models.flow import FlowConfig, FlowHead, _gamma, rational_quadratic_spline
from chai.models.gaussian import AmortizedPosterior, HeadConfig, draw_z, head_config_from
from chai.train.data import (
    AFT_FLAGS,
    AFT_INPUTS,
    GRAIN_INPUTS,
    CorpusSplit,
    EffortPrior,
    NoisyBatches,
    WallDequantiser,
    fixed_batches,
)
from chai.train.loop import TrainConfig, train

torch.set_default_dtype(torch.float32)

SMALL = FlowConfig(n_layers=4, n_bins=6, d_hidden=32, n_hidden=1)
SMALL_T = FlowConfig(n_layers=4, n_bins=6, d_hidden=32, n_hidden=1, base="student_t", df_init=6.0)
ENC = EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8)


@pytest.fixture(scope="module")
def batches(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "val", latent=tiny_latent, max_rows=1024)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(p_no_aft=0.2, p_no_he=0.2), seed=0)
    return fixed_batches(ds, 512, seed=5), c


def _model(latent, c, head) -> AmortizedPosterior:
    torch.manual_seed(0)
    m = AmortizedPosterior(latent, n_aft=len(AFT_INPUTS), n_bins=c.length_bins.size,
                           n_grain=len(GRAIN_INPUTS), aft_flags=AFT_FLAGS, encoder=ENC, head=head)
    return m


# ---------------------------------------------------------------------------
# The spline
# ---------------------------------------------------------------------------

def test_the_spline_is_the_identity_at_zero_parameters_and_outside_the_bound():
    x = torch.linspace(-8, 8, 1001, dtype=torch.float64)
    K = 8
    z = torch.zeros(1001, K, dtype=torch.float64)
    y, ld = rational_quadratic_spline(x, z, z, z[:, :-1], inverse=False, tail_bound=5.0)
    torch.testing.assert_close(y, x)
    assert ld.abs().max() < 1e-12
    w, h, d = (torch.randn(1001, K, dtype=torch.float64) for _ in range(3))
    y, ld = rational_quadratic_spline(x, w, h, d[:, :-1], inverse=False, tail_bound=5.0)
    out = x.abs() >= 5.0
    torch.testing.assert_close(y[out], x[out])
    assert ld[out].abs().max() == 0.0


def test_the_spline_inverts_and_its_log_derivative_matches_autograd():
    torch.manual_seed(1)
    x = torch.linspace(-4.99, 4.99, 2001, dtype=torch.float64)
    K = 8
    w, h = torch.randn(2001, K, dtype=torch.float64), torch.randn(2001, K, dtype=torch.float64)
    d = 2 * torch.randn(2001, K - 1, dtype=torch.float64)
    y, ld = rational_quadratic_spline(x, w, h, d, inverse=False, tail_bound=5.0)
    xb, ldb = rational_quadratic_spline(y, w, h, d, inverse=True, tail_bound=5.0)
    torch.testing.assert_close(xb, x, atol=1e-9, rtol=0)
    torch.testing.assert_close(ld + ldb, torch.zeros_like(ld), atol=1e-9, rtol=0)
    xg = x.clone().requires_grad_(True)
    yg, ldg = rational_quadratic_spline(xg, w, h, d, inverse=False, tail_bound=5.0)
    g = torch.autograd.grad(yg.sum(), xg)[0]
    assert (g > 0).all(), "monotone"
    torch.testing.assert_close(torch.log(g), ldg, atol=1e-9, rtol=0)


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------

def test_the_flow_starts_as_the_gaussian_head():
    torch.manual_seed(0)
    head = FlowHead(32, 19, SMALL)
    e = torch.randn(64, 32) * 3
    q = head(e)
    z = torch.randn(64, 19)
    # The affine layer starts at N(0, I) and the couplings at the identity.
    torch.testing.assert_close(q.log_prob(z), AmortizedPosterior.prior_nll(z).neg(), atol=0.5, rtol=0)
    # And the flow's density agrees with its own affine layer's Gaussian exactly
    # when the couplings are the identity.
    mvn = head.affine(e)
    torch.testing.assert_close(q.log_prob(z), mvn.log_prob(z), atol=0.05, rtol=0)   # couplings start at 1e-3 of the identity


def test_log_prob_is_a_normalised_density_and_sampling_is_its_inverse():
    """Push random couplings away from the identity and check the density in 2-D."""
    torch.manual_seed(2)
    head = FlowHead(8, 2, FlowConfig(n_layers=4, n_bins=6, d_hidden=16, n_hidden=1, tail_bound=4.0))
    with torch.no_grad():                       # mild: a 0.5 sd here makes spikes a grid cannot resolve
        for layer in head.layers:
            layer.net[-1].weight.normal_(0, 0.15)
            layer.net[-1].bias.normal_(0, 0.15)
        head.affine.net[-1].weight.normal_(0, 0.1)
    e = torch.randn(1, 8)
    q = head(e)
    # Numerical integral over a wide grid.
    g = torch.linspace(-12, 12, 1201)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    z = torch.stack([X.reshape(-1), Y.reshape(-1)], -1).unsqueeze(1)          # (M, 1, 2)
    lp = q.log_prob(z)
    mass = float(torch.exp(lp).sum() * (g[1] - g[0]) ** 2)
    assert abs(mass - 1.0) < 2e-2, mass
    # sample -> standardise is the identity on the base draws.
    u = torch.randn(5000, 1, 2)
    zz = q.push(u)
    ub, _ = q.standardise(zz)
    torch.testing.assert_close(ub, u, atol=1e-3, rtol=0)
    # And the samples' first moments agree with the density's on the grid.
    s = q.sample(20_000)[:, 0]
    p = torch.exp(lp).reshape(-1)
    mean_grid = (z[:, 0] * p[:, None]).sum(0) / p.sum()
    torch.testing.assert_close(s.mean(0), mean_grid, atol=0.1, rtol=0)


def test_the_student_t_base_has_the_documented_density_sampler_and_residual():
    torch.manual_seed(3)
    head = FlowHead(8, 5, SMALL_T)
    assert torch.allclose(head.df, torch.full((5,), 6.0), atol=1e-4)
    u = torch.randn(1000, 5) * 2
    from torch.distributions import StudentT
    ref = StudentT(torch.full((5,), 6.0)).log_prob(u).sum(-1)
    torch.testing.assert_close(head.base_log_prob(u), ref, atol=1e-4, rtol=0)
    gen = torch.Generator().manual_seed(0)
    s = head.base_sample((200_000, 5), gen, like=u)
    # A t_6 has variance 6/4 = 1.5 and kurtosis 3 + 6/(6-4) = 6.
    assert abs(float(s.var()) - 1.5) < 0.05
    assert abs(float((s / s.std()).pow(4).mean()) - 6.0) < 1.0
    # The residual is the normal quantile of the t CDF: N(0, 1) for t draws.
    r = head.base_to_normal(s[:20_000])[:, 0]
    assert kstest(r.numpy(), "norm").pvalue > 1e-3
    # Same generator, same draws.
    g1 = _gamma(torch.tensor(3.0), (7,), torch.Generator().manual_seed(4))
    g2 = _gamma(torch.tensor(3.0), (7,), torch.Generator().manual_seed(4))
    torch.testing.assert_close(g1, g2)
    big = _gamma(torch.tensor(3.0), (100_000,), torch.Generator().manual_seed(5))
    assert abs(float(big.mean()) - 3.0) < 0.03 and abs(float(big.var()) - 3.0) < 0.1


# ---------------------------------------------------------------------------
# In the model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [SMALL, SMALL_T], ids=["gaussian-base", "student-t-base"])
def test_the_model_takes_a_flow_head_in_the_gaussian_slot(batches, tiny_latent, cfg, tmp_path):
    bs, c = batches
    b = bs[0]
    m = _model(tiny_latent, c, cfg)
    m.fit_normalisation(b)
    assert not m.is_gaussian
    nll = m.nll(b)
    assert nll.shape == (512,) and torch.isfinite(nll).all()
    if cfg.base == "gaussian":
        torch.testing.assert_close(nll, m.prior_nll(b["z"]), atol=0.5, rtol=0.05)
    T, _ = m.sample_paths(b, 3)
    assert T.shape == (3, 512, 16) and (T >= 0).all() and (T <= 200).all()
    z = draw_z(m(b), 7)
    assert z.shape == (7, 512, 19)
    r = m.standardised_residual(b)
    assert r.shape == (512, 19)
    if cfg.base == "gaussian":          # untrained, the flow is N(0, I): residual variance 1
        assert abs(float(r.var(0).mean()) - 1.0) < 0.2
    m.save(tmp_path / "flow.pt")
    back = AmortizedPosterior.load(tmp_path / "flow.pt")
    assert isinstance(back.head, FlowHead) and back.head.config == cfg
    with torch.no_grad():
        torch.testing.assert_close(back.nll(b), nll)


def test_head_config_dispatch_and_yaml_round_trip():
    assert isinstance(head_config_from({}), HeadConfig)
    assert head_config_from({"kind": "gaussian", "d_hidden": 8}) == HeadConfig(d_hidden=8)
    f = head_config_from({"kind": "flow", "n_layers": 3, "base": "student_t"})
    assert isinstance(f, FlowConfig) and f.n_layers == 3
    with pytest.raises(ValueError):
        head_config_from({"kind": "nope"})
    cfg = TrainConfig(head=f, dequantise=WallDequantiser(width=4.0))
    back = TrainConfig.from_config(cfg.to_config())
    assert back.head == f and back.dequantise == WallDequantiser(width=4.0)
    assert TrainConfig.from_config({}).dequantise is None


def test_sbc_and_the_deep_past_run_on_a_flow_head(batches, tiny_latent, tiny_corpus):
    bs, c = batches
    m = _model(tiny_latent, c, SMALL)
    m.fit_normalisation(bs[0])
    r = sbc(m, bs, device="cpu", n_draws=40, seed=0, aft_true=c.aft_age[:1024])
    assert r["pit_T"].shape == (1024, 16) and r["pit_z"].shape == (1024, 19)
    assert r["pit_T_exact"].shape == (1024, 16)
    assert ((r["pit_z"] >= 0) & (r["pit_z"] <= 1)).all()
    # Untrained, the flow is N(0, I) in z, so the hyperparameters' PITs are uniform.
    for i in range(3):
        assert kstest(r["pit_z"][:, i], "uniform").pvalue > 1e-3
    train_T = CorpusSplit(tiny_corpus, "train", max_rows=3000).T
    d = deep_past_test(m, bs[:1], train_T, device="cpu", n_draws=8, K=20, seed=0, max_rows=64)
    assert "control" in d and "head_given_truth" not in d


# ---------------------------------------------------------------------------
# The dequantiser
# ---------------------------------------------------------------------------

def test_the_dequantiser_moves_only_wall_nodes_and_clamping_undoes_it():
    rng = np.random.default_rng(0)
    lo, hi = np.zeros(6), np.full(6, 200.0)
    lo[0], hi[0] = 0.0, 25.0
    T = np.array([[12.0, 0.0, 0.0, 30.0, 200.0, 200.0],
                  [3.0, 5.0, 0.0, 0.0, 0.0, 190.0]], dtype=np.float32)
    dq = WallDequantiser(width=10.0, jitter=1.0)
    out = dq.apply(T, lo, hi, rng)
    on_wall = (T <= lo) | (T >= hi)
    np.testing.assert_array_equal(out[~on_wall], T[~on_wall])
    assert (out[T <= lo] < 0).all() and (out[T <= lo] > -11.0).all()
    assert (out[T >= hi] > 200).all() and (out[T >= hi] < 211.0).all()
    # One offset per row and wall: nodes on the same wall in a row differ by at most the jitter.
    row0_lo = out[0, T[0] <= lo]
    assert np.ptp(row0_lo) < 1.0
    np.testing.assert_array_equal(np.clip(out, lo, hi), T)
    # With a noise floor every node moves, by about the floor.
    noisy = WallDequantiser(width=10.0, jitter=1.0, noise_sd=0.5).apply(T, lo, hi, rng)
    d = noisy[~on_wall] - T[~on_wall]
    assert (d != 0).all() and abs(d).max() < 3.0


def test_dequantised_batches_carry_moved_z_and_the_stored_truth(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "val", latent=tiny_latent, max_rows=256)
    plain = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=0)
    deq = NoisyBatches(c, ObservationModel(), EffortPrior(), seed=0, dequantise=WallDequantiser())
    rows = np.arange(256)
    b0 = plain.observe(rows, np.random.default_rng(0))
    b1 = deq.observe(rows, np.random.default_rng(0))
    torch.testing.assert_close(b0["T"], b1["T"])
    lo, hi = c.box
    wall_rows = ((c.T[rows] <= lo) | (c.T[rows] >= hi)).any(1)
    assert wall_rows.any(), "the tiny prior should put some nodes on a wall"
    moved = (b0["z"] - b1["z"]).abs().sum(1) > 1e-4
    np.testing.assert_array_equal(moved.numpy(), wall_rows)
    # The whitening of the stored truth is the stored z.
    torch.testing.assert_close(c.whiten(c.T[rows], rows), b0["z"], atol=1e-4, rtol=0)


def test_a_short_flow_run_beats_the_prior_and_reloads(tiny_corpus, tmp_path):
    from chai.models.latent import Latent
    lat_path = tmp_path / "latent.pt"
    Latent.fit(tiny_corpus).save(lat_path)
    cfg = TrainConfig(corpus=tiny_corpus, latent=str(lat_path), observation=None,
                      effort=EffortPrior(), encoder=EncoderConfig(d_embed=64, d_hidden=64,
                                                                   d_grain=32, d_hist=16, n_trunk=2),
                      head=FlowConfig(n_layers=2, n_bins=4, d_hidden=32, n_hidden=1,
                                      base="student_t", affine_hidden=64),
                      dequantise=WallDequantiser(noise_sd=0.5), batch_size=512, steps=120, warmup=10,
                      lr=3e-3, val_every=60, val_rows=600, num_workers=0, seed=0,
                      n_diag_samples=16, save_every=60, save_from=100)
    _, hist = train(cfg, device="cpu", out_dir=tmp_path / "run", log=lambda *_: None)
    assert sorted(p.name for p in (tmp_path / "run").glob("model_*.pt")) == ["model_120.pt"]
    snap = AmortizedPosterior.load(tmp_path / "run" / "model_120.pt")
    assert snap.meta["best_step"] == 120
    last = hist[-1]
    assert last["val_nll"] < last["val_nll_prior"] - 1.0
    assert "df_min" in last and last["df_min"] >= 2.5
    reloaded = AmortizedPosterior.load(tmp_path / "run" / "model.pt")
    assert isinstance(reloaded.head, FlowHead)
    d = np.load(tmp_path / "run" / "diagnostics.npz")
    assert d["post_sd"].shape == (5, 16)
