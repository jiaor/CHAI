"""The encoder and the Gaussian head: invariances, the N(0, I) start, and that it learns."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chai.data.observe import ObservationModel
from chai.models.encoder import AliquotEncoder, EncoderConfig, Standardize
from chai.models.gaussian import AmortizedPosterior, GaussianHead, HeadConfig, remove_floor
from chai.train.data import (
    AFT_FLAGS,
    AFT_INPUTS,
    GRAIN_INPUTS,
    CorpusSplit,
    EffortPrior,
    NoisyBatches,
    features_from_aliquot,
)
from chai.train.loop import TrainConfig, train

torch.set_default_dtype(torch.float32)


@pytest.fixture(scope="module")
def batch(tiny_corpus, tiny_latent):
    c = CorpusSplit(tiny_corpus, "val", latent=tiny_latent, max_rows=512)
    ds = NoisyBatches(c, ObservationModel(), EffortPrior(p_no_aft=0.2, p_no_he=0.2), seed=0)
    return ds.observe(np.arange(512), np.random.default_rng(0)), c


def _model(latent, c, **kw) -> AmortizedPosterior:
    return AmortizedPosterior(latent, n_aft=len(AFT_INPUTS), n_bins=c.length_bins.size,
                              n_grain=len(GRAIN_INPUTS), aft_flags=AFT_FLAGS, **kw)


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------

def test_the_head_starts_at_the_standard_normal():
    torch.manual_seed(0)
    head = GaussianHead(32, 19)
    e = torch.randn(64, 32) * 3
    q = head(e)
    assert torch.allclose(q.loc, torch.zeros_like(q.loc), atol=0.05)
    eye = torch.eye(19).expand(64, 19, 19)
    assert torch.allclose(q.scale_tril, eye, atol=0.05)
    z = torch.randn(64, 19)
    assert torch.allclose(-q.log_prob(z), AmortizedPosterior.prior_nll(z), atol=0.5)


def test_the_cholesky_factor_is_lower_triangular_with_a_floored_diagonal():
    head = GaussianHead(8, 5, HeadConfig(min_scale=0.1))
    raw = torch.randn(3, head.n_tril) * 10 - 20          # would drive softplus to ~0
    L = head.scale_tril(raw)
    assert torch.equal(L, L.tril())
    assert (torch.diagonal(L, dim1=-2, dim2=-1) >= 0.1).all()


def test_log_prob_matches_the_hand_computed_gaussian():
    torch.manual_seed(1)
    head = GaussianHead(8, 6)
    with torch.no_grad():
        head.net[-1].weight.mul_(50.0)                  # make it non-trivial
    q = head(torch.randn(4, 8))
    z = torch.randn(4, 6)
    r = torch.linalg.solve_triangular(q.scale_tril, (z - q.loc).unsqueeze(-1), upper=False).squeeze(-1)
    logdet = torch.log(torch.diagonal(q.scale_tril, dim1=-2, dim2=-1)).sum(-1)
    expect = -0.5 * r.pow(2).sum(-1) - logdet - 3 * np.log(2 * np.pi)
    assert torch.allclose(q.log_prob(z), expect, atol=1e-4)


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------

def test_standardize_skips_flags_and_respects_the_row_mask():
    s = Standardize(3, skip=(2,))
    x = torch.tensor([[1.0, 10.0, 1.0], [3.0, 30.0, 0.0], [100.0, 0.0, 1.0]])
    s.fit(x, mask=torch.tensor([True, True, False]))
    y = s(x[:2])
    assert torch.allclose(y[:, :2].mean(0), torch.zeros(2), atol=1e-6)
    assert torch.equal(y[:, 2], x[:2, 2]), "flags pass through untouched"


def test_the_encoder_is_permutation_invariant_and_ignores_padding(batch):
    b, c = batch
    torch.manual_seed(0)
    enc = AliquotEncoder(len(AFT_INPUTS), c.length_bins.size, len(GRAIN_INPUTS),
                         aft_flags=AFT_FLAGS, config=EncoderConfig(d_embed=32, d_hidden=32,
                                                                   d_grain=16, d_hist=8))
    enc.fit_normalisation(b)
    enc.eval()
    with torch.no_grad():
        e0 = enc(b)
        # Shuffle the grain axis independently per row.
        order = torch.argsort(torch.rand(b["grain_mask"].shape), dim=1)
        shuffled = dict(b)
        shuffled["grains"] = torch.gather(b["grains"], 1, order.unsqueeze(-1).expand_as(b["grains"]))
        shuffled["grain_mask"] = torch.gather(b["grain_mask"], 1, order)
        e1 = enc(shuffled)
        # Pad three more (absent) grains onto every row.
        padded = dict(b)
        padded["grains"] = torch.cat([b["grains"], torch.zeros(512, 3, 4)], dim=1)
        padded["grain_mask"] = torch.cat([b["grain_mask"], torch.zeros(512, 3, dtype=torch.bool)], 1)
        e2 = enc(padded)
    assert torch.allclose(e0, e1, atol=1e-5)
    assert torch.allclose(e0, e2, atol=1e-5)
    assert torch.isfinite(e0).all()


def test_features_agree_with_aliquot_permute_grains(tiny_corpus):
    """The container's own permutation and the encoder's invariance are the same statement."""
    c = CorpusSplit(tiny_corpus, "val", max_rows=64)
    obs = ObservationModel().observe(c.truth(np.arange(64)), np.random.default_rng(0),
                                     effort={"n_he_grains": np.arange(64) % 6})
    f0 = features_from_aliquot(obs)
    f1 = features_from_aliquot(obs.permute_grains(np.random.default_rng(1)))
    np.testing.assert_array_equal(f0["aft"], f1["aft"])
    # Same multiset of grains per row.
    for i in range(64):
        a = f0["grains"][i][f0["grain_mask"][i]]
        b = f1["grains"][i][f1["grain_mask"][i]]
        np.testing.assert_allclose(np.sort(a, axis=0), np.sort(b, axis=0))


# ---------------------------------------------------------------------------
# The whole model
# ---------------------------------------------------------------------------

def test_shapes_come_from_the_latent_and_the_data(batch, tiny_latent):
    b, c = batch
    m = _model(tiny_latent, c)
    m.fit_normalisation(b)
    assert m.dim == tiny_latent.dim == 19
    nll = m.nll(b)
    assert nll.shape == (512,) and torch.isfinite(nll).all()
    assert torch.allclose(nll, m.prior_nll(b["z"]), rtol=0.05, atol=0.5), "untrained = prior baseline"
    T, hyper = m.sample_paths(b, 3)
    assert T.shape == (3, 512, 16)
    assert (T >= 0).all() and (T <= 200).all(), "clamped onto the box"
    assert set(hyper) == {"tau", "rate_sd", "drift", "T_0"}
    r = m.standardised_residual(b)
    assert r.shape == (512, 19)
    assert abs(float(r.var(0).mean()) - 1.0) < 0.15


def test_save_and_load_give_the_same_distribution(batch, tiny_latent, tmp_path):
    b, c = batch
    torch.manual_seed(0)
    m = _model(tiny_latent, c, encoder=EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8),
               head=HeadConfig(d_hidden=32), meta={"note": "test"})
    m.fit_normalisation(b)
    m.save(tmp_path / "m.pt")
    back = AmortizedPosterior.load(tmp_path / "m.pt")
    assert back.meta["note"] == "test" and back.dim == 19
    with torch.no_grad():
        torch.testing.assert_close(back.nll(b), m.nll(b))
    torch.testing.assert_close(back.latent.cov, m.latent.cov)


def test_a_short_run_beats_the_prior_baseline_and_writes_its_files(tiny_corpus, tmp_path):
    lat_path = tmp_path / "latent.pt"
    from chai.models.latent import Latent
    Latent.fit(tiny_corpus).save(lat_path)
    cfg = TrainConfig(corpus=tiny_corpus, latent=str(lat_path), observation=None,
                      effort=EffortPrior(), encoder=EncoderConfig(d_embed=64, d_hidden=64,
                                                                   d_grain=32, d_hist=16, n_trunk=2),
                      head=HeadConfig(d_hidden=64), batch_size=512, steps=120, warmup=10,
                      lr=3e-3, val_every=60, val_rows=600, num_workers=0, seed=0,
                      n_diag_samples=16)
    _, hist = train(cfg, device="cpu", out_dir=tmp_path / "run", log=lambda *_: None)
    last = hist[-1]
    assert last["val_nll"] < last["val_nll_prior"] - 1.0, "the data bought nothing"
    assert hist[0]["val_nll"] > hist[-1]["val_nll"]
    for name in ("config.yaml", "history.jsonl", "model.pt", "diagnostics.npz"):
        assert (tmp_path / "run" / name).exists(), name
    d = np.load(tmp_path / "run" / "diagnostics.npz")
    assert d["post_sd"].shape == (5, 16) and d["resid_var"].shape == (5, 19)
    assert d["n"][-1] == 600
    # The posterior is narrower than the prior where the made-up observables look.
    assert (d["post_sd"][-1, :8] < d["prior_sd"][:8]).all()
    reloaded = AmortizedPosterior.load(tmp_path / "run" / "model.pt")
    assert reloaded.meta["best_step"] in {60, 120}


def test_remove_floor_takes_a_known_floor_back_out_of_smooth_draws():
    """Stage 19e: a smooth 3-event posterior, drawn with a 1 K white floor on every node but T_0."""
    g = torch.Generator().manual_seed(0)
    S, rows, n = 2000, 6, 95
    t = torch.linspace(0, 1, n, dtype=torch.float64)
    basis = torch.stack([t, torch.sin(torch.pi * t), torch.sin(2 * torch.pi * t)])     # (3, n)
    scale = torch.tensor([30.0, 8.0, 3.0], dtype=torch.float64)
    T = 20 + (torch.randn(S, rows, 3, generator=g, dtype=torch.float64) * scale) @ basis
    T[..., 0] = 15.0
    y = T.clone()
    y[..., 1:] += torch.randn(S, rows, n - 1, generator=g, dtype=torch.float64)
    out = remove_floor(y, 1.0, first=1)
    assert torch.equal(out[..., 0], y[..., 0])
    # Per-node sd: the floor makes it 2.4x too wide where the posterior is 0.5 K wide.  Taken out to within 3%
    # where the posterior is at least the floor's width (an sd from 2000 draws has 1.6% sampling error of its
    # own), and within 10% below it, where the draws' eigenvectors are no longer resolved from the floor's.
    sd_true, sd_y, sd_out = T[..., 1:].std(0), y[..., 1:].std(0), out[..., 1:].std(0)
    assert (sd_y / sd_true).max() > 2
    err = (sd_out / sd_true - 1).abs()
    assert (err[sd_true >= 1] < 0.03).all() and (err < 0.10).all()
    # Node-to-node steps: the floor's sqrt(2) K per step is gone (1.47 K with it, 0.40 K without).
    step = lambda x: x[..., 2:] - x[..., 1:-1]
    assert step(y).std(0).mean() > 1.3
    gap = (step(out).std(0) - step(T).std(0)).abs()
    assert gap.mean() < 0.04 and gap.max() < 0.15                # measured: 0.025 and 0.105 K
    # The mean is left where the draws put it.
    assert torch.allclose(out.mean(0), y.mean(0))
    with pytest.raises(ValueError, match="draws cannot resolve"):
        remove_floor(y[:100], 1.0, first=1)
