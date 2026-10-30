"""The whitened latent: a bijection whose base distribution is the prior's own moments.

Everything here runs on a small synthetic corpus drawn from a cheap copy of
the shipped prior, so the properties pinned are properties of the *map*, not
of one corpus file.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from chai.data.priors import ThermalHistoryPrior
from chai.models.latent import HYPER_NAMES, HyperSpec, Latent, LatentConfig

PRIOR = ThermalHistoryPrior(n_steps=256, n_features=64)
N_GRID = 16
N_FIT = 20_000


@pytest.fixture(scope="module")
def corpus():
    """``(T, hyper, t_grid)`` for a small corpus on a 16-node grid."""
    t_grid = np.linspace(0.0, PRIOR.t_max, N_GRID)
    T, batch = PRIOR.sample_on(N_FIT, np.random.default_rng(0), t_grid)
    return T.astype(np.float32), {k: v.astype(np.float32) for k, v in batch.hyper.items()}, t_grid


@pytest.fixture(scope="module")
def latent(corpus):
    T, hyper, t_grid = corpus
    return Latent.fit_arrays(T, hyper, t_grid=t_grid, prior_config=PRIOR.to_config())


def _held_out(n: int = 5_000, seed: int = 1):
    t_grid = np.linspace(0.0, PRIOR.t_max, N_GRID)
    T, batch = PRIOR.sample_on(n, np.random.default_rng(seed), t_grid)
    return T.astype(np.float32), {k: v.astype(np.float32) for k, v in batch.hyper.items()}


# ---------------------------------------------------------------------------
# Shape is read from the corpus, never assumed
# ---------------------------------------------------------------------------

def test_the_shape_comes_from_the_corpus(latent):
    assert latent.n_grid == N_GRID
    assert latent.n_hyper == 3
    assert latent.dim == N_GRID + 3
    assert latent.hyper_names == HYPER_NAMES
    assert "T_0" not in latent.hyper_names, "T_0 is T[0]; carrying both makes the covariance singular"
    assert latent.node_index(0) == 3


def test_the_box_is_the_priors(latent):
    lo, hi = latent.box_lo.numpy(), latent.box_hi.numpy()
    assert (lo[0], hi[0]) == PRIOR.T_present_range
    assert (lo[1:] == PRIOR.T_min).all() and (hi[1:] == PRIOR.T_max).all()


def test_drift_is_stored_per_ma_and_the_spec_knows_it():
    spec = {s.name: s for s in HyperSpec.from_prior_config(PRIOR.to_config())}
    lo, hi = PRIOR.drift_total_range
    assert (spec["drift"].lo, spec["drift"].hi) == (lo / PRIOR.t_max, hi / PRIOR.t_max)
    assert spec["tau"].log_axis and spec["rate_sd"].log_axis and not spec["drift"].log_axis


def test_wrong_node_count_is_refused(latent):
    T, hyper = _held_out(4)
    with pytest.raises(ValueError, match="nodes"):
        latent(T[:, :-1], hyper)
    with pytest.raises(ValueError, match="coordinates"):
        latent.inverse(torch.zeros(4, latent.dim + 1))


def test_a_present_day_node_that_disagrees_with_T_0_is_refused(latent):
    T, hyper = _held_out(4)
    hyper = dict(hyper)
    hyper["T_0"] = hyper["T_0"] + 1.0
    with pytest.raises(ValueError, match="T_0"):
        latent(T, hyper)


# ---------------------------------------------------------------------------
# It is a bijection with the Jacobian it claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", [
    LatentConfig(),
    LatentConfig(hyper_map="loglogit"),
    LatentConfig(box_logit=True, box_margin=0.5),
    LatentConfig(method="pca"),
])
def test_round_trip_and_jacobian(corpus, config):
    T, hyper, t_grid = corpus
    lat = Latent.fit_arrays(T[:2000], {k: v[:2000] for k, v in hyper.items()}, t_grid=t_grid,
                            prior_config=PRIOR.to_config(), config=config)
    Th, hh = _held_out(8)
    Td = torch.as_tensor(Th, dtype=torch.float64)
    hd = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in hh.items()}
    z, ld = lat.forward_with_logdet(Td, hd)
    Tb, hb, ld_inv = lat.inverse_with_logdet(z)
    assert torch.allclose(Tb, Td, atol=1e-9)
    for k in HYPER_NAMES:
        assert torch.allclose(hb[k], hd[k], rtol=1e-10, atol=1e-10)
    assert torch.allclose(hb["T_0"], Td[:, 0])
    assert torch.allclose(ld + ld_inv, torch.zeros_like(ld), atol=1e-9)

    # The claimed log|det| against autograd, one row.
    names = HYPER_NAMES

    def f(v):
        return lat(v[3:][None], {k: v[i][None] for i, k in enumerate(names)})[0]

    v0 = torch.cat([torch.stack([hd[k][0] for k in names]), Td[0]])
    J = torch.autograd.functional.jacobian(f, v0)
    assert torch.isclose(torch.linalg.slogdet(J).logabsdet, ld[0], atol=1e-8)


def test_float32_in_float32_out_and_float64_inside(latent):
    T, hyper = _held_out(16)
    z = latent(T, hyper)
    assert z.dtype == torch.float32
    Tb, hb = latent.inverse(z)
    assert Tb.dtype == torch.float32
    assert np.abs(Tb.numpy() - T).max() < 1e-3, "a float32 round trip must hold to milli-kelvin"
    assert np.abs(hb["tau"].numpy() - hyper["tau"]).max() < 1e-3


# ---------------------------------------------------------------------------
# The base distribution is the prior's own
# ---------------------------------------------------------------------------

def test_the_fitted_split_is_whitened(latent, corpus):
    T, hyper, _ = corpus
    z = latent(torch.as_tensor(T, dtype=torch.float64),
               {k: torch.as_tensor(v, dtype=torch.float64) for k, v in hyper.items()}).numpy()
    assert np.abs(z.mean(0)).max() < 1e-6
    assert np.abs(np.cov(z.T) - np.eye(latent.dim)).max() < 1e-6


def test_probit_makes_each_hyperparameter_standard_normal_under_the_prior():
    """Exact, not approximate: the probit of a uniform CDF is N(0, 1)."""
    T, hyper = _held_out(20_000, seed=7)
    t_grid = np.linspace(0.0, PRIOR.t_max, N_GRID)
    d = 3 + N_GRID
    proto = Latent(LatentConfig(), HyperSpec.from_prior_config(PRIOR.to_config()), t_grid,
                   np.zeros(N_GRID), np.full(N_GRID, 200.0), np.zeros(d), np.eye(d))
    h = proto._stack_hyper(hyper, T)
    y, _ = proto._to_real_line(torch.as_tensor(T, dtype=torch.float64), h)
    for i, name in enumerate(HYPER_NAMES):
        p = kstest(y[:, i].numpy(), "norm").pvalue
        assert p > 1e-3, f"{name}: probit coordinate is not N(0,1), KS p = {p:.2g}"


def test_loglogit_leaves_a_uniform_and_a_logistic():
    """The review's map, kept for comparison, is *not* Gaussian: pin what it is."""
    T, hyper = _held_out(20_000, seed=7)
    t_grid = np.linspace(0.0, PRIOR.t_max, N_GRID)
    d = 3 + N_GRID
    proto = Latent(LatentConfig(hyper_map="loglogit"), HyperSpec.from_prior_config(PRIOR.to_config()),
                   t_grid, np.zeros(N_GRID), np.full(N_GRID, 200.0), np.zeros(d), np.eye(d))
    y, _ = proto._to_real_line(torch.as_tensor(T, dtype=torch.float64), proto._stack_hyper(hyper, T))
    lo, hi = np.log(PRIOR.tau_range)
    assert kstest(y[:, 0].numpy(), "uniform", args=(lo, hi - lo)).pvalue > 1e-3
    assert kstest(y[:, 2].numpy(), "logistic").pvalue > 1e-3


def test_cholesky_coordinates_are_innovations_ordered_into_the_past(latent):
    """``z_k`` depends on the hyperparameters and on nodes ``<= k`` only.

    This is what lets the Stage 12 deep-past test read "z is standard normal
    past the last node the data see": a change in a deep node cannot reach a
    recent coordinate.
    """
    T, hyper = _held_out(4)
    z0 = latent(T, hyper)
    T1 = T.copy()
    j = N_GRID - 3
    T1[:, j:] += 5.0  # move the deep past only
    z1 = latent(T1, hyper)
    k = latent.node_index(j)
    assert torch.allclose(z1[:, :k], z0[:, :k], atol=1e-5)
    assert not torch.allclose(z1[:, k:], z0[:, k:], atol=1e-3)


def test_pca_is_a_rotation_of_the_same_gaussian(corpus):
    T, hyper, t_grid = corpus
    chol = Latent.fit_arrays(T, hyper, t_grid=t_grid, prior_config=PRIOR.to_config())
    pca = Latent.fit_arrays(T, hyper, t_grid=t_grid, prior_config=PRIOR.to_config(),
                            config=LatentConfig(method="pca"))
    Th, hh = _held_out(64)
    zc, ldc = chol.forward_with_logdet(Th, hh)
    zp, ldp = pca.forward_with_logdet(Th, hh)
    assert torch.allclose((zc**2).sum(1), (zp**2).sum(1), rtol=1e-4)
    assert torch.allclose(ldc, ldp, rtol=1e-5)


def test_the_conditional_is_the_covariances_regression(latent, corpus):
    """On a Gaussian sample the conditional mean is unbiased and the residual
    covariance matches; here it is checked for internal consistency against
    the fitted moments, which is all a Gaussian conditional can promise."""
    T, _hyper, _ = corpus
    given = [latent.node_index(k) for k in range(6)]
    target = [latent.node_index(k) for k in range(6, N_GRID)]
    A, b, S = latent.conditional(target, given)
    x = torch.as_tensor(T, dtype=torch.float64)
    resid = x[:, 6:] - (x[:, :6] @ A.T + b)
    assert torch.abs(resid.mean(0)).max() < 1e-6
    assert torch.allclose(torch.cov(resid.T), S, atol=1e-6 * S.diagonal().max())
    # Conditioning cannot widen anything.
    assert (torch.diagonal(S) <= latent.cov[target, target] + 1e-9).all()


def test_innovation_sd_is_the_cholesky_diagonal_and_shrinks_with_conditioning(latent):
    sd = latent.innovation_sd()
    marginal = torch.sqrt(torch.diagonal(latent.cov))
    assert (sd <= marginal + 1e-9).all()
    assert torch.isclose(sd[0], marginal[0])  # nothing precedes the first coordinate


def test_sample_prior_and_clamp(latent):
    T, h = latent.sample_prior(1000, generator=torch.Generator().manual_seed(0))
    assert T.shape == (1000, N_GRID) and set(h) == {*HYPER_NAMES, "T_0"}
    Tc, _ = latent.sample_prior(1000, generator=torch.Generator().manual_seed(0), clamp=True)
    assert (Tc >= latent.box_lo).all() and (Tc <= latent.box_hi).all()
    # The Gaussian leaks past the walls -- that is the recorded limit, and clamping is the projection.
    assert ((T < latent.box_lo) | (T > latent.box_hi)).any()


# ---------------------------------------------------------------------------
# Persistence and a corpus file
# ---------------------------------------------------------------------------

def test_save_load_round_trip(latent, tmp_path):
    p = tmp_path / "latent.pt"
    latent.save(p)
    back = Latent.load(p)
    T, hyper = _held_out(8)
    assert torch.equal(back(T, hyper), latent(T, hyper))
    assert back.config == latent.config and back.hyper == latent.hyper
    assert back.meta["n_fit"] == N_FIT


def test_fit_reads_a_corpus_file(corpus, tmp_path):
    import json

    import h5py

    T, hyper, t_grid = corpus
    n = T.shape[0]
    split = np.zeros(n, np.uint8)
    split[n // 2:] = 1
    p = tmp_path / "c.h5"
    with h5py.File(p, "w") as f:
        f.attrs["split_names"] = np.array(["train", "val", "test"], dtype=object)
        f.attrs["prior_config"] = json.dumps(PRIOR.to_config())
        f.attrs["git_sha"] = "deadbeef"
        f["T"] = T
        f["split"] = split
        f["t_grid"] = t_grid
        for k, v in hyper.items():
            f[f"hyper/{k}"] = v
    lat = Latent.fit(p, split="val", max_rows=1000)
    assert lat.meta["n_fit"] == 1000 and lat.meta["split"] == "val"
    ref = Latent.fit_arrays(T[n // 2:n // 2 + 1000], {k: v[n // 2:n // 2 + 1000] for k, v in hyper.items()},
                            t_grid=t_grid, prior_config=PRIOR.to_config())
    assert torch.allclose(lat.mean, ref.mean) and torch.allclose(lat.cov, ref.cov)


def test_a_pulse_prior_latent_is_the_path_alone():
    """PulsePrior stores counts among its latents; the latent carries none, and its box
    is the prior's support rather than walls (docs/prior-v2.md)."""
    from chai.data.priors import PulsePrior
    from chai.paths import fixed_grid

    prior = PulsePrior(n_steps=512)
    t = fixed_grid(32, 100.0, spacing="uniform")
    T, b = prior.sample_on(4000, np.random.default_rng(0), t)
    L = Latent.fit_arrays(T, {"T_0": b.hyper["T_0"]}, t_grid=t, prior_config=prior.to_config())
    assert L.n_hyper == 0 and L.dim == 32
    assert float(L.box_hi[5]) == prior.T_max and float(L.box_lo[5]) == prior.T_present_range[0]
    z, ld = L.forward_with_logdet(T[:8], {})
    T2, hyper, ld2 = L.inverse_with_logdet(z)
    np.testing.assert_allclose(T2.numpy(), T[:8], atol=1e-8)
    np.testing.assert_allclose((ld + ld2).numpy(), 0.0, atol=1e-8)
    assert set(hyper) == {"T_0"}


@pytest.fixture(scope="module")
def v2_present():
    """A pulse-prior latent that takes the present day as an input, and its paths."""
    from chai.data.priors import PulsePrior
    from chai.paths import fixed_grid

    prior = PulsePrior(n_steps=512)
    t = fixed_grid(32, 100.0, spacing="uniform")
    T, _ = prior.sample_on(4000, np.random.default_rng(0), t)
    L = Latent.fit_arrays(T, {}, t_grid=t, prior_config=prior.to_config(),
                          config=LatentConfig(present_input=True))
    return L, T


def test_a_present_input_latent_carries_the_excess_over_T_0(v2_present):
    """Node 0 leaves the vector; the rest enter as T_k - T_0, a shift, so the Jacobian is W's alone."""
    L, T = v2_present
    assert L.present_input and L.dim == 31
    z, ld = L.forward_with_logdet(T[:8], {})
    x = T[:8, 1:].astype(np.float64) - T[:8, :1]
    np.testing.assert_allclose(z.numpy(), ((x - L.mean.numpy()) @ L.W.numpy().T), atol=1e-6)
    np.testing.assert_allclose(ld.numpy(), float(L.logdet_W), atol=1e-8)
    T2, hyper, ld2 = L.inverse_with_logdet(z, T0=torch.as_tensor(T[:8, 0]))
    np.testing.assert_allclose(T2.numpy(), T[:8], atol=1e-5)
    np.testing.assert_allclose((ld + ld2).numpy(), 0.0, atol=1e-8)
    np.testing.assert_allclose(hyper["T_0"].numpy(), T[:8, 0])
    with pytest.raises(IndexError):
        L.node_index(0)
    assert L.node_index(1) == 0


def test_a_present_input_latent_needs_T0_and_clamps_to_it(v2_present):
    L, T = v2_present
    with pytest.raises(ValueError, match="pass T0"):
        L.inverse(torch.zeros(3, L.dim))
    # draws in front, one T0 per row behind them; no node is ever colder than today
    T0 = torch.tensor([2.0, 12.0, 24.0], dtype=torch.float64)
    Td, _ = L.inverse(3.0 * torch.randn(50, 3, L.dim, dtype=torch.float64), clamp=True, T0=T0)
    assert Td.shape == (50, 3, 32)
    assert torch.equal(Td[..., 0], T0.expand(50, 3))
    assert bool((Td >= T0[:, None]).all()) and bool((Td <= L.box_hi).all())
    with pytest.raises(ValueError, match="takes no T0"):
        Latent.fit_arrays(T, {}, t_grid=L.t_grid.numpy(), prior_config=L.meta["prior_config"]).inverse(
            torch.zeros(1, 32), T0=5.0)


def test_the_path_conditional_is_in_degrees_whatever_the_coordinates(v2_present, tmp_path):
    """The deep-past reference regresses on the path's own moments, and survives a save."""
    L, T = v2_present
    A, b, S = L.path_conditional([20, 31], [0, 1, 2])
    Tn = T.astype(np.float64)
    X = np.column_stack([Tn[:, :3], np.ones(len(Tn))])
    coef, *_ = np.linalg.lstsq(X, Tn[:, [20, 31]], rcond=None)
    np.testing.assert_allclose(A.numpy(), coef[:3].T, rtol=1e-6, atol=1e-8)
    L.save(tmp_path / "l.pt")
    L2 = Latent.load(tmp_path / "l.pt")
    assert L2.present_input and torch.equal(L2.path_cov, L.path_cov) and L2.dim == L.dim


def test_an_old_latent_takes_its_path_moments_from_the_node_block(latent):
    np.testing.assert_allclose(latent.path_mean.numpy(), latent.mean[latent.n_hyper:].numpy())
    A, b, S = latent.path_conditional([10, 15], [0, 1, 2])
    A2, b2, S2 = latent.conditional([latent.node_index(k) for k in (10, 15)],
                                    [latent.node_index(k) for k in (0, 1, 2)])
    np.testing.assert_allclose(A.numpy(), A2.numpy(), atol=1e-10)
