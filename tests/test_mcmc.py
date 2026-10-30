"""Stage 8 MCMC supplement: the prior as a map of standard normals, and a sampler exact for it.

The two claims the supplement rests on, each tested where it can be checked
exactly: that :class:`GenerativeMap` *is* the corpus prior (it reproduces
:meth:`ThermalHistoryPrior.sample_on` draw for draw from the latents the numpy
prior drew), and that the batched elliptical slice sampler and the tempered
SMC around it target ``N(0, I) L`` (on a conjugate Gaussian, where the
posterior and the evidence are closed-form).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from chai.data.priors import ThermalHistoryPrior
from chai.eval.mcmc import (
    GenerativeMap,
    SMCConfig,
    elliptical_slice,
    latent_of_draws,
    smc_posterior,
)

PRIOR = ThermalHistoryPrior(n_steps=256, n_features=64)
T_GRID = np.linspace(0.0, PRIOR.t_max, 16)


def _numpy_draws(n: int, seed: int):
    """What ``sample_on`` returns, and the latents it drew on the way."""
    T, batch = PRIOR.sample_on(n, np.random.default_rng(seed), T_GRID)
    hyper, omega, phase = PRIOR.draw_latents(n, np.random.default_rng(seed))
    return T, batch, hyper, omega, phase


def test_draw_latents_is_what_sample_draws():
    _, batch, hyper, _, _ = _numpy_draws(50, 3)
    for k in PRIOR.hyper_names:
        np.testing.assert_array_equal(batch.hyper[k], hyper[k])


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-6), (torch.float32, 2e-2)])
def test_generative_map_reproduces_the_numpy_prior(dtype, tol):
    T, _, hyper, omega, phase = _numpy_draws(200, 11)
    xi = latent_of_draws(PRIOR, hyper, omega, phase, np.random.default_rng(0))
    g = GenerativeMap(PRIOR, T_GRID, dtype=dtype, chunk=100)
    T_map, h = g(torch.as_tensor(xi, dtype=dtype))
    assert float(np.abs(T_map.double().numpy() - T).max()) < tol
    for k in PRIOR.hyper_names:
        np.testing.assert_allclose(h[k].double().numpy(), hyper[k], rtol=1e-5 if dtype == torch.float64 else 1e-4)


def test_latents_of_prior_draws_are_standard_normal():
    _, _, hyper, omega, phase = _numpy_draws(4000, 5)
    xi = latent_of_draws(PRIOR, hyper, omega, phase, np.random.default_rng(1))
    assert xi.shape == (4000, 4 + 3 * PRIOR.n_features)
    assert abs(xi.mean()) < 0.01 and abs(xi.std() - 1.0) < 0.01
    # Each block separately: hyperparameters, frequencies, and both phase coordinates.
    for sl in (slice(0, 4), slice(4, 68), slice(68, 132), slice(132, 196)):
        assert abs(xi[:, sl].std() - 1.0) < 0.02


def test_generative_map_starts_at_the_present_day():
    # Not a wall test: the Hermite read-off can overshoot a wall between RK4
    # steps, and the corpus's own ``HistoryBatch.at`` does exactly the same.
    g = GenerativeMap(PRIOR, T_GRID)
    T, h = g(torch.randn(500, g.dim, generator=torch.Generator().manual_seed(0)))
    torch.testing.assert_close(T[:, 0], h["T_0"])
    lo, hi = PRIOR.T_present_range
    assert float(h["T_0"].min()) >= lo and float(h["T_0"].max()) <= hi


# ---------------------------------------------------------------------------
# The sampler, on a conjugate Gaussian
# ---------------------------------------------------------------------------

D = 6
MU = torch.tensor([1.5, -0.5, 0.0, 2.0, 0.3, -1.0], dtype=torch.float64)
S2 = 0.25                                    # likelihood variance per coordinate


def _loglik(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((x - MU) ** 2).sum(-1) / S2


def _posterior():
    var = 1.0 / (1.0 + 1.0 / S2)
    mean = var * MU / S2
    # log of the integral of N(x; 0, I) exp(_loglik(x)): the likelihood above has no normaliser.
    log_z = float(0.5 * D * math.log(S2 / (1 + S2)) - 0.5 * (MU ** 2).sum() / (1 + S2))
    return mean, var, log_z


def test_elliptical_slice_targets_the_tempered_posterior():
    gen = torch.Generator().manual_seed(0)
    beta = 0.5
    x = torch.randn((4000, D), generator=gen, dtype=torch.float64)
    ll = _loglik(x)
    for _ in range(40):
        x, ll, _ = elliptical_slice(x, ll, _loglik, beta, gen)
    var = 1.0 / (1.0 + beta / S2)
    torch.testing.assert_close(x.mean(0), var * beta * MU / S2, atol=0.03, rtol=0)
    torch.testing.assert_close(x.var(0), torch.full((D,), var, dtype=torch.float64), atol=0.02, rtol=0)
    torch.testing.assert_close(ll, _loglik(x))


def test_smc_recovers_the_posterior_and_the_evidence():
    mean, var, log_z = _posterior()
    r = smc_posterior(lambda x: _loglik(x.double()), D, SMCConfig(n_particles=4000, n_moves=4, n_final=8, n_keep=2),
                      seed=1)
    x = r["xi"][-1].double()
    torch.testing.assert_close(x.mean(0), mean, atol=0.03, rtol=0)
    torch.testing.assert_close(x.var(0), torch.full((D,), var, dtype=torch.float64), atol=0.02, rtol=0)
    assert abs(r["log_evidence"] - log_z) < 0.15
    assert r["betas"][0] == 0.0 and r["betas"][-1] == 1.0 and len(r["betas"]) > 2
    assert r["xi"].shape == (2, 4000, D) and r["ll_trace"].shape == (8, 4000)


def test_smc_from_a_given_start_skips_the_annealing():
    mean, var, _ = _posterior()
    x0 = mean.expand(3000, D).clone()                      # every chain at one posterior point
    r = smc_posterior(lambda x: _loglik(x.double()), D, SMCConfig(n_particles=3000, n_final=30, n_keep=1),
                      xi0=x0, seed=2)
    assert list(r["betas"]) == [0.0, 1.0] and math.isnan(r["log_evidence"])
    torch.testing.assert_close(r["xi"][-1].double().var(0), torch.full((D,), var, dtype=torch.float64), atol=0.02, rtol=0)


def test_slice_sweep_targets_the_tempered_posterior_and_records_by_step_count():
    from chai.eval.mcmc import slice_sweep

    gen = torch.Generator().manual_seed(4)
    beta = 0.5
    x = torch.randn((4000, D), generator=gen, dtype=torch.float64)
    recs, n_evals, n_stuck = slice_sweep(x, _loglik(x), _loglik, beta, gen, (10, 40))
    var = 1.0 / (1.0 + beta / S2)
    xi, ll = recs[-1]
    torch.testing.assert_close(xi.mean(0), var * beta * MU / S2, atol=0.03, rtol=0)
    torch.testing.assert_close(xi.var(0), torch.full((D,), var, dtype=torch.float64), atol=0.02, rtol=0)
    torch.testing.assert_close(ll, _loglik(xi))
    assert n_stuck == 0 and n_evals % 4000 == 0 and n_evals >= 40 * 4000
    assert not torch.equal(recs[0][0], recs[1][0])


def test_preconditioned_sweep_keeps_the_target_and_mixes_a_narrow_likelihood_faster():
    """Target N(0, I) L with L sharp in two coordinates and silent in the rest: the
    plain ellipses crawl in the silent ones, the fitted ones do not, and both are exact."""
    from chai.eval.mcmc import Preconditioner, slice_sweep

    d, s2 = 40, 1e-4
    mu = torch.zeros(d, dtype=torch.float64)
    mu[:2] = 0.5

    def ll(x: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((x[:, :2] - mu[:2]) ** 2).sum(-1) / s2

    var = torch.ones(d, dtype=torch.float64)
    var[:2] = 1.0 / (1.0 + 1.0 / s2)
    mean = torch.zeros(d, dtype=torch.float64)
    mean[:2] = var[:2] * mu[:2] / s2
    gen = torch.Generator().manual_seed(5)
    n = 3000
    start = mean.expand(n, d).clone()                       # every chain at the posterior mean
    pop = mean + torch.randn((n, d), generator=gen, dtype=torch.float64) * var.sqrt()
    pc = Preconditioner.fit(pop)
    spread = {}
    for name, p in (("plain", None), ("fitted", pc)):
        ((x, lx),), _, _ = slice_sweep(start, ll(start), ll, 1.0, gen, (30,), precond=p)
        torch.testing.assert_close(lx, ll(x))
        spread[name] = float(x[:, 2:].var(0).mean())
    assert spread["fitted"] > 0.9 and spread["plain"] < 0.2
    ((x, _),), _, _ = slice_sweep(start, ll(start), ll, 1.0, gen, (60,), precond=pc)
    torch.testing.assert_close(x.mean(0), mean, atol=0.05, rtol=0)
    torch.testing.assert_close(x.var(0), var, atol=0.08, rtol=0.05)


def test_preconditioner_fit_recovers_a_covariance_it_can_see():
    from chai.eval.mcmc import Preconditioner

    gen = torch.Generator().manual_seed(6)
    A = torch.randn((10, 10), generator=gen, dtype=torch.float64) * 0.3 + torch.eye(10, dtype=torch.float64)
    x = torch.randn((20_000, 10), generator=gen, dtype=torch.float64) @ A.T + 2.0
    pc = Preconditioner.fit(x)
    torch.testing.assert_close(pc.mu, torch.full((10,), 2.0, dtype=torch.float64), atol=0.05, rtol=0)
    torch.testing.assert_close(pc.L @ pc.L.T, A @ A.T, atol=0.08, rtol=0.05)
