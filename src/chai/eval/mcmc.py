"""Exact posteriors by MCMC in the prior's own coordinates, for the queries the corpus cannot serve.

:mod:`chai.eval.reference` weights the 1.6 x 10^6 stored train histories
with the exact likelihood, and where the data are sharp enough that only a
handful of them fit, its effective sample size collapses (23 of 130 queries
below 1 000, ``docs/sharpness.md`` §1).  This module is the supplement: a
sampler that is exact for the *same* prior and the *same* likelihood, and whose
Monte Carlo error does not depend on how much prior mass the data leave.

The coordinates
---------------
The prior (:class:`chai.data.priors.ThermalHistoryPrior`) has no tractable
density on the 64 stored nodes -- the walls put atoms on it and Stage 10
measured its whitened marginals at excess kurtosis 40-160 -- but it is a
deterministic map of independent draws with known densities: four
hyperparameters, 256 random-feature frequencies and 256 phases.  Each of those
is a transform of standard normals,

.. code-block:: text

    tau      = exp(log lo + (log hi - log lo) Phi(h_0))       LogUniform, exactly
    rate_sd  = exp(log lo + (log hi - log lo) Phi(h_1))
    drift    = (lo + (hi - lo) Phi(h_2)) / t_max              Uniform, exactly
    T_0      = lo + (hi - lo) Phi(h_3)
    omega_j  = e_j / tau                                      N(0, 1/tau^2), exactly
    phi_j    = atan2(b_j, a_j)                                U(-pi, pi], exactly

so with ``xi = (h, e, a, b)`` in ``R^772`` the prior is ``N(0, I)`` and the
path is ``T(xi)``, solved by the same RK4-with-gates integration and read off
at the nodes by the same Hermite interpolant the corpus used
(:func:`generate`, checked against :meth:`ThermalHistoryPrior.sample_on` on
the corpus's own rows).  The radius ``|(a_j, b_j)|`` is an auxiliary variable
the path ignores; it costs nothing and makes the phase Gaussian.  The posterior
is ``N(xi; 0, I) p(x | T(xi))``, exactly, with no density to approximate.

The sampler
-----------
A Gaussian prior times a likelihood is what **elliptical slice sampling**
(Murray, Adams & MacKay 2010) is for: every move is on an ellipse through the
current point and a fresh prior draw, it is always accepted, it has no step
size, and its efficiency does not degrade with dimension, because the prior
part of the target is handled exactly and only the likelihood is sliced.  It
needs no gradient -- the gates and the clip make ``T(xi)`` only piecewise
smooth, and the gradient of RDAAM through 2 048 RK4 steps is the expensive
part of the forward model -- and it is batched: every chain draws its slice
level and ellipse at once, and only chains still shrinking are evaluated
again (:func:`elliptical_slice`).

The likelihood of a thin-corner query is concentrated far from where the
prior puts its mass, so chains are not started from the prior at the
posterior.  :func:`smc_posterior` anneals instead, ``p_beta ∝ N(xi) L^beta``
from ``beta = 0`` to 1, choosing each increment so that the particles' ESS
falls to half, resampling, and moving every particle with a few slice steps at
the new temperature -- a sequential Monte Carlo sampler whose moves are the
slice sampler.  It never reads the pool, so its answer is an independent check
on :func:`chai.eval.reference.reference_posterior` wherever that one is well
supported, and the log-evidence it accumulates is a second check on the same.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from chai.data.priors import ThermalHistoryPrior

__all__ = ["GenerativeMap", "Preconditioner", "SMCConfig", "corpus_latent", "elliptical_slice", "latent_of_draws", "slice_sweep",
           "smc_posterior"]


class GenerativeMap:
    """``xi`` in ``R^(4 + 3 M)`` standard normal to paths on a grid: the prior, as a function.

    Args:
        prior: the corpus's prior; its ranges, cap, walls, feature count and
            RK4 step count are all used as they are.
        t_grid: ``(n_grid,)`` the node times the paths are read off at.
        device, dtype: where the map runs.  float32 reproduces the numpy fp64
            prior to a few hundredths of a kelvin (``tests/test_mcmc.py``),
            far inside the 0.5 K floor the heads train on.
        chunk: half-grid times per random-feature evaluation; memory only.
    """

    def __init__(self, prior: ThermalHistoryPrior, t_grid: np.ndarray, *, device: Any = "cpu",
                 dtype: torch.dtype = torch.float32, chunk: int = 512):
        self.prior = prior
        self.device = torch.device(device)
        self.dtype = dtype
        self.M = prior.n_features
        self.dim = 4 + 3 * self.M
        self.t_grid = torch.as_tensor(np.asarray(t_grid, float), dtype=dtype, device=self.device)
        self.t_half = torch.linspace(0.0, prior.t_max, 2 * prior.n_steps + 1, dtype=dtype,
                                     device=self.device)
        self.chunk = chunk

    # -- the four hyperparameters -------------------------------------------------

    def _unit(self, h: Tensor) -> Tensor:
        return torch.special.ndtr(h).clamp(1e-12, 1.0 - 1e-12)

    def hyper(self, xi: Tensor) -> dict[str, Tensor]:
        """The stored hyperparameters at ``xi`` ``(B, dim)``, in :data:`HYPER_NAMES` order and units."""
        p = self.prior
        u = self._unit(xi[:, :4])
        (tl, th), (sl, sh) = (np.log(p.tau_range), np.log(p.rate_sd_range))
        (dl, dh), (Tl, Th) = p.drift_total_range, p.T_present_range
        return {"tau": torch.exp(tl + (th - tl) * u[:, 0]),
                "rate_sd": torch.exp(sl + (sh - sl) * u[:, 1]),
                "drift": (dl + (dh - dl) * u[:, 2]) / p.t_max,
                "T_0": Tl + (Th - Tl) * u[:, 3]}

    # -- the path -------------------------------------------------------------------

    def rate(self, xi: Tensor, hyper: dict[str, Tensor]) -> Tensor:
        """``(B, 2 n_steps + 1)`` raw rate ``u`` on the half-grid, before the cap."""
        M = self.M
        omega = xi[:, 4:4 + M] / hyper["tau"][:, None]
        phase = torch.atan2(xi[:, 4 + 2 * M:4 + 3 * M], xi[:, 4 + M:4 + 2 * M])
        z = torch.empty((xi.shape[0], self.t_half.numel()), dtype=self.dtype, device=self.device)
        for lo in range(0, self.t_half.numel(), self.chunk):
            t = self.t_half[lo:lo + self.chunk]
            z[:, lo:lo + t.numel()] = torch.cos(omega[:, None, :] * t[None, :, None]
                                                + phase[:, None, :]).sum(-1)
        return hyper["drift"][:, None] + hyper["rate_sd"][:, None] * z * math.sqrt(2.0 / M)

    def __call__(self, xi: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """``(T (B, n_grid), hyper)``: the path at the nodes and the stored hyperparameters."""
        p = self.prior
        xi = xi.to(self.dtype)
        hyper = self.hyper(xi)
        v = p.rate_max * torch.tanh(self.rate(xi, hyper) / p.rate_max)
        N = p.n_steps
        h = p.t_max / N
        lo, hi = p.T_min, p.T_max

        def gate(T: Tensor, vv: Tensor) -> Tensor:
            return vv * torch.where(vv > 0.0, T < hi, T > lo)

        T = torch.empty((xi.shape[0], N + 1), dtype=self.dtype, device=self.device)
        T[:, 0] = hyper["T_0"]
        Tk = hyper["T_0"]
        for k in range(N):
            va, vm, vb = v[:, 2 * k], v[:, 2 * k + 1], v[:, 2 * k + 2]
            k1 = gate(Tk, va)
            k2 = gate(Tk + 0.5 * h * k1, vm)
            k3 = gate(Tk + 0.5 * h * k2, vm)
            k4 = gate(Tk + h * k3, vb)
            Tk = (Tk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).clamp(lo, hi)
            T[:, k + 1] = Tk
        dTdt = gate(T, v[:, ::2])
        # The corpus's cubic Hermite read-off (HistoryBatch.at), at the nodes.
        tq = self.t_grid.clamp(0.0, p.t_max)
        k = (tq / h).long().clamp(0, N - 1)
        s = (tq - k.to(self.dtype) * h) / h
        s2, s3 = s * s, s * s * s
        out = ((2 * s3 - 3 * s2 + 1) * T[:, k] + h * (s3 - 2 * s2 + s) * dTdt[:, k]
               + (-2 * s3 + 3 * s2) * T[:, k + 1] + h * (s3 - s2) * dTdt[:, k + 1])
        return out, hyper


def latent_of_draws(prior: ThermalHistoryPrior, hyper: dict[str, np.ndarray], omega: np.ndarray,
                    phase: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``(n, dim)`` the ``xi`` of draws the numpy prior made, given what it drew.

    The inverse of :class:`GenerativeMap` on its image: inverse CDFs for the
    hyperparameters, ``omega tau`` for the frequencies, and each phase placed
    on a circle of Rayleigh-distributed radius (drawn from ``rng``), which is
    exactly how a standard normal pair decomposes.  Used by the tests, and to
    start a chain at a query's true latent, which under the joint is an exact
    posterior draw.
    """
    from scipy.special import ndtri

    def inv_log(x, r):
        return ndtri(np.clip((np.log(x) - np.log(r[0])) / (np.log(r[1]) - np.log(r[0])), 1e-12, 1 - 1e-12))

    def inv_lin(x, r):
        return ndtri(np.clip((x - r[0]) / (r[1] - r[0]), 1e-12, 1 - 1e-12))

    h = np.stack([inv_log(hyper["tau"], prior.tau_range), inv_log(hyper["rate_sd"], prior.rate_sd_range),
                  inv_lin(hyper["drift"] * prior.t_max, prior.drift_total_range),
                  inv_lin(hyper["T_0"], prior.T_present_range)], 1)
    e = omega * hyper["tau"][:, None]
    r = np.sqrt(rng.chisquare(2, size=phase.shape))
    return np.concatenate([h, e, r * np.cos(phase), r * np.sin(phase)], 1)


def corpus_latent(path: str, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``(len(rows), dim)`` the ``xi`` of stored corpus rows, redrawn from the corpus's provenance.

    Each row's block is redrawn exactly as :func:`chai.data.simulate.reproduce`
    does, up to the latents (no integration, no oracles), and inverted by
    :func:`latent_of_draws`.  For a query's own row this is its true latent:
    the data were drawn from it, so under the joint it is an exact draw from
    the posterior, and a chain started there is stationary from its first step.
    """
    import json

    import h5py

    from chai.data.simulate import CorpusSpec, _seed_sequence

    with h5py.File(path, "r") as f:
        spec = CorpusSpec(**{k: tuple(v) if k == "split_fractions" else v
                             for k, v in json.loads(f.attrs["spec"]).items()})
        prior = ThermalHistoryPrior.from_config(json.loads(f.attrs["prior_config"]))
        blocks = f["block"][:][np.asarray(rows)]
    children = _seed_sequence(spec).spawn(spec.n_blocks)
    out = np.empty((len(rows), 4 + 3 * prior.n_features))
    for j, (row, b) in enumerate(zip(np.asarray(rows), blocks)):
        sl = spec.block_slice(int(b))
        gen = np.random.Generator(np.random.PCG64(children[int(b)]))
        hyper, omega, phase = prior.draw_latents(sl.stop - sl.start, gen)
        i = int(row) - sl.start
        out[j] = latent_of_draws(prior, {k: v[i:i + 1] for k, v in hyper.items()}, omega[i:i + 1],
                                 phase[i:i + 1], rng)[0]
    return out


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def elliptical_slice(xi: Tensor, ll: Tensor, log_lik: Callable[[Tensor], Tensor], beta: float,
                     gen: torch.Generator, *, max_shrink: int = 60) -> tuple[Tensor, Tensor, int]:
    """One elliptical slice step for every chain at once, targeting ``N(0, I) L^beta``.

    Args:
        xi: ``(B, dim)`` current states.
        ll: ``(B,)`` their log-likelihoods (``beta = 1`` scale).
        log_lik: ``(n, dim) -> (n,)``; only called on chains still shrinking.
        beta: the tempering exponent.
        gen: random stream.
        max_shrink: a chain that has not accepted after this many shrinks
            stays where it is -- a move to itself, which is valid -- and is
            counted; the bracket is then narrower than a float can resolve.

    Returns:
        ``(xi, ll, n_evals)``.
    """
    B, D = xi.shape
    dev = xi.device
    nu = torch.randn((B, D), generator=gen, device=dev, dtype=xi.dtype)
    log_y = beta * ll + torch.log(torch.rand(B, generator=gen, device=dev, dtype=ll.dtype))
    theta = torch.rand(B, generator=gen, device=dev, dtype=xi.dtype) * (2 * math.pi)
    lo, hi = theta - 2 * math.pi, theta.clone()
    out_xi, out_ll = xi.clone(), ll.clone()
    active = torch.arange(B, device=dev)
    n_evals = 0
    for _ in range(max_shrink):
        th = theta[active]
        prop = xi[active] * torch.cos(th)[:, None] + nu[active] * torch.sin(th)[:, None]
        lp = log_lik(prop)
        n_evals += int(active.numel())
        ok = beta * lp > log_y[active]
        acc = active[ok]
        out_xi[acc], out_ll[acc] = prop[ok], lp[ok]
        rej = active[~ok]
        if rej.numel() == 0:
            active = rej
            break
        # Shrink the bracket towards zero, then draw inside it.
        t = theta[rej]
        neg = t < 0
        lo[rej] = torch.where(neg, t, lo[rej])
        hi[rej] = torch.where(neg, hi[rej], t)
        theta[rej] = lo[rej] + torch.rand(rej.numel(), generator=gen, device=dev, dtype=xi.dtype) * (hi[rej] - lo[rej])
        active = rej
    return out_xi, out_ll, n_evals


@dataclass(frozen=True)
class Preconditioner:
    """A Gaussian ``N(mu, L L^T)`` for the ellipses to live on, in place of the prior's ``N(0, I)``.

    Elliptical slice sampling proposes on ellipses drawn from whatever Gaussian
    it is told the prior is.  With the prior's own ``N(0, I)`` and a likelihood
    far narrower than it, the accepted angles shrink to a few hundredths and
    *every* coordinate moves that little per step -- including the hundreds
    the data say nothing about, which then take ~``1 / theta^2`` steps to mix.
    Writing the target as ``N(xi; mu, S) * [N(xi; 0, I) L^beta / N(xi; mu, S)]``
    and slicing on the bracket instead (Nishihara, Murray & Adams 2014) leaves
    the target exactly as it was, and with ``N(mu, S)`` fitted to the current
    population the ellipses are the posterior's own shape: wide where the data
    are silent, narrow where they are not.  Any ``mu, S`` is valid; the fit
    only decides how fast the chains move.
    """

    mu: Tensor
    L: Tensor
    nu: float | None = 8.0

    def mahalanobis(self, x: Tensor) -> Tensor:
        z = torch.linalg.solve_triangular(self.L, (x - self.mu).T, upper=False).T
        return (z * z).sum(-1)

    def log_ratio(self, x: Tensor) -> Tensor:
        """``log N(x; 0, I) - log q(x)``, up to a constant, ``q`` the Gaussian or Student-t fit."""
        Q = self.mahalanobis(x)
        d = self.mu.numel()
        log_q = -0.5 * Q if self.nu is None else -0.5 * (self.nu + d) * torch.log1p(Q / self.nu)
        return -0.5 * (x * x).sum(-1) - log_q

    def draw(self, n: int, gen: torch.Generator, x: Tensor | None = None) -> Tensor:
        """``(n, dim)`` draws from ``N(0, s S)`` (centred: the ellipse adds ``mu`` back).

        For the Student-t fit, ``s`` is the scale-mixture variable drawn
        exactly from its conditional given each chain's state ``x``,
        ``InvGamma((nu + d) / 2, (nu + Q(x)) / 2)`` -- the Gibbs half of
        Nishihara et al.'s generalised elliptical slice sampler.  It sets each
        ellipse to the chain's own distance from the fit, which is what keeps
        a chain in the fit's tails from sticking there.
        """
        d = self.mu.numel()
        z = torch.randn((n, d), generator=gen, device=self.mu.device, dtype=self.mu.dtype)
        v = z @ self.L.T
        if self.nu is None or x is None:
            return v
        Q = self.mahalanobis(x)
        shape = torch.full((n,), 0.5 * (self.nu + d), dtype=self.mu.dtype, device=self.mu.device)
        g = torch._standard_gamma(shape, generator=gen)       # Gamma(shape, 1)
        s = 0.5 * (self.nu + Q) / g
        return v * s.sqrt()[:, None]

    @classmethod
    def fit(cls, x: Tensor, *, floor: float = 1e-3, nu: float | None = 8.0) -> Preconditioner:
        """Mean and shrinkage covariance of a population ``x`` ``(n, dim)``.

        The sample covariance of ~1 000 particles in 772 dimensions is
        singular, so it is shrunk towards its own diagonal by the Schafer &
        Strimmer (2005) estimate of the optimal intensity -- the sampling
        variance of the off-diagonal correlations against their size, both in
        closed form -- which needs no tuning.
        """
        x = x.double()
        n = x.shape[0]
        mu = x.mean(0)
        xc = x - mu
        sd = xc.pow(2).mean(0).sqrt().clamp_min(floor)
        z = xc / sd
        r = z.T @ z / n                                        # correlations, biased
        w2 = (z * z).T @ (z * z) / n                           # mean of (z_i z_j)^2 over the population
        var_r = (w2 - r * r) * n / (n - 1) ** 2                # sampling variance of each r_ij
        off = ~torch.eye(r.shape[0], dtype=torch.bool, device=x.device)
        lam = float((var_r[off].sum() / (r[off] * r[off]).sum().clamp_min(1e-300)).clamp(0.0, 1.0))
        corr = (1.0 - lam) * r
        corr.fill_diagonal_(1.0)
        cov = corr * sd[:, None] * sd[None, :]
        L = torch.linalg.cholesky(cov + floor ** 2 * torch.eye(cov.shape[0], dtype=cov.dtype, device=x.device))
        return cls(mu, L, nu)


@torch.no_grad()
def slice_sweep(xi: Tensor, ll: Tensor, log_lik: Callable[[Tensor], Tensor], beta: float,
                gen: torch.Generator, record_at: tuple[int, ...], *, precond: Preconditioner | None = None,
                max_shrink: int = 60) -> tuple[list[tuple[Tensor, Tensor]], int, int]:
    """``max(record_at)`` elliptical slice steps per chain, every call at full batch.

    :func:`elliptical_slice` finishes one step for every chain before any
    starts the next, so its last shrink rounds evaluate a handful of
    stragglers -- and the forward models cost the same ~0.4 s per call at 5
    paths as at 4 000, because RK4 and RDAAM are thousands of sequential
    kernel launches.  Here each chain runs its own slice sampler out of step
    with the others: every round evaluates one proposal for **every** chain,
    a chain that accepts starts its next ellipse on the next round, and one
    that rejects shrinks.  Each chain is still exactly an elliptical slice
    sampler; only the bookkeeping is shared.  A chain's state is recorded when
    *its own* completed-step count reaches each entry of ``record_at`` -- never
    at a wall-clock time, which would favour states whose steps happen to be
    quick -- and chains that are done keep stepping, unrecorded, until the
    slowest is.

    With ``precond`` the ellipses are drawn from it and the slice is on
    ``beta * ll + precond.log_ratio``; without, from ``N(0, I)`` on
    ``beta * ll``.  The target is ``N(0, I) L^beta`` either way.

    Returns:
        ``(records, n_evals, n_stuck)``: one ``(xi, ll)`` per ``record_at``
        entry; ``n_stuck`` counts steps that hit ``max_shrink`` and stayed put.
    """
    B, D = xi.shape
    dev = xi.device
    x, lx = xi.clone(), ll.clone()
    if precond is None:
        mu = torch.zeros(D, dtype=x.dtype, device=dev)

        def extra(v: Tensor) -> Tensor:
            return torch.zeros(v.shape[0], dtype=lx.dtype, device=dev)

        def draw(idx: Tensor) -> Tensor:
            return torch.randn((idx.numel(), D), generator=gen, device=dev, dtype=x.dtype)
    else:
        pc = Preconditioner(precond.mu.to(dev, torch.float64), precond.L.to(dev, torch.float64), precond.nu)
        mu = pc.mu.to(x.dtype)

        def extra(v: Tensor) -> Tensor:
            return pc.log_ratio(v.double()).to(lx.dtype)

        def draw(idx: Tensor) -> Tensor:
            return pc.draw(idx.numel(), gen, x[idx].double()).to(x.dtype)

    ex = extra(x)
    targets = sorted({int(k) for k in record_at})
    rec = {k: (torch.empty_like(x), torch.empty_like(lx)) for k in targets}
    done = torch.zeros(B, dtype=torch.long, device=dev)
    shrinks = torch.zeros(B, dtype=torch.long, device=dev)
    nu = torch.empty_like(x)
    log_y, theta = torch.empty_like(lx), torch.empty(B, dtype=x.dtype, device=dev)
    lo, hi = torch.empty_like(theta), torch.empty_like(theta)

    def fresh(idx: Tensor) -> None:
        nu[idx] = draw(idx)
        log_y[idx] = beta * lx[idx] + ex[idx] + torch.log(torch.rand(idx.numel(), generator=gen, device=dev,
                                                                      dtype=lx.dtype))
        th = torch.rand(idx.numel(), generator=gen, device=dev, dtype=x.dtype) * (2 * math.pi)
        theta[idx], lo[idx], hi[idx] = th, th - 2 * math.pi, th
        shrinks[idx] = 0

    fresh(torch.arange(B, device=dev))
    n_evals, n_stuck = 0, 0
    last = targets[-1]
    while int(done.min()) < last:
        prop = mu + (x - mu) * torch.cos(theta)[:, None] + nu * torch.sin(theta)[:, None]
        lp = log_lik(prop)
        ep = extra(prop)
        n_evals += B
        ok = beta * lp + ep > log_y
        stuck = ~ok & (shrinks >= max_shrink)
        n_stuck += int((stuck & (done < last)).sum())
        x = torch.where(ok[:, None], prop, x)
        lx = torch.where(ok, lp, lx)
        ex = torch.where(ok, ep, ex)
        fin = ok | stuck
        done = done + fin.long()
        for k in targets:
            hit = fin & (done == k)
            if bool(hit.any()):
                rec[k][0][hit], rec[k][1][hit] = x[hit], lx[hit]
        rej = torch.nonzero(~fin).squeeze(-1)
        t = theta[rej]
        neg = t < 0
        lo[rej] = torch.where(neg, t, lo[rej])
        hi[rej] = torch.where(neg, hi[rej], t)
        theta[rej] = lo[rej] + torch.rand(rej.numel(), generator=gen, device=dev, dtype=x.dtype) * (hi[rej] - lo[rej])
        shrinks[rej] += 1
        fresh(torch.nonzero(fin).squeeze(-1))
    return [rec[k] for k in targets], n_evals, n_stuck


@dataclass(frozen=True)
class SMCConfig:
    """Particles, temperatures and moves.

    Args:
        n_particles: population size.  Also the number of chains at ``beta = 1``.
        target_ess: fraction of the population the ESS may fall to between
            temperatures; the next ``beta`` is the bisection root.
        n_moves: slice steps per particle at each intermediate temperature.
        n_final: slice steps per particle at ``beta = 1``, after the last
            resample.  Draws are taken from the last ``n_keep`` of them.
        n_keep: states per chain retained at ``beta = 1`` (every
            ``n_final // n_keep``-th step).
        eval_batch: states per likelihood call.
        precondition: slice on ellipses fitted to the population
            (:class:`Preconditioner`) rather than the prior's ``N(0, I)``.
    """

    n_particles: int = 1024
    target_ess: float = 0.5
    n_moves: int = 4
    n_final: int = 24
    n_keep: int = 4
    eval_batch: int = 4096
    precondition: bool = True

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


def _systematic(w: Tensor, gen: torch.Generator) -> Tensor:
    n = w.numel()
    u = (torch.rand(1, generator=gen, device=w.device, dtype=torch.float64)
         + torch.arange(n, device=w.device, dtype=torch.float64)) / n
    c = torch.cumsum(w.double(), 0)
    c = c / c[-1]
    return torch.searchsorted(c, u).clamp(max=n - 1)


def _next_beta(ll: Tensor, beta: float, target: float) -> float:
    """Largest ``beta' <= 1`` at which the incremental weights ``L^(beta' - beta)`` keep ESS >= target."""
    ll = ll.double()

    def ess(b: float) -> float:
        lw = (b - beta) * ll
        w = torch.exp(lw - lw.max())
        return float(w.sum() ** 2 / (w * w).sum()) / ll.numel()

    if ess(1.0) >= target:
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ess(mid) >= target:
            lo = mid
        else:
            hi = mid
    return max(lo, beta + 1e-12)


@torch.no_grad()
def smc_posterior(log_lik: Callable[[Tensor], Tensor], dim: int, cfg: SMCConfig | None = None, *,
                  device: Any = "cpu", seed: int = 0, xi0: Tensor | None = None,
                  precond: Preconditioner | None = None, verbose: bool = False) -> dict[str, Any]:
    """Anneal from ``N(0, I)`` to ``N(0, I) L`` with slice moves; return the chains at ``beta = 1``.

    Args:
        log_lik: ``(n, dim) -> (n,)`` the exact log-likelihood of ``T(xi)``.
        dim: latent dimension.
        xi0: optional ``(n_particles, dim)`` start at ``beta = 1`` instead of
            annealing -- e.g. every chain at a query's true latent, which under
            the joint is an exact posterior draw; the chains then only have to
            decorrelate, not find the posterior.
        precond: the ellipses' Gaussian for the ``xi0`` route (an annealed
            run's ``precond``, say).  When annealing, and when
            ``cfg.precondition``, each temperature's is fitted to the
            population it starts from (:meth:`Preconditioner.fit`), before
            resampling, so the particles are distinct and a shade wider than
            the new target -- the safe side.

    Returns:
        ``xi`` ``(n_keep, n_particles, dim)`` the retained states, ``ll`` their
        log-likelihoods, ``betas`` the temperature ladder, ``log_evidence``
        (the product of the mean incremental weights: ``log p(x)`` under the
        prior, the same quantity the importance sampler estimates), the
        per-chain log-likelihood trace at ``beta = 1`` (for mixing), and the
        likelihood-evaluation count and wall-clock, and the last
        ``precond`` used.
    """
    cfg = cfg if cfg is not None else SMCConfig()
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = cfg.n_particles
    t0 = time.perf_counter()
    evals = 0

    def batched(x: Tensor) -> Tensor:
        return torch.cat([log_lik(x[i:i + cfg.eval_batch]) for i in range(0, x.shape[0], cfg.eval_batch)])

    betas = [0.0]
    log_z = 0.0
    n_stuck = 0
    if xi0 is None:
        xi = torch.randn((n, dim), generator=gen, device=dev)
        ll = batched(xi)
        evals += n
        beta = 0.0
        while beta < 1.0:
            if cfg.precondition:
                precond = Preconditioner.fit(xi)
            nb = _next_beta(ll, beta, cfg.target_ess)
            lw = (nb - beta) * ll.double()
            m = lw.max()
            log_z += float(m + torch.log(torch.exp(lw - m).mean()))
            idx = _systematic(torch.exp(lw - m), gen)
            xi, ll = xi[idx], ll[idx]
            beta = nb
            betas.append(beta)
            if beta < 1.0:
                recs, e, st = slice_sweep(xi, ll, batched, beta, gen, (cfg.n_moves,), precond=precond)
                (xi, ll), = recs
                evals += e
                n_stuck += st
            if verbose:
                print(f"      beta {beta:.3e}  ll p50 {float(ll.median()):9.2f}  "
                      f"unique {torch.unique(idx).numel():5d}  evals {evals:9,d}  "
                      f"{time.perf_counter() - t0:6.0f} s", flush=True)
    else:
        xi = xi0.to(dev).clone()
        ll = batched(xi)
        evals += n
        betas.append(1.0)
        log_z = float("nan")
    every = max(1, cfg.n_final // cfg.n_keep)
    keep_at = tuple(cfg.n_final - every * j for j in range(cfg.n_keep))[::-1]
    recs, e, st = slice_sweep(xi, ll, batched, 1.0, gen, tuple(range(1, cfg.n_final + 1)), precond=precond)
    evals += e
    n_stuck += st
    kept = [recs[k - 1] for k in keep_at]
    return {"xi": torch.stack([k[0] for k in kept]), "ll": torch.stack([k[1] for k in kept]),
            "ll_trace": torch.stack([r[1].cpu() for r in recs]), "betas": np.asarray(betas),
            "log_evidence": log_z, "n_evals": evals, "n_stuck": n_stuck, "time": time.perf_counter() - t0,
            "precond": precond}
