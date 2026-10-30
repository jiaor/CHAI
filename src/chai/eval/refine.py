"""The exact likelihood as a correction to the amortized posterior.

The flow head is fast and, by Stages 8 and 12, honest and about as sharp as
the data allow -- on average.  One query at a time it is still an
approximation, and where it is wrong nothing downstream can tell.  This
module turns it into a **proposal**: draw from it, weight each draw by what
the exact posterior says of it, and report how much the weights had to do.

    w(z) = p(z) L(x | T(z)) / q_mix(z | x),    q_mix = (1 - eps) q(z | x) + eps p(z)

Self-normalised, the weighted draws are a sample from the exact posterior
whatever the flow got wrong, provided ``q_mix`` covers it; the effective
sample size and the Pareto ``k-hat`` of the weights say how well.  The
defensive component is PLAN.md's *(review)* correction of "train
conservative": with ``eps`` of the draws from the prior, ``w <= L / eps``
wherever the prior has mass, so a head that is too narrow somewhere is
*recoverable* -- the prior's draws find what it missed -- where plain
importance sampling from a too-narrow proposal is not.

The density on the prior
------------------------
The corpus prior has no density on the nodes (``docs/mcmc.md`` §1): it is a
772-dimensional map pushed through gates and a clamp.  What it does have, in
the heads' own coordinates, is a density once the walls are dequantised and
every node carries the 0.5 K floor -- which is exactly what the flow head was
trained on, so :class:`PriorDensity` is the same model trained on empty
samples (``TrainConfig.unconditional``).  Both densities then live on the
same dequantised ``z``, and the target the weights define is

    pi(z | x) ∝ p_hat(z) L(x | clamp(T(z)))

whose path marginal is the corpus posterior blurred by the floor.  The
clamp undoes the wall dequantisation exactly; the 0.5 K floor on interior
nodes remains, and ``docs/refinement.md`` measures what it does to ``L``.
``p_hat`` is a fitted density, so the correction is exact relative to it and
no better than it: a prior flow that is wrong about the prior moves the
answer, and the study scores the refined posterior against the corpus
reference, which uses no fitted prior at all, to see by how much.

The mixture is sampled **deterministically**: ``round(eps n)`` draws from
the prior and the rest from the head, each weighted against the full
mixture density (the balance heuristic).  Unbiased for the self-normalised
estimator like the random mixture, with less variance.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from chai.data.observe import ObservationModel
from chai.eval.reference import (
    DEFAULT_CONFIG,
    QUANTILES,
    Query,
    ReferenceConfig,
    path_log_likelihood,
    weighted_quantiles,
)
from chai.models.gaussian import AmortizedPosterior
from chai.train.data import empty_features, features_from_aliquot

__all__ = ["PriorDensity", "RefineConfig", "Refined", "ScreenConfig", "ScreenPool", "ess", "pareto_khat", "query_batch",
           "refine", "screened_reference"]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def ess(log_w: Tensor | np.ndarray) -> float:
    """Kish's effective sample size of self-normalised weights ``exp(log_w)``."""
    lw = torch.as_tensor(log_w, dtype=torch.float64)
    lw = lw[torch.isfinite(lw)]
    if lw.numel() == 0:
        return 0.0
    w = torch.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w * w).sum())


def _gpd_fit(x: np.ndarray, prior_k: float = 10.0) -> float:
    """Shape of a generalized Pareto fitted to exceedances ``x`` (> 0, ascending).

    Zhang & Stephens (2009), with the weakly informative shrinkage toward 0.5
    of Vehtari et al. (2024, "Pareto smoothed importance sampling", JMLR) --
    the estimator ArviZ and ``loo`` use.
    """
    n = x.size
    m = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3.0 * x[max(int(n / 4 + 0.5) - 1, 0)]) + 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ll = n * (np.log(-b / k) - k - 1.0)
    wts = 1.0 / np.exp(ll - ll[:, None]).sum(axis=1)
    wts = wts / wts.sum()
    b_post = float((b * wts).sum())
    k_post = float(np.log1p(-b_post * x).mean())
    return (n * k_post + 0.5 * prior_k) / (n + prior_k)


def pareto_khat(log_w: Tensor | np.ndarray) -> float:
    """Pareto ``k-hat`` of the importance weights' upper tail.

    The reliability diagnostic of Vehtari et al. (2024): below ~0.7 the
    self-normalised estimate converges at a usable rate; above it the
    variance of the weights is effectively infinite and the ESS itself is
    not to be trusted, however large it looks.  The tail is the largest
    ``min(0.2 S, 3 sqrt S)`` weights.
    """
    lw = np.sort(np.asarray(torch.as_tensor(log_w, dtype=torch.float64).cpu()))
    lw = lw[np.isfinite(lw)]
    S = lw.size
    M = math.ceil(min(0.2 * S, 3.0 * math.sqrt(S)))
    if M < 5 or S <= M:
        return float("nan")
    top = lw[-M:] - lw[-1]
    cut = lw[-M - 1] - lw[-1]
    x = np.exp(top) - np.exp(cut)
    x = x[x > 0]
    if x.size < 5:
        return float("-inf")                       # a flat tail: weights all equal
    return _gpd_fit(np.sort(x))


# ---------------------------------------------------------------------------
# The prior's density
# ---------------------------------------------------------------------------

class PriorDensity:
    """``p_hat(z)`` on the dequantised latent: an unconditional head's answer.

    Args:
        net: an :class:`AmortizedPosterior` trained with ``unconditional``;
            its latent must be the conditional head's.
        device: where it runs.
    """

    def __init__(self, net: AmortizedPosterior, device: Any = "cpu"):
        c = net.meta.get("config", {})
        if not (c.get("unconditional", False) or float(c.get("p_empty", 0.0) or 0.0) > 0):
            raise ValueError("this head was not trained unconditionally or with empty samples; "
                             "it is not a density on the prior")
        self.net = net.eval().to(device)
        self.device = torch.device(device)
        enc = net.encoder
        f = empty_features(1, enc.n_bins, 1)
        f["grains"] = np.zeros((1, 1, enc.n_grain), dtype=np.float32)
        self._batch = {k: torch.as_tensor(v, device=self.device) for k, v in f.items()}
        with torch.no_grad():
            self._q = self.net(self._batch)

    @classmethod
    def load(cls, run: str | Path, device: Any = "cpu") -> PriorDensity:
        run = Path(run)
        return cls(AmortizedPosterior.load(run / "model.pt" if run.is_dir() else run, map_location=device), device)

    @property
    def latent(self) -> Any:
        return self.net.latent

    @torch.no_grad()
    def log_prob(self, z: Tensor) -> Tensor:
        """``(n,)`` log-density of ``z`` ``(n, dim)``."""
        return self._q.log_prob(z[:, None, :])[:, 0]

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """``(n, dim)`` draws."""
        return self._q.sample(n, generator)[:, 0]


def query_batch(net: AmortizedPosterior, query: Query, device: Any) -> dict[str, Tensor]:
    """The encoder's input for a :class:`Query`."""
    return {k: torch.as_tensor(np.ascontiguousarray(v), device=device)
            for k, v in features_from_aliquot(query.obs).items()}


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefineConfig:
    """
    Args:
        n: proposal draws; each costs one exact likelihood.
        epsilon: share of them drawn from the prior (the defensive component).
        likelihood: the forward models' refinement, as the Stage 8 reference
            ran them.
        chunk: draws per likelihood call.
    """

    n: int = 4000
    epsilon: float = 0.05
    likelihood: ReferenceConfig = DEFAULT_CONFIG
    chunk: int = 8192

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError(f"epsilon must be in [0, 1), got {self.epsilon}")
        if self.n < 2:
            raise ValueError("need at least two draws")

    def to_config(self) -> dict[str, Any]:
        d = asdict(self)
        d["likelihood"] = self.likelihood.to_config()
        return d


@dataclass
class Refined:
    """Weighted draws from the exact posterior, and how much the weights had to do.

    Attributes:
        z: ``(n, dim)`` the proposal draws, dequantised latent.
        T: ``(n, n_grid)`` their paths, clamped to the walls, °C.
        hyper: their hyperparameters.
        log_w: ``(n,)`` unnormalised log-weights.
        from_prior: ``(n,)`` which draws came from the defensive component.
        log_q, log_p, log_lik: ``(n,)`` the three terms of the weight.
        log_q_head: ``(n,)`` the head's own density, without the prior's
            share -- what plain importance sampling from the head would weight
            by, so the study can score that estimator on the same draws.
        ess: effective sample size of the weights.
        khat: Pareto ``k-hat``; above 0.7 the estimate is unreliable.
        log_evidence: ``log p(x)`` estimated under ``p_hat`` -- the mean
            weight.  A sample the prior finds implausible has a low one,
            whatever the head says.
        seconds: ``{"draw": ..., "likelihood": ...}``.
    """

    z: Tensor
    T: Tensor
    hyper: dict[str, Tensor]
    log_w: Tensor
    from_prior: Tensor
    log_q: Tensor
    log_p: Tensor
    log_lik: Tensor
    log_q_head: Tensor
    ess: float
    khat: float
    log_evidence: float
    seconds: dict[str, float]

    @property
    def n(self) -> int:
        return int(self.log_w.numel())

    @property
    def weights(self) -> Tensor:
        """``(n,)`` normalised, float64."""
        lw = self.log_w.double()
        w = torch.exp(lw - lw.max())
        return w / w.sum()

    @property
    def ess_fraction(self) -> float:
        return self.ess / self.n

    @property
    def prior_mass(self) -> float:
        """The share of the posterior weight carried by the prior's draws."""
        return float(self.weights[self.from_prior].sum())

    def resample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """``(n,)`` indices, systematic resampling: an equal-weight sample of the refined posterior."""
        w = self.weights
        u = (torch.rand((), generator=generator, device=w.device, dtype=w.dtype)
             + torch.arange(n, device=w.device, dtype=w.dtype)) / n
        return torch.searchsorted(torch.cumsum(w, 0), u).clamp(max=self.n - 1)

    def summary(self, qs: tuple[float, ...] = QUANTILES) -> dict[str, np.ndarray]:
        """Weighted mean, sd and quantiles per node, °C."""
        w = self.weights
        T = self.T.double()
        mean = (w[:, None] * T).sum(0)
        sd = ((w[:, None] * (T - mean) ** 2).sum(0)).sqrt()
        return {"mean": mean.cpu().numpy(), "sd": sd.cpu().numpy(),
                "q": weighted_quantiles(T, w, qs).cpu().numpy()}


@torch.no_grad()
def refine(net: AmortizedPosterior, prior: PriorDensity, query: Query, observation: ObservationModel,
           cfg: RefineConfig | None = None, *, generator: torch.Generator | None = None) -> Refined:
    """Importance-sample the exact posterior of ``query`` with ``net`` as the proposal.

    ``net`` and ``prior`` must share a latent -- the same whitening and the
    same dequantisation -- or the ratio of their densities is meaningless.
    """
    cfg = cfg or RefineConfig()
    dev = prior.device
    L = net.latent
    if not (torch.equal(L.mean.cpu(), prior.latent.mean.cpu()) and torch.equal(L.W.cpu(), prior.latent.W.cpu())):
        raise ValueError("the head and the prior density are in different coordinates")
    t0 = time.perf_counter()
    q = net(query_batch(net, query, dev))
    n0 = round(cfg.epsilon * cfg.n)
    z = torch.cat([q.sample(cfg.n - n0, generator)[:, 0], prior.sample(n0, generator)]) if n0 else \
        q.sample(cfg.n, generator)[:, 0]
    from_prior = torch.zeros(cfg.n, dtype=torch.bool, device=dev)
    from_prior[cfg.n - n0:] = True
    log_q_head = q.log_prob(z[:, None, :])[:, 0]
    log_p = prior.log_prob(z)
    if cfg.epsilon > 0:
        log_q = torch.logaddexp(math.log1p(-cfg.epsilon) + log_q_head, math.log(cfg.epsilon) + log_p)
    else:
        log_q = log_q_head
    T, hyper = L.inverse(z, clamp=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    t_grid = L.t_grid.to(dev, torch.float32)
    log_lik = torch.cat([path_log_likelihood(T[s:s + cfg.chunk].float(), t_grid, query, observation, cfg.likelihood)
                         for s in range(0, cfg.n, cfg.chunk)])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t2 = time.perf_counter()
    log_w = (log_p + log_lik - log_q).double()
    log_w = torch.where(torch.isfinite(log_w), log_w, torch.full_like(log_w, -math.inf))
    finite = torch.isfinite(log_w)
    log_ev = float(torch.logsumexp(log_w[finite], 0) - math.log(cfg.n)) if bool(finite.any()) else -math.inf
    return Refined(z=z, T=T, hyper=hyper, log_w=log_w, from_prior=from_prior, log_q=log_q, log_p=log_p,
                   log_lik=log_lik, log_q_head=log_q_head, ess=ess(log_w), khat=pareto_khat(log_w), log_evidence=log_ev,
                   seconds={"draw": t1 - t0, "likelihood": t2 - t1})


# ---------------------------------------------------------------------------
# The exact correction that needs no density: the corpus, screened by the head
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenConfig:
    """
    Args:
        rank: how the head orders the pool.  ``"nearest"``: by each row's
            distance to the nearest of ``n_draws`` of the head's paths, in
            units of the head's own sd per node (floored at 1 K) -- a
            statement about where the posterior's mass is, which the head
            gets right (Stage 12).  ``"density"``: by ``log q(z_i | x)`` at the
            row's latent -- a statement about the density at a point, which
            two converged heads disagree on by 5-10 nats (docs/refinement.md).
        n_draws: head draws the nearest-neighbour rank measures against.
        block: pool rows evaluated per step, in the head's order.
        stop_share: the ordered pass stops when a block adds less than this
            share of the mass evaluated so far, twice running.
        max_rows: the most rows the ordered pass evaluates.
        tail: rows drawn uniformly from the remainder and weighted up by the
            number they stand for -- the defensive component, and what keeps
            the estimate unbiased whatever the head's order missed.
        likelihood: the forward models' refinement.
        keep_margin: nats below the top weight at which a row still enters
            the summaries.
    """

    rank: str = "nearest"
    n_draws: int = 512
    block: int = 4096
    stop_share: float = 1e-3
    max_rows: int = 131_072
    tail: int = 8192
    likelihood: ReferenceConfig = DEFAULT_CONFIG
    keep_margin: float = 25.0

    def to_config(self) -> dict[str, Any]:
        d = asdict(self)
        d["likelihood"] = self.likelihood.to_config()
        return d


class ScreenPool:
    """A :class:`~chai.eval.reference.ReferencePool` with every row's latent, for the head to rank.

    Wall nodes are placed at the dequantiser's mean offset beyond the wall
    (``width / 2 + jitter / 2``), with no floor noise: the head's density is
    on dequantised ``z``, and this is a point near the middle of where it put
    each stored path.  Only the ranking reads it; nothing is weighted by it.
    """

    def __init__(self, pool: Any, net: AmortizedPosterior, *, chunk: int = 262_144):
        self.pool = pool
        L = net.latent
        deq = net.meta.get("config", {}).get("dequantise") or {}
        off = 0.5 * float(deq.get("width", 0.0)) + 0.5 * float(deq.get("jitter", 0.0))
        lo = L.box_lo.to(pool.device, torch.float64)
        hi = L.box_hi.to(pool.device, torch.float64)
        names = L.hyper_names
        z = torch.empty((len(pool), L.dim), dtype=torch.float32, device=pool.device)
        Ld = L.to(pool.device)
        with torch.no_grad():
            for s in range(0, len(pool), chunk):
                T = pool.T[s:s + chunk].double()
                T = torch.where(T <= lo, lo - off, torch.where(T >= hi, hi + off, T))
                h = {k: pool.hyper_t[s:s + chunk, i].double() for i, k in enumerate(names)}
                z[s:s + chunk] = Ld(T, h).float()
        self.z = z

    def __len__(self) -> int:
        return len(self.pool)


@torch.no_grad()
def screened_reference(net: AmortizedPosterior, spool: ScreenPool, query: Query, observation: ObservationModel,
                       cfg: ScreenConfig | None = None, *, seed: int = 0, chunk: int = 131_072) -> dict[str, Any]:
    """The corpus posterior of ``query``, with the head choosing which rows get the exact likelihood.

    The pool is a sample from the prior, so a row's weight is its
    likelihood and no density enters.  The head's ``log q(z_i | x)`` orders
    every row; the likelihood runs in that order until a block adds almost
    nothing, and then on a uniform draw from the rest, weighted up by the
    number of rows it stands for.  The two strata together are an unbiased
    estimate of the corpus posterior -- the one ``reference_posterior``
    computes by ordering on the fission-track likelihood -- and the tail's
    measured share of the mass says how much the head's order left behind.
    Returns the summary of
    :func:`~chai.eval.reference.posterior_summary_from_weights` plus the
    screen's record.
    """
    from chai.eval.reference import posterior_summary_from_weights

    cfg = cfg or ScreenConfig()
    pool = spool.pool
    dev = pool.device
    n = len(pool)
    t0 = time.perf_counter()
    q = net(query_batch(net, query, dev))
    if cfg.rank == "density":
        score = torch.cat([q.log_prob(spool.z[s:s + chunk, None, :])[:, 0] for s in range(0, n, chunk)])
    elif cfg.rank == "nearest":
        gen = torch.Generator(device=dev).manual_seed(seed)
        D = net.latent.inverse(q.sample(cfg.n_draws, gen)[:, 0], clamp=True)[0].float()   # (S, n_grid)
        scale = D.std(0).clamp_min(1.0)
        Dn = D / scale
        dn2 = (Dn * Dn).sum(1)
        score = torch.empty(n, dtype=torch.float32, device=dev)
        for s in range(0, n, chunk):
            X = pool.T[s:s + chunk].float() / scale
            d2 = (X * X).sum(1, keepdim=True) + dn2[None, :] - 2.0 * X @ Dn.T
            score[s:s + chunk] = -d2.min(1).values
    else:
        raise ValueError(f"rank must be 'nearest' or 'density', got {cfg.rank!r}")
    score = torch.where(torch.isfinite(score), score, torch.full_like(score, -math.inf))
    order = torch.argsort(score, descending=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_rank = time.perf_counter() - t0

    t0 = time.perf_counter()
    log_w = torch.full((n,), -math.inf, dtype=torch.float64, device=dev)
    pos, small, mass = 0, 0, -math.inf
    while pos < min(n, cfg.max_rows):
        take = order[pos:min(pos + cfg.block, n, cfg.max_rows)]
        lw = path_log_likelihood(pool.T[take], pool.t, query, observation, cfg.likelihood).double()
        log_w[take] = lw
        pos += int(take.numel())
        add = float(torch.logsumexp(lw, 0))
        share = math.exp(add - float(np.logaddexp(mass, add))) if math.isfinite(add) else 0.0
        mass = float(np.logaddexp(mass, add))
        small = small + 1 if share < cfg.stop_share else 0
        if small >= 2:
            break
    n_ordered = pos
    n_tail, tail_mass = 0, 0.0
    if n_ordered < n and cfg.tail > 0:
        rest = order[n_ordered:]
        gen = torch.Generator(device=dev).manual_seed(seed)
        pick = rest[torch.randperm(rest.numel(), generator=gen, device=dev)[:cfg.tail]]
        lw = path_log_likelihood(pool.T[pick], pool.t, query, observation, cfg.likelihood).double()
        log_w[pick] = lw + math.log(rest.numel() / pick.numel())
        n_tail = int(pick.numel())
        tail = float(torch.logsumexp(log_w[pick], 0))
        tail_mass = math.exp(tail - float(np.logaddexp(mass, tail))) if math.isfinite(tail) else 0.0
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_lik = time.perf_counter() - t0
    out = posterior_summary_from_weights(pool, log_w.float(), keep_margin=cfg.keep_margin)
    out.update({"log_w": log_w, "n_ordered": n_ordered, "n_tail": n_tail, "tail_mass": tail_mass,
                "n_evaluated": n_ordered + n_tail, "time_rank": t_rank, "time_lik": t_lik})
    return out
