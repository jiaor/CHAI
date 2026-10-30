"""Reference posteriors from the corpus, by importance sampling with the exact likelihood.

Stage 12 says whether a head is *honest*: its credible intervals hold the
truth as often as they claim.  It cannot say whether the head is *sharp* --
as tight as the data allow -- because that needs the posterior the data
actually imply, and calibration never computes one.  This module does, for
one observed aliquot at a time, without a sampler and without the network:

    the corpus is a sample from the prior, and the likelihood is exact.

Every stored train history ``T_i`` is a draw from ``p(T)``.  Self-normalised
importance sampling with ``w_i = p(x | T_i)`` therefore turns the pool into a
sample from ``p(T | x)`` -- the Stage 1 audit's kernel estimator with the true
likelihood (:meth:`chai.data.observe.ObservationModel.log_prob`) in place
of its Gaussian acceptance kernel.  The kernel was conservative by an unknown
amount (``docs/prior-sensitivity.md`` open item 2); this is the posterior,
exact up to Monte Carlo error wherever the pool is dense, and the effective
sample size says where it is not.

What has to be recomputed, and what does not
--------------------------------------------
The likelihood scores the query's data against a pool history's *predicted*
observables **at the query's own grain properties** -- its Dpar, and the
radius and eU of each helium grain it reports.  The corpus stores every
history's observables at that history's own grains, which is a different
question (the audit module's docstring says why), so the forward models run
again here, over the pool, per query: :func:`aft_forward` at the query's
Dpar for every pool row (15 000 paths/s at ``n_sub = 8`` with two studies
sharing the machine, converged against the oracle to 0.2% in age at the
99th percentile), and :func:`rdaam_age` at the query's grains (3 400
grain-evaluations/s at ``n_sub = 8``, ``rdim = 128``, the same way, within
1% of the oracle on every grain checked against a 6.5% measurement
sigma).  Both figures are measured in ``docs/sharpness.md`` against the stored
oracle values of the pool rows themselves.

Helium is the cost, and most of it is wasted: a pool row whose fission-track
prediction is 200 nats from the data cannot be rescued by its helium ages.
:func:`reference_posterior` therefore evaluates the AFT likelihood over the
whole pool first and the helium likelihood in order of decreasing AFT
likelihood, stopping when **no unevaluated row can matter**: the helium
log-likelihood has a closed-form maximum (a perfect fit to every grain), so
``lpA_i + lpH_max`` bounds row ``i``'s total, and once the best remaining
bound falls ``screen_margin`` nats below the best total already found, every
remaining row carries a relative weight below ``exp(-margin)``.  The screen is
exact, not heuristic; when it cannot close inside ``max_he_rows`` -- a query
with no fission-track analysis has nothing to screen on -- the bound on the
mass dropped is reported alongside the estimate rather than hidden in it.

What is conditioned on
----------------------
The reported grain properties, as any inference on real data would: the
measured Dpar, radius and eU, with each grain's Th/U taken as measured (the
corpus stores U and Th separately; a laboratory measures them separately
too).  Conditioning on reported rather than true grain properties is what
``log_prob`` does by construction and what the network does by necessity,
so the reference and the head see the same data.

Reading the result
------------------
Per node: the weighted mean, sd and quantiles of ``T``; the same for the
functionals Stage 12 ranks and for the stored hyperparameters.  Per query:
the effective sample size, the split-half disagreement of the sd (the
reference's own Monte Carlo error, from the two halves of the pool by
parity), how far the screen got, and the wall-clock -- the denominator behind
"milliseconds instead of hours".
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor

from chai.data.aliquot import Aliquot
from chai.data.observe import ObservationModel
from chai.eval.calibration import FUNCTIONALS, functional_values
from chai.forward.torchimpl.ketcham import aft_forward, length_bin_centres
from chai.forward.torchimpl.rdaam import rdaam_age
from chai.models.latent import HYPER_NAMES

__all__ = [
    "QUANTILES",
    "Query",
    "ReferenceConfig",
    "ReferencePool",
    "aft_log_likelihood",
    "aft_log_likelihood_paths",
    "he_log_likelihood",
    "he_log_likelihood_max",
    "he_log_likelihood_paths",
    "marginal_w1",
    "path_log_likelihood",
    "posterior_summary_from_weights",
    "reference_posterior",
    "weighted_quantiles",
]

#: Quantiles every summary carries: the central 95% and 68% intervals and the median.
QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)

#: The Th/U proxy the corpus uses for effective uranium (``GrainPrior``).
_EU_TH_WEIGHT = 0.235


@dataclass(frozen=True)
class ReferenceConfig:
    """Refinement, batching and the screen.

    Args:
        n_sub_aft: sub-intervals per grid segment for the fission-track port.
            8 is converged against the oracle (``docs/sharpness.md`` §1).
        n_sub_he, rdim: the same for RDAAM, and its radial nodes.  8 / 128 is
            within 1% of the oracle on every grain checked and 2.4x faster
            than 8 / 256, whose gain in accuracy is invisible under a 6.5%
            measurement sigma.
        aft_chunk: pool rows per fission-track forward call.
        he_batch: grain-evaluations per RDAAM call.  16 384 is where the port's
            throughput peaks; larger batches are slower.
        he_block: pool rows taken per screening step.
        screen_margin: nats.  The screen closes when the bound on the weight
            mass of every unevaluated row *together* is this far below the
            mass already evaluated: at 9.2 nats the rows never evaluated
            carry at most one part in ten thousand of the posterior.
        max_he_rows: the most pool rows whose helium is evaluated in order
            for one query.  Reached only when the screen cannot close -- a
            query whose fission-track data is an age alone orders the pool
            weakly, and one with no fission-track analysis not at all.
        tail_sample: when the screen is open at ``max_he_rows`` and the query
            has fission-track data, this many of the rows never reached are
            drawn at random, evaluated, and weighted up by the number they
            stand for.  The estimator is then unbiased whatever the screen
            dropped, and the tail's share of the posterior mass is measured
            rather than bounded.  A query with no fission-track analysis
            skips this: its first ``max_he_rows`` rows are already a uniform
            sub-sample of the prior, and an unbiased estimate as they stand.
        keep_margin: nats below the maximum log-weight at which a row is kept
            for the quantile and functional summaries; ``exp(-25)`` of the top
            weight is far below anything a summary can see.
    """

    n_sub_aft: int = 8
    n_sub_he: int = 8
    rdim: int = 128
    aft_chunk: int = 16_384
    he_batch: int = 16_384
    he_block: int = 16_384
    screen_margin: float = 9.2
    max_he_rows: int = 200_000
    tail_sample: int = 16_384
    keep_margin: float = 25.0

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = ReferenceConfig()


class ReferencePool:
    """The train split's paths on the device, with the stored hyperparameters beside them.

    Args:
        corpus: the corpus file.
        device: where the paths live; ``(n, n_grid)`` float32, 410 MB for the
            full 1.6 x 10^6-row train split.
        max_rows: the first rows of the split only.
        split: which split is the pool.  Train, always, in this project --
            queries come from val and the test split stays shut.
    """

    def __init__(self, corpus: str | Path, *, device: Any = "cuda", max_rows: int | None = None,
                 split: str = "train"):
        self.path = Path(corpus)
        self.device = torch.device(device)
        with h5py.File(self.path, "r") as f:
            names = list(f.attrs["split_names"])
            idx = np.flatnonzero(f["split"][:] == names.index(split))
            if max_rows is not None:
                idx = idx[:max_rows]
            self.index = idx
            self.t_grid = f["t_grid"][:].astype(np.float64)
            T = f["T"][:][idx]
            # A GP prior's latent hyperparameters; a pulse-prior corpus has none the heads model.
            names = HYPER_NAMES if all(f"hyper/{k}" in f for k in HYPER_NAMES) else ()
            self.hyper = {k: f[f"hyper/{k}"][:][idx].astype(np.float64) for k in names}
            self.meta = {"corpus": str(self.path), "split": split, "n_rows": int(idx.size),
                         "corpus_git_sha": str(f.attrs.get("git_sha", ""))}
        self.T = torch.as_tensor(T, dtype=torch.float32, device=self.device)
        self.t = torch.as_tensor(self.t_grid, dtype=torch.float32, device=self.device)
        self.hyper_t = (torch.stack([torch.as_tensor(self.hyper[k], dtype=torch.float32,
                                                     device=self.device) for k in self.hyper], 1)
                        if self.hyper else torch.zeros(len(idx), 0, device=self.device))

    def __len__(self) -> int:
        return int(self.index.size)

    @property
    def n_grid(self) -> int:
        return int(self.t_grid.size)


@dataclass(frozen=True)
class Query:
    """One observed aliquot, and what the reference needs to know about its grains.

    Args:
        obs: the aliquot, length 1, as :meth:`ObservationModel.observe` drew it.
        he_U, he_Th: ``(G,)`` parent concentrations **as conditioned on**: the
            reported eU split by the grain's measured Th/U.  Zero where the
            grain is absent.
        T_true: ``(n_grid,)`` the true path, for coverage; optional.
        row: the corpus row the query came from; optional.
    """

    obs: Aliquot
    he_U: np.ndarray
    he_Th: np.ndarray
    T_true: np.ndarray | None = None
    row: int | None = None

    def __post_init__(self) -> None:
        if len(self.obs) != 1:
            raise ValueError(f"a Query holds one aliquot, got {len(self.obs)}")

    @classmethod
    def from_observation(cls, obs: Aliquot, i: int, *, U_true: np.ndarray, Th_true: np.ndarray,
                         T_true: np.ndarray | None = None, row: int | None = None) -> Query:
        """Row ``i`` of a batch ``obs``, with its grains' Th/U read from the corpus.

        ``U_true`` and ``Th_true`` are ``(G,)`` in the corpus's stored grain
        order; ``obs.meta["grain_order"]`` maps them to the reported order.
        The conditioned-on parents are ``eU_reported / (1 + 0.235 Th/U)`` and
        ``Th/U`` times that, so the eU the network reads and the eU the
        forward model is run at are the same number.
        """
        one = obs[i]
        order = np.asarray(obs.meta["grain_order"])[i] if obs.meta and "grain_order" in obs.meta \
            else np.arange(one.he_age.shape[1])
        ratio = np.asarray(Th_true, float)[order] / np.maximum(np.asarray(U_true, float)[order], 1e-9)
        eU = one.he_eU[0]
        U = eU / (1.0 + _EU_TH_WEIGHT * ratio)
        mask = one.he_mask[0]
        return cls(one, U * mask, U * ratio * mask, T_true, row)

    @property
    def has_aft(self) -> bool:
        return bool(self.obs.aft_mask[0])

    @property
    def n_he(self) -> int:
        return int(self.obs.he_mask[0].sum())


# ---------------------------------------------------------------------------
# The two halves of the likelihood
# ---------------------------------------------------------------------------

def _aft_view(obs: Aliquot) -> Aliquot:
    """``obs`` with its helium grains masked out, so ``log_prob`` returns the AFT terms alone."""
    return replace(obs, he_mask=np.zeros_like(obs.he_mask), meta=None)


def _he_view(obs: Aliquot) -> Aliquot:
    """``obs`` with no fission-track analysis, so ``log_prob`` returns the helium term alone."""
    return replace(obs, aft_mask=np.zeros(len(obs), dtype=bool),
                   length_counts=np.zeros_like(obs.length_counts), meta=None)


def _repeat(obs: Aliquot, n: int) -> Aliquot:
    return obs[np.zeros(n, dtype=np.int64)]


@torch.no_grad()
def aft_log_likelihood_paths(T: Tensor, t: Tensor, query: Query, observation: ObservationModel,
                             cfg: ReferenceConfig = DEFAULT_CONFIG) -> Tensor:
    """``(n,)`` log-likelihood of the query's fission-track data under the paths ``T`` ``(n, n_grid)``.

    Zero everywhere when the query has no fission-track analysis.  The
    forward model runs at the query's reported Dpar, on ``t`` ``(n_grid,)``.
    """
    n = T.shape[0]
    dev = T.device
    if not query.has_aft:
        return torch.zeros(n, dtype=torch.float32, device=dev)
    view = _aft_view(query.obs)
    dpar = float(query.obs.dpar[0])
    G = query.obs.he_age.shape[1]
    bins = length_bin_centres(dtype=torch.float32, device=dev)
    out = torch.empty(n, dtype=torch.float32, device=dev)
    for lo in range(0, n, cfg.aft_chunk):
        Tb = T[lo:lo + cfg.aft_chunk].float()
        b = Tb.shape[0]
        aft = aft_forward(t.expand(b, -1), Tb, torch.full((b,), dpar, device=dev),
                          n_sub=cfg.n_sub_aft, bins=bins)
        pred = {"aft_age": aft["age"], "length_pdf": aft["length_pdf"],
                "he_age": torch.ones((b, G), device=dev)}
        out[lo:lo + b] = observation.log_prob(_repeat(view, b), pred)
    return out


@torch.no_grad()
def he_log_likelihood_paths(T: Tensor, t: Tensor, query: Query, observation: ObservationModel,
                            cfg: ReferenceConfig = DEFAULT_CONFIG) -> Tensor:
    """``(n,)`` log-likelihood of the query's helium ages under the paths ``T`` ``(n, n_grid)``.

    RDAAM runs once per (path, reported grain) at the grain's reported
    radius and the parents :class:`Query` conditions on.  Zero everywhere when
    the query reports no grains.
    """
    m = T.shape[0]
    dev = T.device
    mask = query.obs.he_mask[0]
    grains = np.flatnonzero(mask)
    if grains.size == 0 or m == 0:
        return torch.zeros(m, dtype=torch.float32, device=dev)
    k = grains.size
    G = mask.size
    radius = torch.as_tensor(query.obs.he_radius[0, grains], dtype=torch.float32, device=dev)
    U = torch.as_tensor(query.he_U[grains], dtype=torch.float32, device=dev)
    Th = torch.as_tensor(query.he_Th[grains], dtype=torch.float32, device=dev)
    ages = torch.empty((m, k), dtype=torch.float32, device=dev)
    # Pairs (row, grain) flattened row-major, so a batch is a run of rows.
    total = m * k
    for lo in range(0, total, cfg.he_batch):
        hi = min(lo + cfg.he_batch, total)
        pair = torch.arange(lo, hi, device=dev)
        r, g = pair // k, pair % k
        he = rdaam_age(t.expand(hi - lo, -1), T[r].float(), radius[g], U[g], Th[g],
                       n_sub=cfg.n_sub_he, rdim=cfg.rdim, use_checkpoint=False)
        ages.view(-1)[lo:hi] = he["corrected_age"]
    pred_he = torch.zeros((m, G), dtype=torch.float32, device=dev)
    pred_he[:, torch.as_tensor(grains, device=dev)] = ages
    pred = {"aft_age": torch.ones(m, device=dev),
            "length_pdf": torch.ones((m, query.obs.length_counts.shape[1]), device=dev),
            "he_age": pred_he}
    return observation.log_prob(_repeat(_he_view(query.obs), m), pred)


@torch.no_grad()
def path_log_likelihood(T: Tensor, t: Tensor, query: Query, observation: ObservationModel,
                        cfg: ReferenceConfig = DEFAULT_CONFIG) -> Tensor:
    """``(n,)`` the query's full log-likelihood under the paths ``T``: both halves, no screen."""
    return (aft_log_likelihood_paths(T, t, query, observation, cfg)
            + he_log_likelihood_paths(T, t, query, observation, cfg))


@torch.no_grad()
def aft_log_likelihood(pool: ReferencePool, query: Query, observation: ObservationModel,
                       cfg: ReferenceConfig = DEFAULT_CONFIG) -> Tensor:
    """``(n_pool,)`` log-likelihood of the query's fission-track data under every pool path.

    Zero everywhere when the query has no fission-track analysis.  The
    forward model runs at the query's reported Dpar.
    """
    return aft_log_likelihood_paths(pool.T, pool.t, query, observation, cfg)


@torch.no_grad()
def he_log_likelihood(pool: ReferencePool, rows: Tensor, query: Query,
                      observation: ObservationModel, cfg: ReferenceConfig = DEFAULT_CONFIG
                      ) -> Tensor:
    """``(len(rows),)`` log-likelihood of the query's helium ages under the pool paths ``rows``."""
    return he_log_likelihood_paths(pool.T[rows], pool.t, query, observation, cfg)


@torch.no_grad()
def he_log_likelihood_max(query: Query, observation: ObservationModel, device: Any = "cpu"
                          ) -> float:
    """The largest helium log-likelihood any prediction can reach: a perfect fit to every grain.

    The per-grain term is a log-normal in ``log(obs) - log(pred)``, maximal at
    zero, so the bound is exact and is what makes the screen in
    :func:`reference_posterior` a proof rather than a heuristic.
    """
    view = _he_view(query.obs)
    pred = {"aft_age": torch.ones(1, device=device),
            "length_pdf": torch.ones((1, query.obs.length_counts.shape[1]), device=device),
            "he_age": torch.as_tensor(np.where(view.he_mask, view.he_age, 1.0),
                                      dtype=torch.float32, device=device)}
    return float(observation.log_prob(view, pred)[0])


# ---------------------------------------------------------------------------
# Weights to a posterior
# ---------------------------------------------------------------------------

def weighted_quantiles(x: Tensor, w: Tensor, qs: tuple[float, ...] = QUANTILES) -> Tensor:
    """``(len(qs), k)`` weighted quantiles of ``x`` ``(n, k)`` under weights ``w`` ``(n,)``.

    The usual cumulative-weight definition, column by column: the smallest
    value whose cumulative weight reaches ``q`` of the total.
    """
    order = torch.argsort(x, dim=0)
    xs = torch.gather(x, 0, order)
    ws = w[order]                                              # (n, k)
    c = torch.cumsum(ws, dim=0) / ws.sum(0, keepdim=True)
    q = torch.as_tensor(qs, dtype=c.dtype, device=c.device)
    idx = torch.searchsorted(c.T.contiguous(), q.expand(x.shape[1], -1).contiguous())
    idx = idx.clamp(0, x.shape[0] - 1).T                       # (n_q, k)
    return torch.gather(xs, 0, idx)


@torch.no_grad()
def posterior_summary_from_weights(pool: ReferencePool, log_w: Tensor, *,
                                   keep_margin: float = 25.0) -> dict[str, Any]:
    """What the weighted pool says: moments and quantiles of ``T``, the functionals, the hyperparameters.

    ``log_w`` is ``(n_pool,)`` with ``-inf`` for rows that were never
    evaluated or are out of support.  The split-half sd is the sd estimated
    from the even- and odd-position halves of the pool separately -- two
    independent importance-sampling estimates of the same posterior, whose
    disagreement is the reference's own Monte Carlo error on its width.
    """
    finite = torch.isfinite(log_w)
    if not bool(finite.any()):
        raise ValueError("no pool row has a finite log-weight")
    lw_max = log_w[finite].max()
    w_all = torch.where(finite, torch.exp(log_w - lw_max), torch.zeros_like(log_w))
    sw = w_all.sum()
    ess = float(sw * sw / (w_all * w_all).sum())
    keep = torch.nonzero(log_w > lw_max - keep_margin).squeeze(-1)
    w = w_all[keep].double()
    T = pool.T[keep].double()
    mean = (w[:, None] * T).sum(0) / w.sum()
    var = (w[:, None] * (T - mean) ** 2).sum(0) / w.sum()
    out: dict[str, Any] = {
        "mean": mean.cpu().numpy(), "sd": var.sqrt().cpu().numpy(),
        "quantiles": weighted_quantiles(T, w).cpu().numpy(),
        "ess": ess, "n_kept": int(keep.numel()),
        "top_weight": float(w.max() / w.sum()),
        "log_evidence": float(lw_max + torch.log(sw) - np.log(len(pool))),
    }
    fn = functional_values(T, pool.t_grid)
    fmean = (w[:, None] * fn).sum(0) / w.sum()
    out["fn_mean"] = fmean.cpu().numpy()
    out["fn_sd"] = ((w[:, None] * (fn - fmean) ** 2).sum(0) / w.sum()).sqrt().cpu().numpy()
    out["fn_quantiles"] = weighted_quantiles(fn, w).cpu().numpy()
    h = pool.hyper_t[keep].double()
    hmean = (w[:, None] * h).sum(0) / w.sum()
    out["hyper_mean"] = hmean.cpu().numpy()
    out["hyper_sd"] = ((w[:, None] * (h - hmean) ** 2).sum(0) / w.sum()).sqrt().cpu().numpy()
    halves = []
    for parity in (0, 1):
        sel = (keep % 2) == parity
        if int(sel.sum()) < 2:
            halves.append(np.full(pool.n_grid, np.nan))
            continue
        wh, Th = w[sel], T[sel]
        mh = (wh[:, None] * Th).sum(0) / wh.sum()
        halves.append(((wh[:, None] * (Th - mh) ** 2).sum(0) / wh.sum()).sqrt().cpu().numpy())
    out["sd_halves"] = np.stack(halves)
    return out


@torch.no_grad()
def reference_posterior(pool: ReferencePool, query: Query, observation: ObservationModel,
                        cfg: ReferenceConfig = DEFAULT_CONFIG, *, verbose: bool = False,
                        present: tuple[float, float] | None = None) -> dict[str, Any]:
    """The exact-likelihood importance-sampling posterior of one query over the pool.

    With ``present``, a ``(lo, hi)`` present-day range (what a member given the
    present day is handed, Stage 19c), the posterior is conditioned on it: pool
    rows whose ``T_0`` lies outside carry no weight, and the evidence is over the
    rows inside.  Those rows are exact draws of ``p(T | T_0 in range)``.

    Returns the summary of :func:`posterior_summary_from_weights` plus the
    log-weights themselves (``log_w``, ``(n_pool,)``, ``-inf`` where the
    helium was never evaluated; tail rows carry their up-weighting), the
    screen's record -- ``n_he_rows`` evaluated in order, ``screen_closed``,
    ``log_dropped_bound`` (an upper bound on the log-ratio of the mass the
    ordered pass left unevaluated to the mass it evaluated; ``-inf`` when it
    reached the end), ``n_tail`` rows sampled from the remainder and
    ``tail_mass`` (the measured share of the posterior they carry; ``nan``
    when none were sampled) -- and the wall-clock of each half.
    """
    dev = pool.device
    n = len(pool)
    t0 = time.perf_counter()
    lpA = aft_log_likelihood(pool, query, observation, cfg)
    n_in = n
    if present is not None:
        lo, hi = present
        inside = (pool.T[:, 0] >= lo) & (pool.T[:, 0] <= hi)
        n_in = int(inside.sum())
        if n_in == 0:
            raise ValueError(f"no pool row has its present day in ({lo:g}, {hi:g}) °C")
        lpA = torch.where(inside, lpA, torch.full_like(lpA, -np.inf))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_aft = time.perf_counter() - t0

    t0 = time.perf_counter()
    log_w = torch.full((n,), -np.inf, dtype=torch.float32, device=dev)
    n_he_rows, closed, log_dropped, n_tail, tail_mass = 0, True, -np.inf, 0, float("nan")
    if query.n_he == 0:
        log_w[:] = lpA
    else:
        lpH_max = he_log_likelihood_max(query, observation)
        order = torch.argsort(lpA, descending=True)
        bound = (lpA[order] + lpH_max).double()                # every row's ceiling, best first
        # Suffix log-sum-exp of the ceilings: the most weight the rows from
        # ``pos`` onwards could carry between them, however their helium fits.
        suffix = torch.logcumsumexp(bound.flip(0), 0).flip(0)
        pos = 0
        while pos < n:
            if pos >= cfg.max_he_rows:
                closed = False
                break
            take = order[pos:min(pos + cfg.he_block, n, cfg.max_he_rows)]
            lpH = he_log_likelihood(pool, take, query, observation, cfg)
            log_w[take] = lpA[take] + lpH
            pos += int(take.numel())
            if pos < n:
                kept = torch.logsumexp(log_w[order[:pos]].double(), 0)
                log_dropped = float(suffix[pos] - kept)
                if log_dropped < -cfg.screen_margin:
                    break
        n_he_rows = pos
        if pos >= n:
            log_dropped = -np.inf
        elif query.has_aft and cfg.tail_sample > 0:
            rest = order[pos:]
            gen = torch.Generator(device=dev).manual_seed(int(rest.numel()))
            pick = rest[torch.randperm(rest.numel(), generator=gen, device=dev)[:cfg.tail_sample]]
            lpH = he_log_likelihood(pool, pick, query, observation, cfg)
            log_w[pick] = lpA[pick] + lpH + float(np.log(rest.numel() / pick.numel()))
            n_tail = int(pick.numel())
            head_mass = torch.logsumexp(log_w[order[:pos]].double(), 0)
            tail = torch.logsumexp(log_w[pick].double(), 0)
            tail_mass = float(torch.exp(tail - torch.logaddexp(head_mass, tail)))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_he = time.perf_counter() - t0

    out = posterior_summary_from_weights(pool, log_w, keep_margin=cfg.keep_margin)
    out["log_evidence"] += float(np.log(n / n_in))                # over the rows in the present-day range
    out.update({"log_w": log_w, "n_pool": n_in, "n_he_rows": n_he_rows, "screen_closed": closed,
                "log_dropped_bound": log_dropped, "n_tail": n_tail, "tail_mass": tail_mass,
                "time_aft": t_aft, "time_he": t_he})
    if verbose:
        tail = f"  tail {n_tail} rows carry {tail_mass:.1e}" if n_tail else ""
        print(f"    ESS {out['ess']:8.1f}  kept {out['n_kept']:6d}  He rows {n_he_rows:7d}"
              f"{'' if closed else ' (screen open)'}  bound e^{log_dropped:.1f}{tail}  "
              f"AFT {t_aft:5.1f} s  He {t_he:6.1f} s", flush=True)
    return out


@torch.no_grad()
def marginal_w1(x_ref: Tensor, w_ref: Tensor, x_other: Tensor, *, lo: float = 0.0, hi: float = 200.0,
                step: float = 0.25) -> Tensor:
    """``(k,)`` Wasserstein-1 distance per column between two marginals, in the columns' units.

    ``x_ref`` ``(n, k)`` with weights ``w_ref`` ``(n,)`` against unweighted
    draws ``x_other`` ``(m, k)``: the integral of the absolute difference of
    the two CDFs on a grid from ``lo`` to ``hi`` in steps of ``step``.  A
    width ratio says how much wider one posterior is than the other; this
    says how far apart they are, shape and location included, in kelvin.
    """
    grid = torch.arange(lo, hi + step, step, dtype=torch.float64, device=x_ref.device)
    def cdf(x: Tensor, w: Tensor | None) -> Tensor:
        idx = torch.bucketize(x.double(), grid, right=True).clamp(max=grid.numel())   # (n, k)
        out = torch.zeros((grid.numel() + 1, x.shape[1]), dtype=torch.float64, device=x.device)
        ww = torch.ones(x.shape[0], dtype=torch.float64, device=x.device) if w is None else w.double()
        out.scatter_add_(0, idx, ww[:, None].expand(-1, x.shape[1]))
        c = torch.cumsum(out, 0)[:-1]
        return c / c[-1:].clamp_min(1e-300)
    return (cdf(x_ref, w_ref) - cdf(x_other, None)).abs().sum(0) * step


#: Names of the functionals the summary carries, re-exported for report code.
FUNCTIONAL_NAMES = FUNCTIONALS
