"""Does the deep past follow the prior's conditional?

Past the oldest age a sample records, the data say nothing directly, and
the Stage 1 audit found the posterior there is *not* the prior: the prior's
own correlation carries a well-constrained recent path backwards (the sd at
100 Ma is 55% below the prior's).  A head that returned the marginal prior out
there would be wrong; so would one that invented structure.  The testable
property is that the head's **conditional of the deep nodes given its own
recent path equals the prior's conditional** -- data enter only through the
nodes they can see.

The reference cannot be the Gaussian regression (``Latent.conditional``):
the prior is not Gaussian, and ``docs/latent.md`` measured the regression's
conditional at 56 K where a nearest-neighbour estimate gives 40 K, with the
two conditional means 28 K rms apart.  The reference here is **the corpus
itself**: for each posterior draw, the ``K`` train paths closest to it over
the recent nodes, whose deep nodes are a sample from the prior's conditional
given (approximately) that recent path.  The regression is kept as the
Gaussian bound, for contrast.

The test is then simulation-based calibration with the neighbours as the
reference posterior: the rank of each draw's deep temperature among its
neighbours' is uniform if the two conditionals agree, node by node past the
boundary.  Because a draw is never *exactly* its neighbours, the kNN
conditional is wider than the truth by the neighbours' own spread; the
report carries that spread so the reader can see how much slack the
reference has.

The boundary is **per sample** -- the first node older than the oldest age
the sample's observables record, times a margin -- not a fixed 67 Ma, which
is only the corpus-wide maximum.  Both are offered.

Two readings of the same head come out.  **Given its own recent draws**
(:func:`conditional_pits` on the head's paths) is the property itself, and a
head whose recent draws are unrealistic -- jittery, off the walls -- is
charged for it here, because the prior's conditional at an unrealistic path
is the conditional at the nearest realistic ones, which is wider.  **Given
the true recent path** (:func:`head_conditional_pits`) asks only whether the
head's *conditional* -- the Gaussian its full covariance implies for the deep
nodes given the recent ones, evaluated at the truth -- describes the true
deep past as well as the prior's conditional does.  The first is what a user
gets; the second says where a failure of the first comes from.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from chai.models.gaussian import AmortizedPosterior, draw_z

__all__ = ["boundary_nodes", "conditional_pits", "deep_past_test", "head_conditional_pits",
           "oldest_observed_age"]


def oldest_observed_age(batch: dict[str, Any]) -> np.ndarray:
    """``(n,)`` the oldest age among a sample's observables, in Ma; ``nan`` if none.

    The AFT pooled age where one was reported (``has_age``), and every dated
    helium grain's age.  Confined lengths carry information older than the
    age -- the partial annealing zone remembers -- which is what the margin
    in :func:`boundary_nodes` is for.
    """
    aft = np.asarray(batch["aft"], dtype=np.float64)
    has_age = aft[:, 3] > 0.5
    ages = np.where(has_age, np.exp(aft[:, 0]), np.nan)
    grains = np.asarray(batch["grains"], dtype=np.float64)
    mask = np.asarray(batch["grain_mask"], dtype=bool)
    he = np.where(mask, np.exp(grains[..., 0]), -np.inf).max(axis=1)
    he = np.where(mask.any(1), he, np.nan)
    return np.fmax(ages, he)


def boundary_nodes(oldest: np.ndarray, t_grid: np.ndarray, *, margin: float = 1.0
                   ) -> np.ndarray:
    """``(n,)`` index of the first node older than ``margin * oldest``; ``n_grid`` if none.

    Every node from there on is "deep": nothing the sample measured reaches
    it except through the prior.
    """
    t = np.asarray(t_grid, dtype=np.float64)
    edge = margin * np.asarray(oldest, dtype=np.float64)
    out = np.full(edge.shape, t.size, dtype=np.int64)
    ok = np.isfinite(edge)
    out[ok] = np.searchsorted(t, edge[ok], side="right")
    return out


class _TrainPaths:
    """The train paths on the device, with prefix norms for every boundary."""

    def __init__(self, T: np.ndarray, device: Any):
        self.T = torch.as_tensor(np.asarray(T, dtype=np.float32), device=device)
        self.sq_prefix = torch.cat([torch.zeros(self.T.shape[0], 1, device=device),
                                    (self.T ** 2).cumsum(1)], dim=1)     # (N, n_grid + 1)

    def neighbours(self, recent: Tensor, K: int, chunk: int = 256) -> tuple[Tensor, Tensor]:
        """``(m, K)`` train indices nearest each of ``m`` recent paths of ``b`` nodes,
        and the rms per-node distance to them."""
        b = recent.shape[1]
        train = self.T[:, :b]
        sq = self.sq_prefix[:, b]
        idx, d = [], []
        for s in range(0, recent.shape[0], chunk):
            q = recent[s:s + chunk]
            d2 = sq[None, :] - 2.0 * q @ train.T + (q ** 2).sum(1)[:, None]
            top = torch.topk(d2, K, dim=1, largest=False)
            idx.append(top.indices)
            d.append(torch.sqrt(top.values.clamp_min(0.0) / b))
        return torch.cat(idx), torch.cat(d)


@torch.no_grad()
def conditional_pits(T_draws: Tensor, bounds: np.ndarray, paths: _TrainPaths, latent: Any, *,
                     K: int = 100, gen: torch.Generator) -> dict[str, Any]:
    """The test proper, on any set of paths.

    Args:
        T_draws: ``(n, S, n_grid)`` paths in degrees on the device -- a head's
            clamped draws, or the true paths themselves with ``S = 1`` (the
            control: the truth's deep part *is* a draw from the prior's
            conditional, so whatever the control reports is the reference's
            own slack, and the head is read against it).
        bounds: ``(n,)`` boundary node per row; rows with fewer than two deep
            nodes are skipped.

    Returns arrays indexed by **offset** past the boundary ``j = node - b``, so
    rows with different boundaries line up on how far past the data a node is:
    ``offset_ma`` the mean over rows of ``t[b + j] - t[b]``, which on a uniform
    grid is ``j * dt`` and on a piecewise one depends on where ``b`` fell;
    ``pit_knn`` / ``pit_reg`` lists of ``(n_j,)`` PIT arrays; ``sd_net``,
    ``sd_knn``, ``sd_reg`` the rms over rows of the per-row sd of the draw set,
    of its pooled neighbours, and of the regression; ``shift_knn`` /
    ``shift_reg`` the rms over rows of the difference of means, in K.  Per
    boundary: ``nn_dist`` the rms per-node distance to the neighbours.
    """
    dev = T_draws.device
    n_grid = T_draws.shape[-1]
    t_grid = latent.t_grid.cpu().numpy()
    usable = bounds <= n_grid - 2
    past_ma: dict[int, list[np.ndarray]] = {}
    reg_cache: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
    acc: dict[str, dict[int, list[Tensor]]] = {k: {} for k in (
        "pit_knn", "pit_reg", "sd_net", "sd_knn", "sd_reg", "shift_knn", "shift_reg")}
    nn_dist: dict[int, float] = {}
    n_rows_at: dict[int, int] = {}

    def push(key: str, j: int, v: Tensor) -> None:
        acc[key].setdefault(j, []).append(v)

    for b in np.unique(bounds[usable]):
        b = int(b)
        sel = np.flatnonzero(usable & (bounds == b))
        n_rows_at[b] = sel.size
        S = T_draws.shape[1]
        D = n_grid - b
        # The neighbours' deep parts are (rows x S, K, D) floats, and each comparison with them a bool that the
        # count casts to int64: 13 bytes an element, so chunk the rows to keep that under ~6 GB (58 GiB of floats
        # alone at 4 096 rows x 400 draws x K = 100 in one go).  The tie-breaking uniforms are drawn for the
        # whole boundary first, so the stream, and every number, is the same however the rows are split.
        chunk = max(1, int(6e9 // (S * K * D * 13)))
        v_all = torch.rand((sel.size * S, D), generator=gen, device=dev)
        sq_sum = 0.0
        if b not in reg_cache:
            A, c, Sg = latent.path_conditional(list(range(b, n_grid)), list(range(b)))
            reg_cache[b] = (A.to(dev, torch.float32), c.to(dev, torch.float32),
                            torch.sqrt(torch.diagonal(Sg)).to(dev, torch.float32))
        A, c, s_reg = reg_cache[b]
        for lo in range(0, sel.size, chunk):
            v = v_all[lo * S:(lo + chunk) * S]
            T = T_draws[sel[lo:lo + chunk]]                                  # (m, S, n_grid)
            m = T.shape[0]
            recent = T[:, :, :b].reshape(m * S, b)
            deep = T[:, :, b:].reshape(m * S, D)
            idx, dist = paths.neighbours(recent, K)                          # (mS, K)
            nb = paths.T[idx][:, :, b:]                                      # (mS, K, D)
            sq_sum += float((dist ** 2).mean()) * m
            below = (nb < deep[:, None, :]).sum(1).float()
            equal = (nb == deep[:, None, :]).sum(1).float()
            u_knn = (below + v * (equal + 1.0)) / (K + 1.0)
            mu_reg = recent @ A.T + c
            u_reg = torch.special.ndtr((deep - mu_reg) / s_reg)
            deep_r = deep.reshape(m, S, D)
            nb_r = nb.reshape(m, S * K, D)
            mu_r = mu_reg.reshape(m, S, D)
            net_sd = deep_r.std(1) if S > 1 else torch.zeros(m, D, device=dev)
            reg_sd = torch.sqrt((mu_r.var(1) if S > 1 else torch.zeros(m, D, device=dev)) + s_reg ** 2)
            for j in range(D):
                past_ma.setdefault(j, []).append(np.full(m, t_grid[b + j] - t_grid[b]))
                push("pit_knn", j, u_knn[:, j])
                push("pit_reg", j, u_reg[:, j])
                push("sd_net", j, net_sd[:, j])
                push("sd_knn", j, nb_r[:, :, j].std(1))
                push("sd_reg", j, reg_sd[:, j])
                push("shift_knn", j, deep_r[:, :, j].mean(1) - nb_r[:, :, j].mean(1))
                push("shift_reg", j, deep_r[:, :, j].mean(1) - mu_r[:, :, j].mean(1))
            del nb, below, equal, idx, dist
        nn_dist[b] = float(np.sqrt(sq_sum / sel.size))

    offsets = sorted(acc["pit_knn"])

    def cat(key: str) -> list[np.ndarray]:
        return [torch.cat(acc[key][j]).cpu().numpy() for j in offsets]

    def rms(key: str) -> np.ndarray:
        return np.array([float(np.sqrt(np.mean(a ** 2))) for a in cat(key)])

    return {
        "offsets": np.array(offsets),
        "offset_ma": np.array([np.concatenate(past_ma[j]).mean() for j in offsets]),
        "pit_knn": cat("pit_knn"), "pit_reg": cat("pit_reg"),
        "sd_net": rms("sd_net"), "sd_knn": rms("sd_knn"), "sd_reg": rms("sd_reg"),
        "shift_knn": rms("shift_knn"), "shift_reg": rms("shift_reg"),
        "n_at_offset": np.array([a.size for a in cat("pit_knn")]),
        "boundary": bounds, "usable": usable,
        "nn_dist": np.array([nn_dist[b] for b in sorted(nn_dist)]),
        "nn_dist_boundary": np.array(sorted(nn_dist)),
        "n_rows_at_boundary": np.array([n_rows_at[b] for b in sorted(n_rows_at)]),
        "K": K,
    }


@torch.no_grad()
def head_conditional_pits(model: AmortizedPosterior, batches: list[dict[str, Tensor]],
                          bounds: np.ndarray, *, device: Any = "cpu",
                          max_rows: int | None = None) -> dict[str, Any]:
    """PIT of the true deep nodes under the head's conditional given the true recent nodes.

    The head is Gaussian in ``z`` and the map to degrees is affine, so its
    law over the nodes is Gaussian with a per-row mean and covariance, and
    its conditional of the nodes past ``b`` given the nodes before it is
    closed-form.  Returned per offset like :func:`conditional_pits`:
    ``pit_head`` and ``sd_head`` (rms over rows of the conditional sd).
    In float64, because the head's covariance in degrees spans 1e-3 K to 80 K.
    """
    model.eval()
    dev = torch.device(device)
    latent = model.latent
    n_grid = latent.n_grid
    node_ix = [latent.node_index(k) for k in range(n_grid)]
    W = latent.W_inv[node_ix].to(dev, torch.float64)                    # (n_grid, dim)
    m0 = latent.mean[node_ix].to(dev, torch.float64)
    means, covs, truths = [], [], []
    for raw in batches:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
        q = model(b)
        loc, cov = q.loc.to(torch.float64), q.covariance_matrix.to(torch.float64)
        means.append(m0 + loc @ W.T)
        covs.append(torch.einsum("kd,nde,je->nkj", W, cov, W))
        truths.append(b["T"].to(torch.float64))
    mean, cov, T = torch.cat(means), torch.cat(covs), torch.cat(truths)
    if max_rows is not None:
        mean, cov, T = mean[:max_rows], cov[:max_rows], T[:max_rows]
    bounds = np.asarray(bounds)[: len(T)]
    usable = bounds <= n_grid - 2
    pits: dict[int, list[Tensor]] = {}
    sds: dict[int, list[Tensor]] = {}
    for bnd in np.unique(bounds[usable]):
        bnd = int(bnd)
        sel = torch.as_tensor(np.flatnonzero(usable & (bounds == bnd)), device=dev)
        mu, C, Tt = mean[sel], cov[sel], T[sel]
        C_rr, C_dr, C_dd = C[:, :bnd, :bnd], C[:, bnd:, :bnd], C[:, bnd:, bnd:]
        # Gain G = C_dr C_rr^-1 through a Cholesky solve, in float64.
        L = torch.linalg.cholesky(C_rr)
        G = torch.cholesky_solve(C_dr.transpose(1, 2), L).transpose(1, 2)      # (m, D, b)
        cond_mean = mu[:, bnd:] + torch.einsum("mdb,mb->md", G, Tt[:, :bnd] - mu[:, :bnd])
        cond_var = torch.diagonal(C_dd - G @ C_dr.transpose(1, 2), dim1=1, dim2=2).clamp_min(1e-12)
        u = torch.special.ndtr((Tt[:, bnd:] - cond_mean) / cond_var.sqrt())
        for j in range(n_grid - bnd):
            pits.setdefault(j, []).append(u[:, j])
            sds.setdefault(j, []).append(cond_var[:, j].sqrt())
    offsets = sorted(pits)
    return {
        "offsets": np.array(offsets),
        "pit_head": [torch.cat(pits[j]).cpu().numpy() for j in offsets],
        "sd_head": np.array([float(torch.sqrt((torch.cat(sds[j]) ** 2).mean())) for j in offsets]),
    }


@torch.no_grad()
def deep_past_test(model: AmortizedPosterior, batches: list[dict[str, Tensor]],
                   T_train: np.ndarray, *, device: Any = "cpu", n_draws: int = 32, K: int = 100,
                   margin: float = 1.0, fixed_boundary: int | None = None, seed: int = 0,
                   max_rows: int | None = None, control: bool = True) -> dict[str, Any]:
    """The deep-past conditional test of a head over fixed observed batches.

    For every row with a boundary ``b <= n_grid - 2`` (per sample from
    :func:`boundary_nodes` with ``margin``, or ``fixed_boundary`` for all),
    ``n_draws`` clamped posterior paths are drawn and handed to
    :func:`conditional_pits`; with ``control=True`` the rows' true paths go
    through the same machinery and come back under ``"control"``, and for a
    Gaussian head its closed-form conditional at the true recent path under
    ``"head_given_truth"``.  ``dt`` is
    the first grid step, so on a uniform grid offset ``j`` is ``j * dt`` Ma past
    the boundary; ``offset_ma`` holds the times on any grid, and ``uniform``
    says which kind this was.
    """
    model.eval()
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(seed)
    latent = model.latent
    t_grid = latent.t_grid.cpu().numpy()
    paths = _TrainPaths(T_train, dev)

    draws, truths, bounds = [], [], []
    for raw in batches:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
        if fixed_boundary is not None:
            bnd = np.full(len(b["T"]), int(fixed_boundary))
        else:
            bnd = boundary_nodes(oldest_observed_age(raw), t_grid, margin=margin)
        z = draw_z(model(b), n_draws, gen)
        T, _ = model.paths(z, b)                                         # (S, n, n_grid)
        draws.append(T.transpose(0, 1).contiguous())
        truths.append(b["T"].to(T.dtype)[:, None, :])
        bounds.append(bnd)
    T_all, T_true, bounds = torch.cat(draws), torch.cat(truths), np.concatenate(bounds)
    if max_rows is not None:
        T_all, T_true, bounds = T_all[:max_rows], T_true[:max_rows], bounds[:max_rows]
    out = conditional_pits(T_all, bounds, paths, latent, K=K, gen=gen)
    steps = np.diff(t_grid)
    out.update({"dt": float(steps[0]), "uniform": bool(np.allclose(steps, steps[0])),
                "t_grid": t_grid, "n_draws": n_draws, "margin": margin,
                "fixed_boundary": fixed_boundary, "n_rows": len(bounds)})
    if control:
        out["control"] = conditional_pits(T_true, bounds, paths, latent, K=K, gen=gen)
        if model.is_gaussian:                 # closed-form conditional: Gaussian heads only
            out["head_given_truth"] = head_conditional_pits(model, batches, bounds, device=dev,
                                                            max_rows=max_rows)
    return out
