"""What a trained posterior gets right, per node.

Stage 12 owns calibration; this module is the part of it Stage 11 needs to
know whether a head learned anything, written so Stage 12 extends it rather
than replaces it.  Everything is reported **per node**, because a mean over
64 nodes is exactly what hid the helium signal in the Stage 1 audit, and
**per modality** -- what the sample actually had measured -- because the
point of observable dropout is to be right on incomplete samples, and an
average over complete ones cannot show that.

Three views of the same posterior:

* **In ``z``**, the standardised residual ``L^-1 (z_true - mu)``.  Under a
  calibrated Gaussian head it is ``N(0, I)`` coordinate by coordinate, so its
  variance and its 68% / 95% coverage are the whole of simulation-based
  calibration for this head (the rank of the truth among draws from ``q`` is
  its normal CDF).  Cheap enough to run every validation step.
* **In degrees**, from draws inverted through the latent and clamped to the
  box: the posterior sd at each node against the prior's, the rms error of the
  posterior mean, and the empirical coverage of the truth by the central 68%
  and 95% of the draws.  This is what a user of the model sees, and clamping
  is part of what they see, so it is measured with it.
* **The NLL against the prior baseline**, in nats per row: the head at
  initialisation scores ``0.5 (|z|^2 + d log 2 pi)``, and the drop from there
  is the information the data bought.

Stage 12 adds the test that does not assume the head's family: **simulation-
based calibration** (Talts et al. 2018).  Every corpus row is a draw
``theta ~ p(theta)``, its observation a draw ``x ~ p(x | theta)``, so if the
head is the posterior then the rank of ``theta`` among ``L`` draws from
``q(theta | x)`` is uniform on ``{0, ..., L}`` -- for every node, every
derived quantity, every stratum of samples.  :func:`sbc` returns those ranks
as **probability-integral-transform values** ``u`` in ``[0, 1]``, randomised
across ties so that a posterior with atoms (the clamped walls) is tested
fairly; :func:`pit_summary` turns a column of them into the numbers a
criterion can be written against.  The ranks in degrees are what a user
sees; the ranks in ``z`` are the exact normal CDF of the marginal residual
and need no draws, so the two agree wherever clamping did nothing.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from chai.models.gaussian import AmortizedPosterior, draw_z, residual_of

__all__ = ["FUNCTIONALS", "LEVELS", "MODALITIES", "PRESENT_KEYS", "coverage_from_pit",
           "draw_given_present", "functional_values", "modality_of", "pit_summary",
           "posterior_summary", "present_ranges", "sbc", "steepest_rate", "strata_of"]

#: Names of the modality classes :func:`modality_of` assigns.
MODALITIES = ("full", "no_lengths", "aft_only", "he_only")


def modality_of(effort: dict[str, Tensor]) -> np.ndarray:
    """``(n,)`` index into :data:`MODALITIES` from a batch's drawn effort.

    ``full`` has an AFT age, lengths and helium grains; ``no_lengths`` has age
    and grains but no lengths; ``aft_only`` has no grains (with or without
    lengths); ``he_only`` has no fission-track analysis.
    """
    aft = np.asarray(effort["aft_present"], dtype=bool)
    n_len = np.asarray(effort["n_lengths"])
    n_he = np.asarray(effort["n_he_grains"])
    out = np.full(aft.shape, MODALITIES.index("full"), dtype=np.int64)
    out[aft & (n_he > 0) & (n_len == 0)] = MODALITIES.index("no_lengths")
    out[aft & (n_he == 0)] = MODALITIES.index("aft_only")
    out[~aft] = MODALITIES.index("he_only")
    return out


def _to(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def posterior_summary(model: AmortizedPosterior, batches: list[dict[str, Tensor]], *,
                      device: Any = "cpu", n_samples: int = 64, prior_sd: np.ndarray | None = None
                      ) -> dict[str, Any]:
    """Per-node and per-modality diagnostics over some fixed batches.

    Returns a dict of arrays.  Keyed by metric, each ``(n_modalities + 1, ...)``
    with the last row the pool of all modalities:

    ``n`` rows; ``nll`` and ``nll_prior`` (nats/row); ``resid_var``,
    ``resid_cov68``, ``resid_cov95`` ``(..., dim)`` in ``z``; ``post_sd``,
    ``rmse_mean``, ``cov68``, ``cov95`` ``(..., n_grid)`` in degrees;
    ``prior_sd`` ``(n_grid,)`` if given.  Plus ``t_grid``.
    """
    model.eval()
    M = len(MODALITIES)
    dim, n_grid = model.dim, model.latent.n_grid
    acc: dict[str, np.ndarray] = {
        "n": np.zeros(M), "nll": np.zeros(M), "nll_prior": np.zeros(M),
        "resid_sq": np.zeros((M, dim)), "resid_in68": np.zeros((M, dim)),
        "resid_in95": np.zeros((M, dim)),
        "post_var": np.zeros((M, n_grid)), "sq_err": np.zeros((M, n_grid)),
        "in68": np.zeros((M, n_grid)), "in95": np.zeros((M, n_grid)),
    }
    for raw in batches:
        b = _to(raw, device)
        mod = modality_of(raw["effort"]) if "effort" in raw else np.zeros(len(raw["z"]), int)
        q = model(b)
        nll = -q.log_prob(b["z"])
        nll0 = model.prior_nll(b["z"])
        r = residual_of(q, b["z"])
        z = draw_z(q, n_samples)
        T, _ = model.paths(z, b)                                      # (S, n, n_grid)
        T_true = b["T"].to(T.dtype)
        lo68, hi68 = torch.quantile(T, torch.tensor([0.16, 0.84], device=T.device, dtype=T.dtype), dim=0)
        lo95, hi95 = torch.quantile(T, torch.tensor([0.025, 0.975], device=T.device, dtype=T.dtype), dim=0)
        rows = {
            "nll": nll, "nll_prior": nll0,
            "resid_sq": r.pow(2), "resid_in68": (r.abs() < 1.0).to(r.dtype),
            "resid_in95": (r.abs() < 1.959964).to(r.dtype),
            "post_var": T.var(0), "sq_err": (T.mean(0) - T_true).pow(2),
            "in68": ((T_true >= lo68) & (T_true <= hi68)).to(T.dtype),
            "in95": ((T_true >= lo95) & (T_true <= hi95)).to(T.dtype),
        }
        mod_t = torch.as_tensor(mod, device=device)
        for m in range(M):
            sel = mod_t == m
            k = int(sel.sum())
            if k == 0:
                continue
            acc["n"][m] += k
            for key, v in rows.items():
                acc[key][m] += v[sel].sum(0).double().cpu().numpy()

    def with_pool(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a, a.sum(0, keepdims=True)], axis=0)

    n = with_pool(acc["n"])
    safe = np.maximum(n, 1)
    def mean(key: str) -> np.ndarray:
        a = with_pool(acc[key])
        return a / safe.reshape(-1, *([1] * (a.ndim - 1)))

    out: dict[str, Any] = {
        "modalities": np.array(MODALITIES + ("all",)),
        "n": n, "nll": mean("nll"), "nll_prior": mean("nll_prior"),
        "resid_var": mean("resid_sq"), "resid_cov68": mean("resid_in68"),
        "resid_cov95": mean("resid_in95"),
        "post_sd": np.sqrt(mean("post_var")), "rmse_mean": np.sqrt(mean("sq_err")),
        "cov68": mean("in68"), "cov95": mean("in95"),
        "t_grid": model.latent.t_grid.cpu().numpy(),
    }
    if prior_sd is not None:
        out["prior_sd"] = np.asarray(prior_sd, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Stage 12: simulation-based calibration
# ---------------------------------------------------------------------------

#: Central credible levels every coverage table reports.
LEVELS = (0.50, 0.68, 0.90, 0.95, 0.99)


def _nearest_node(t_grid: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t_grid) - t)))


def functional_values(T: Tensor, t_grid: np.ndarray, *, data_edge: float = 67.0) -> Tensor:
    """Derived quantities of a path, ``(..., n_grid) -> (..., n_fn)``, per :data:`FUNCTIONALS`.

    Each one mixes several nodes, so its rank tests the head's *correlations*
    -- which a per-node rank cannot, and which the Stage 1 audit's finding 3
    asked for.  Present -> past along the last axis.

    The last is a rate, ``|dT| / dt`` per segment, so it means the same on a
    piecewise grid as on a uniform one.  On a uniform grid it is the old
    "largest step (K/node)" divided by the spacing, a positive constant, so
    every rank and PIT computed from it is unchanged.
    """
    t = np.asarray(t_grid)
    k10, k30, k60 = (_nearest_node(t, x) for x in (10.0, 30.0, 60.0))
    k_edge = _nearest_node(t, data_edge)
    k40 = _nearest_node(t, 40.0)
    dt = torch.as_tensor(np.diff(t), dtype=T.dtype, device=T.device)
    return torch.stack([
        T[..., k30] - T[..., k10],
        T[..., k60] - T[..., k30],
        T[..., : k_edge + 1].amax(-1),
        T[..., : k40 + 1].mean(-1),
        ((T[..., 1:] - T[..., :-1]).abs() / dt).amax(-1),
    ], dim=-1)


#: Names of what :func:`functional_values` returns, in order.
FUNCTIONALS = ("T30 - T10", "T60 - T30", "max T to 67 Ma", "mean T to 40 Ma",
               "largest rate (K/Ma)")


def steepest_rate(T: Any, t_grid: np.ndarray) -> Any:
    """The steepest grid segment of each path, ``|dT| / dt`` in K/Ma, over the last axis."""
    dt = np.diff(np.asarray(t_grid, dtype=np.float64))
    if torch.is_tensor(T):
        return (T.diff(dim=-1).abs() / torch.as_tensor(dt, dtype=T.dtype, device=T.device)).amax(-1)
    return (np.abs(np.diff(np.asarray(T, dtype=np.float64), axis=-1)) / dt).max(-1)


def strata_of(batch: dict[str, Any], t_grid: np.ndarray, box: tuple[np.ndarray, np.ndarray], *,
              aft_true: np.ndarray | None = None, he_true: np.ndarray | None = None,
              horizon_age: float | None = None, data_edge: float = 67.0,
              old_aft: float = 100.0, young_he: float = 50.0,
              fast_rate: float = 50.0, p_fast: np.ndarray | None = None,
              fast_p: float = 0.1) -> dict[str, np.ndarray]:
    """Boolean masks over a batch's rows for the stratified tables.

    ``box`` is the prior's ``(lo, hi)`` per node, from the latent.
    ``full`` / ``no_lengths`` / ``aft_only`` / ``he_only`` from the drawn
    effort; ``near_horizon`` (true AFT age above ``horizon_age`` -- the
    pile-up the plan asks for, given ``aft_true``; 97% of the time span by
    default, which is 97 Ma on the 100 Ma member and 242.5 on the 250);
    ``touches_wall`` (the true path sits exactly on a wall somewhere in the
    data window, which is where the Gaussian head was predicted to fail first)
    and its complement.  Given ``he_true`` too (the true corrected helium age
    per row, the median over its grains), ``late_exhumation``: AFT older than
    ``old_aft`` with helium younger than ``young_he``, long residence then a
    late exit, which the 250 Ma member's grid was sized for.
    Given ``p_fast`` (per row, the share of the head's own draws whose
    steepest grid segment is faster than ``fast_rate`` K/Ma), ``fast``: rows
    with ``p_fast >= fast_p``, the tail the prior's 100 K/Ma cap opens.  It is
    chosen from the data, as a calibration stratum must be;
    ``fast_true`` (the true path's steepest segment is faster) is a diagnostic
    only: selecting on the truth picks the rows an exact posterior shrinks
    away from, so it under-covers there too.
    """
    if horizon_age is None:
        horizon_age = 0.97 * float(np.asarray(t_grid)[-1])
    n = len(batch["T"])
    out: dict[str, np.ndarray] = {}
    if "effort" in batch:
        mod = modality_of(batch["effort"])
        for i, m in enumerate(MODALITIES):
            out[m] = mod == i
    T = np.asarray(batch["T"], dtype=np.float64)
    k_edge = _nearest_node(t_grid, data_edge)
    lo = np.asarray(box[0], dtype=np.float64)[: k_edge + 1]
    hi = np.asarray(box[1], dtype=np.float64)[: k_edge + 1]
    window = T[:, : k_edge + 1]
    wall = ((window <= lo) | (window >= hi)).any(1)
    out["touches_wall"] = wall
    out["off_walls"] = ~wall
    out["fast_true"] = steepest_rate(T, t_grid) > fast_rate
    if p_fast is not None:
        out["fast"] = np.asarray(p_fast) >= fast_p
    if aft_true is not None:
        out["near_horizon"] = np.asarray(aft_true) > horizon_age
        out["below_horizon"] = ~out["near_horizon"]
        if he_true is not None:
            out["late_exhumation"] = (np.asarray(aft_true) > old_aft) & (np.asarray(he_true) < young_he)
    out["all"] = np.ones(n, dtype=bool)
    return out


def _pit_from_draws(draws: Tensor, truth: Tensor, gen: torch.Generator) -> Tensor:
    """Randomised PIT of ``truth`` ``(n, k)`` among ``draws`` ``(S, n, k)``.

    ``(below + V (equal + 1)) / (S + 1)`` with ``V ~ U(0, 1)``: uniform under a
    calibrated posterior whether or not there are ties, and the truth counts
    as one more member of a tie -- which is what an atom the head reproduces
    by clamping deserves.
    """
    S = draws.shape[0]
    below = (draws < truth).sum(0).to(torch.float32)
    equal = (draws == truth).sum(0).to(torch.float32)
    v = torch.rand(truth.shape, generator=gen, device=truth.device)
    return (below + v * (equal + 1.0)) / (S + 1.0)


# ---------------------------------------------------------------------------
# The present day as the user gives it: a range
# ---------------------------------------------------------------------------

#: The batch entries the encoder reads, besides ``T0``: what a draw's own present day is added to.
PRESENT_KEYS = ("aft", "length_pdf", "grains", "grain_mask")


def present_ranges(T0: Tensor, width: float, lo: float, hi: float, gen: torch.Generator) -> Tensor:
    """``(n, 2)`` present-day ranges a user could have given for rows whose truth is ``T0``.

    The range is ``[a, a + w]`` with ``a = T0 - U w``, cut to the prior's
    ``[lo, hi]``.  Under the prior's uniform ``T_0`` this makes the truth
    uniform on the range it is handed, cut or not: given ``a``, ``T_0`` is
    uniform on ``[a, a + w]`` intersected with the prior's support.  So a
    row's range says "somewhere in here" and nothing more -- the question a
    user asks -- and SBC over these rows is exact for it.  Near the edges a
    range is narrower than ``w``, as a user's would be.
    """
    T0 = T0.to(torch.float64)
    a = T0 - width * torch.rand(T0.shape, generator=gen, device=T0.device, dtype=torch.float64)
    return torch.stack([a.clamp_min(lo), (a + width).clamp_max(hi)], dim=-1)


@torch.no_grad()
def draw_given_present(model: AmortizedPosterior, batch: dict[str, Tensor], n: int, ranges: Tensor,
                       gen: torch.Generator, *, max_rows: int = 65_536) -> tuple[Tensor, Tensor]:
    """``(z, T0)``, ``(n, rows, dim)`` and ``(n, rows)``: draws with their own present day.

    What :meth:`chai.api.Model.infer` does for one sample, for a batch of
    them: each draw's ``T_0`` is uniform on its row's range (``(rows, 2)``,
    from :func:`present_ranges`), the head is run once per draw on the row's
    data and that ``T_0``, and one ``z`` is drawn.  Hand both to
    ``model.latent.inverse(z, T0=T0)``.  ``max_rows`` bounds the rows per
    forward pass (draws times batch rows).
    """
    rows = ranges.shape[0]
    lo, hi = ranges[:, 0], ranges[:, 1]
    T0 = lo + (hi - lo) * torch.rand((n, rows), generator=gen, device=ranges.device, dtype=torch.float64)
    per = max(1, max_rows // rows)
    zs = []
    for s in range(0, n, per):
        k = min(per, n - s)
        feats = {key: batch[key].repeat(k, *([1] * (batch[key].dim() - 1))) for key in PRESENT_KEYS}
        feats["T0"] = T0[s:s + k].reshape(-1).to(torch.float32)
        zs.append(draw_z(model(feats), 1, gen)[0].reshape(k, rows, -1))
    return torch.cat(zs), T0


@torch.no_grad()
def sbc(model: AmortizedPosterior, batches: list[dict[str, Tensor]], *, device: Any = "cpu",
        n_draws: int = 100, seed: int = 0, aft_true: np.ndarray | None = None,
        he_true: np.ndarray | None = None, data_edge: float = 67.0,
        present_width: float | None = None) -> dict[str, Any]:
    """Simulation-based calibration of a head over fixed observed batches.

    Args:
        batches: as :func:`chai.train.data.fixed_batches` returns them, with
            ``z``, ``T`` and ``effort``.
        n_draws: posterior draws per row (Talts et al. use ~100).
        aft_true: the true AFT age per row, in the batches' order, for the
            time span stratum; optional.
        he_true: the true helium age per row (median over grains), with
            ``aft_true`` for the late-exhumation stratum; optional.
        data_edge: Ma, where the functionals' "data window" ends
            (:func:`functional_values`) and the wall stratum looks.
        present_width: K, for a head given the present day
            (``Latent.present_input``) only.  Positive: each row is handed a
            range this wide around its true ``T_0`` (:func:`present_ranges`)
            and each draw its own ``T_0`` on it (:func:`draw_given_present`),
            as :meth:`chai.api.Model.infer` does -- the calibration of what a
            user gets, node 0 included.  ``None`` or 0: the true ``T_0``
            exactly, the calibration of the head's conditional
            ``q(T_1.. | x, T_0)``; node 0 is then the input, not a draw.

    Returns per-row arrays, all ``(n, ...)`` in the batches' order:
    ``pit_T`` ``(n, n_grid)`` PIT of the true path per node in degrees, from
    clamped draws; ``pit_T_exact`` the same without clamping -- for the
    Gaussian head as the exact normal CDF under its marginal at that node
    (the head is Gaussian in ``z`` and the map to degrees is affine), for a
    flow as the randomised PIT among its unclamped draws -- so the gap
    between the two is what clamping the leak does; ``pit_z`` ``(n, dim)``
    the marginal PIT of the truth in ``z`` (hyperparameters first), exact for
    the Gaussian and from draws otherwise; ``pit_fn`` ``(n, n_fn)`` for
    :data:`FUNCTIONALS`; ``nll`` per row; ``in68`` / ``in95`` ``(n, n_grid)``
    whether the truth lies in the central *interval* of the clamped draws --
    Stage 11's statistic, kept so the two can be read side by side: an
    interval whose end lands on an atom holds the whole atom, so it
    over-counts where the PIT does not; ``strata`` a dict of masks;
    ``p_fast`` per row, the share of the clamped draws with a segment faster
    than 50 K/Ma (what the ``fast`` stratum is cut from); plus
    ``t_grid``, ``n_draws``, ``hyper_names``, and ``first_node``, the node the
    per-node arrays start at (1 for a head given ``T_0`` exactly, else 0).
    ``nll`` and ``pit_z`` are always the head given the true ``T_0``: ``z``
    is the excess over it, so a draw made at another ``T_0`` is not in the
    same coordinates.  With a range, ``present_range`` ``(n, 2)`` too.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    t_grid = model.latent.t_grid.cpu().numpy()
    box = (model.latent.box_lo.cpu().numpy(), model.latent.box_hi.cpu().numpy())
    pits_T, pits_Tx, pits_z, pits_fn, nlls, strata_rows, p_fasts = [], [], [], [], [], [], []
    in68, in95 = [], []
    given = model.latent.present_input
    if not given and present_width:
        raise ValueError("present_width is for a head given the present day; this one draws it")
    with_range = bool(given and present_width)
    if with_range and present_width < 0:
        raise ValueError(f"present_width must be positive, got {present_width}")
    # Given T_0 exactly, the head is scored on the nodes it draws: node 0 is the input.
    first = int(given and not with_range)
    p_lo, p_hi = float(model.latent.box_lo[0]), float(model.latent.box_hi[0])
    ranges_all = []
    if model.is_gaussian:
        if given:
            raise NotImplementedError("the closed-form Gaussian marginals assume the latent carries every node")
        # The affine part of z -> T: rows of W_inv for the node coordinates.
        W_nodes = model.latent.W_inv[[model.latent.node_index(k) for k in range(t_grid.size)]]
        W_nodes = W_nodes.to(device, torch.float32)                      # (n_grid, dim)
        mean_nodes = model.latent.mean[[model.latent.node_index(k) for k in range(t_grid.size)]]
        mean_nodes = mean_nodes.to(device, torch.float32)
    offset = 0
    for raw in batches:
        b = _to(raw, device)
        n = len(b["z"])
        q = model(b)
        nlls.append((-q.log_prob(b["z"])).cpu().numpy())
        z = draw_z(q, n_draws, gen)
        T_true = b["T"].to(torch.float32)
        if with_range:
            ranges = present_ranges(b["T0"], present_width, p_lo, p_hi, gen)
            ranges_all.append(ranges.cpu().numpy())
            pits_z.append(_pit_from_draws(z, b["z"], gen).cpu().numpy())
            z, T0 = draw_given_present(model, b, n_draws, ranges, gen)
            b = {**b, "T0": T0}                           # (S, n): one present day per draw
            T_free, _ = model.paths(z, b, clamp=False)
            pits_Tx.append(_pit_from_draws(T_free.to(torch.float32), T_true, gen).cpu().numpy())
        elif model.is_gaussian:
            # exact in z: the marginal CDF of each coordinate under the head
            sd = torch.sqrt(torch.diagonal(q.covariance_matrix, dim1=-2, dim2=-1))
            pits_z.append(torch.special.ndtr((b["z"] - q.loc) / sd).cpu().numpy())
            # in degrees, exactly, before clamping: T = mean + W_inv z is Gaussian per node
            mu_T = mean_nodes + q.loc @ W_nodes.T
            var_T = torch.einsum("kd,nde,ke->nk", W_nodes, q.covariance_matrix, W_nodes)
            pits_Tx.append(torch.special.ndtr((T_true - mu_T) / var_T.sqrt()).cpu().numpy())
        else:
            # no closed form: the randomised PIT among the head's own (continuous) draws
            pits_z.append(_pit_from_draws(z, b["z"], gen).cpu().numpy())
            T_free, _ = model.paths(z, b, clamp=False)
            pits_Tx.append(_pit_from_draws(T_free[..., first:].to(torch.float32), T_true[:, first:],
                                           gen).cpu().numpy())
        # in degrees, from clamped draws (what a user sees)
        T, _ = model.paths(z, b)
        T_true = T_true.to(T.dtype)
        pits_T.append(_pit_from_draws(T[..., first:], T_true[:, first:], gen).cpu().numpy())
        qs = torch.quantile(T[..., first:], torch.tensor([0.025, 0.16, 0.84, 0.975], device=T.device,
                                                         dtype=T.dtype), dim=0)
        Tt = T_true[:, first:]
        in68.append(((Tt >= qs[1]) & (Tt <= qs[2])).cpu().numpy())
        in95.append(((Tt >= qs[0]) & (Tt <= qs[3])).cpu().numpy())
        pits_fn.append(_pit_from_draws(functional_values(T, t_grid, data_edge=data_edge),
                                       functional_values(T_true, t_grid, data_edge=data_edge),
                                       gen).cpu().numpy())
        at = None if aft_true is None else np.asarray(aft_true)[offset:offset + n]
        ht = None if he_true is None else np.asarray(he_true)[offset:offset + n]
        # the head's own probability of a fast segment, over every node of its clamped draws
        # (in chunks of draws: the full diff is another (S, n, n_grid) tensor, and SBC runs near the card's memory)
        p_fast = sum((steepest_rate(T[i:i + 100], t_grid) > 50.0).sum(0) for i in range(0, len(T), 100))
        p_fast = (p_fast / len(T)).cpu().numpy()
        p_fasts.append(p_fast)
        strata_rows.append(strata_of(raw, t_grid, box, aft_true=at, he_true=ht, data_edge=data_edge,
                                     p_fast=p_fast))
        offset += n
    strata = {k: np.concatenate([s[k] for s in strata_rows]) for k in strata_rows[0]}
    return {
        "pit_T": np.concatenate(pits_T), "pit_T_exact": np.concatenate(pits_Tx),
        "pit_z": np.concatenate(pits_z),
        "pit_fn": np.concatenate(pits_fn), "nll": np.concatenate(nlls),
        "in68": np.concatenate(in68), "in95": np.concatenate(in95),
        "strata": strata, "p_fast": np.concatenate(p_fasts),
        "t_grid": t_grid[first:], "n_draws": n_draws, "first_node": first,
        "hyper_names": list(model.latent.hyper_names), "functionals": list(FUNCTIONALS),
        **({"present_range": np.concatenate(ranges_all)} if with_range else {}),
    }


def coverage_from_pit(u: np.ndarray, levels: tuple[float, ...] = LEVELS) -> np.ndarray:
    """Empirical coverage of the central ``levels`` intervals, ``(n_levels, ...)``.

    The truth is inside the central ``a`` interval exactly when its PIT is in
    ``[(1 - a) / 2, (1 + a) / 2]``, so coverage is a property of the ranks.
    """
    u = np.asarray(u)
    return np.stack([((u >= (1 - a) / 2) & (u <= (1 + a) / 2)).mean(0) for a in levels])


def pit_summary(u: np.ndarray, *, n_bins: int = 20, levels: tuple[float, ...] = LEVELS
                ) -> dict[str, np.ndarray]:
    """What a column of PIT values says, per coordinate.  ``u`` is ``(n, k)``.

    Returns, each ``(k,)`` unless stated:

    * ``coverage`` ``(n_levels, k)`` at :data:`LEVELS`;
    * ``ecdf_dev``: the largest deviation of the empirical CDF from the
      uniform's -- a Kolmogorov distance, in probability units, so 0.02 means
      "no interval is off by more than two points";
    * ``chi2`` over ``n_bins`` equal bins, and ``chi2_p`` its p-value with
      ``n_bins - 1`` degrees of freedom -- reported, and not used as the
      criterion, because at 50 000 rows it rejects any real model;
    * ``centre_excess``: fraction in the middle half minus 0.5.  Positive
      means the truth sits nearer the centre than the head expects (a head
      too wide in the bulk); negative, a head too narrow;
    * ``tail_excess``: fraction in the outer 5% (2.5% each side) minus 0.05.
      Positive means tails too light -- the failure importance sampling
      cannot recover from;
    * ``bias``: mean PIT minus 0.5.  Positive means the truth is usually
      above the head's median (in degrees: the head runs cold).
    """
    from scipy.stats import chi2 as _chi2
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    n, k = u.shape
    srt = np.sort(u, axis=0)
    grid = (np.arange(1, n + 1) / n)[:, None]
    ecdf_dev = np.maximum(np.abs(srt - grid), np.abs(srt - (grid - 1.0 / n))).max(0)
    bins = np.minimum((u * n_bins).astype(np.int64), n_bins - 1)
    counts = np.stack([np.bincount(bins[:, j], minlength=n_bins) for j in range(k)], axis=1)
    expected = n / n_bins
    chi2 = ((counts - expected) ** 2 / expected).sum(0)
    return {
        "coverage": coverage_from_pit(u, levels),
        "ecdf_dev": ecdf_dev,
        "chi2": chi2, "chi2_p": _chi2.sf(chi2, n_bins - 1),
        "centre_excess": ((u >= 0.25) & (u <= 0.75)).mean(0) - 0.5,
        "tail_excess": ((u < 0.025) | (u > 0.975)).mean(0) - 0.05,
        "bias": u.mean(0) - 0.5,
        "histogram": counts,
        "n": np.full(k, n),
    }
