"""t-T constraint boxes, applied to posterior draws by rejection.

A HeFTy user brings geology the thermochronometers do not see -- a
deposition age that puts the rock at the surface, an intrusion that heated it
-- as boxes in time-temperature space.  Here a box is applied to the draws of
a trained head, not to the prior the head was trained under, and that is
exact rather than an approximation:

    p_box(T | x) ∝ p(T | x) · 1[T satisfies the boxes]

because a box only *removes* prior mass.  A change to ``p(T | h)`` itself
reshapes the prior everywhere and needs the corpus reweighted (which
:func:`chai.eval.sensitivity.log_weights` refuses for exactly that reason);
an indicator does not, and the draws that survive it carry the constrained
posterior over the hyperparameters with them.  What rejection *cannot* do is
get more out of the head than it put into the box's region: a box that
accepts 0.1% of the draws is answered by the head's tail, where it is least
trustworthy, and the acceptance rate is reported with every result for that
reason.

Two semantics, named separately
-------------------------------
*passes through* (:data:`PASSES`): some time in the window has a temperature
in the range -- HeFTy's usual reading, "the rock was at 60-100 °C at some
point between 40 and 30 Ma".  *stays within* (:data:`WITHIN`): every time in
the window does -- "the rock was at the surface for the whole of the
unconformity".  A box whose window is a single time is the same under both.

Tested on the path, not on its nodes
------------------------------------
A path is piecewise linear on the grid (``docs/prior.md``), so the extremes of
a path over a window are attained at the window's two ends, interpolated, or at
a node strictly inside it -- nowhere else.  :func:`window_extremes` takes that
min and max exactly; the grid spacing is 1.6 Ma (4.7 Ma past 100 Ma on the
250 Ma member's grid) and a box can be narrower than that, so testing the nodes alone would let a path through a box it misses
and reject paths that pass through one between two nodes.  Given the
extremes, and by continuity:

- passes through  ⟺  ``max ≥ T_lo`` and ``min ≤ T_hi``
- stays within    ⟺  ``min ≥ T_lo`` and ``max ≤ T_hi``

Soft boxes
----------
The same machinery with a weight in place of the indicator: ``d`` is how far
the path falls outside the box, in kelvin, by the same extremes (for *passes*,
how far its closest approach misses the range; for *within*, its worst
excursion beyond it), and the weight is ``exp(-d² / 2 soft_sd²)``.  A draw is
kept with probability equal to its weight, so the accepted draws are still an
equal-weight sample and nothing downstream changes.  ``soft_sd = 0`` is the
hard box.

Cooling only
------------
QTQt's "monotonic" option, as one more indicator on the draws: a path is kept
when it never *reheats*, that is when no younger time is hotter than an older
one.  :func:`reheating` measures the worst such rise, ``max over s < t of
T(s) - T(t)`` with ``s`` the younger time (exact on the piecewise-linear path:
the extremes are at nodes).  The test allows :data:`REHEAT_TOL` of it, because
the head was trained on paths known to 0.5 K per node (the dequantised target's
noise floor, ``docs/flow-head.md``), and that noise alone gives a flat path
2-3 K of apparent reheating: strict monotonicity kept under 1% of the draws for
samples whose true path only cooled.  Exact in the same sense as a box: an
indicator only removes prior mass.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = ["PASSES", "REHEAT_TOL", "WITHIN", "Box", "InfeasibleBoxes", "PriorEnvelope", "box_weights",
           "check_feasible", "cooling_weights", "reheating", "window_extremes"]

PASSES = "passes"
WITHIN = "within"
#: Kelvin of reheating a *cooling only* path may show.  0.5 K of independent noise per node gives a
#: flat path 3.1-3.2 K at the 99th percentile (64- and 96-node grids); 5 K (RJ, 2026-09-25) leaves room
#: above that noise floor.
REHEAT_TOL = 5.0


class InfeasibleBoxes(ValueError):
    """The boxes ask for something the prior gives zero probability, so no draw can satisfy them."""


@dataclass(frozen=True)
class Box:
    """One t-T constraint.

    Args:
        t: ``(t_young, t_old)`` window, Ma before present.  Equal ends make a
            box at a single time.
        T: ``(T_lo, T_hi)`` range, °C.
        mode: :data:`PASSES` (some time in the window is in the range) or
            :data:`WITHIN` (every time in the window is).
        soft_sd: kelvin.  Zero is a hard box; above zero a path that misses
            by ``d`` is kept with probability ``exp(-d² / 2 soft_sd²)``.
        label: shown in plots and messages.
    """

    t: tuple[float, float]
    T: tuple[float, float]
    mode: str = PASSES
    soft_sd: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        t0, t1 = (float(x) for x in self.t)
        lo, hi = (float(x) for x in self.T)
        object.__setattr__(self, "t", (t0, t1))
        object.__setattr__(self, "T", (lo, hi))
        if self.mode not in (PASSES, WITHIN):
            raise ValueError(f"mode must be {PASSES!r} or {WITHIN!r}, got {self.mode!r}")
        if not (math.isfinite(t0) and math.isfinite(t1)) or t0 > t1:
            raise ValueError(f"window must be (t_young, t_old) with t_young <= t_old, got {self.t}")
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError(f"range must be (T_lo, T_hi) with T_lo <= T_hi, got {self.T}")
        if not self.soft_sd >= 0.0:
            raise ValueError(f"soft_sd must be >= 0, got {self.soft_sd}")

    @property
    def hard(self) -> bool:
        return self.soft_sd == 0.0

    def describe(self) -> str:
        what = "passes through" if self.mode == PASSES else "stays within"
        soft = f", soft {self.soft_sd:g} K" if not self.hard else ""
        name = f"{self.label}: " if self.label else ""
        when = f"at {self.t[0]:g} Ma" if self.t[0] == self.t[1] else f"{self.t[0]:g}–{self.t[1]:g} Ma"
        return f"{name}{what} {self.T[0]:g}–{self.T[1]:g} °C {when}{soft}"


# ---------------------------------------------------------------------------
# The path over a window
# ---------------------------------------------------------------------------

def _interp(t_grid: Tensor, T: Tensor, t: float) -> Tensor:
    """``T`` ``(..., n_grid)`` linearly interpolated at time ``t``: ``(...,)``."""
    k = int(torch.searchsorted(t_grid, torch.tensor(t, dtype=t_grid.dtype, device=t_grid.device)).item())
    k = min(max(k, 1), t_grid.numel() - 1)
    t0, t1 = float(t_grid[k - 1]), float(t_grid[k])
    a = (t - t0) / (t1 - t0)
    return T[..., k - 1] * (1.0 - a) + T[..., k] * a


def window_extremes(t_grid: Tensor | np.ndarray, T: Tensor, t: tuple[float, float]) -> tuple[Tensor, Tensor]:
    """``(min, max)`` of the piecewise-linear path ``T`` ``(..., n_grid)`` over the window ``t``, exactly.

    The two window ends interpolated, and every node strictly inside.
    """
    tg = torch.as_tensor(t_grid, dtype=torch.float64, device=T.device)
    t0, t1 = t
    if t0 < float(tg[0]) - 1e-9 or t1 > float(tg[-1]) + 1e-9:
        raise ValueError(f"window {t} is outside the grid [{float(tg[0]):g}, {float(tg[-1]):g}] Ma")
    t0, t1 = max(t0, float(tg[0])), min(t1, float(tg[-1]))
    ends = torch.stack([_interp(tg, T, t0), _interp(tg, T, t1)], dim=-1)
    inside = (tg > t0) & (tg < t1)
    pts = torch.cat([ends, T[..., inside]], dim=-1) if bool(inside.any()) else ends
    return pts.amin(-1), pts.amax(-1)


def _miss(box: Box, lo: Tensor, hi: Tensor) -> Tensor:
    """How far, in kelvin, a path with window extremes ``lo, hi`` falls outside ``box``; 0 if it satisfies it."""
    b_lo, b_hi = box.T
    zero = torch.zeros_like(lo)
    if box.mode == PASSES:
        return torch.maximum(torch.maximum(b_lo - hi, lo - b_hi), zero)
    return torch.maximum(torch.maximum(b_lo - lo, hi - b_hi), zero)


def box_weights(boxes: Sequence[Box], t_grid: Tensor | np.ndarray, T: Tensor) -> Tensor:
    """``(...,)`` the product over ``boxes`` of each path's weight: 1 or 0 for a hard box, in (0, 1] for a soft one."""
    w = torch.ones(T.shape[:-1], dtype=T.dtype, device=T.device)
    for box in boxes:
        d = _miss(box, *window_extremes(t_grid, T, box.t))
        w = w * ((d <= 0).to(T.dtype) if box.hard else torch.exp(-0.5 * (d / box.soft_sd) ** 2))
    return w


def reheating(T: Tensor) -> Tensor:
    """``(...,)`` the largest rise in temperature towards the present of each path ``T`` ``(..., n_grid)``, K.

    Node 0 is the present: the largest ``T[s] - T[t]`` over ``s < t``, 0 for a path that only cools.
    """
    return (torch.cummax(T, dim=-1).values - T).amax(-1)


def cooling_weights(T: Tensor, tol: float = REHEAT_TOL) -> Tensor:
    """``(...,)`` 1 for a path that reheats by no more than ``tol`` kelvin, else 0."""
    if not tol >= 0.0:
        raise ValueError(f"reheat_tol must be >= 0, got {tol}")
    return (reheating(T) <= tol).to(T.dtype)


# ---------------------------------------------------------------------------
# What the prior can satisfy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorEnvelope:
    """Where the prior puts any mass: the temperature walls per node, the time span, the rate limit.

    Args:
        t_grid: ``(n_grid,)`` Ma.
        lo, hi: ``(n_grid,)`` the walls, °C (the present-day node's upper wall
            is the prior's ``T_present_range``, the rest ``[T_min, T_max]``).
        rate_max: the hard limit on ``|dT/dt|``, K/Ma.
    """

    t_grid: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    rate_max: float

    @property
    def t_max(self) -> float:
        return float(self.t_grid[-1])

    def walls_over(self, t: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """The walls at the window's ends and every node inside it: ``(lo, hi)``, each ``(k,)``."""
        t0, t1 = t
        inside = (self.t_grid > t0) & (self.t_grid < t1)
        at = np.concatenate([[t0, t1], self.t_grid[inside]])
        return np.interp(at, self.t_grid, self.lo), np.interp(at, self.t_grid, self.hi)


def _anchor_gap(a: Box, b: Box) -> float:
    """The most time a path can put between satisfying ``a`` and satisfying ``b``.

    A *passes* box is satisfied at one time of the path's choosing in its
    window; a *within* box at every time in it.  The rate needed to get from
    one box's range to the other's is least when the two satisfying times are
    furthest apart, so the separation that matters is the largest the
    semantics allow.
    """
    (a0, a1), (b0, b1) = a.t, b.t
    if a.mode == PASSES and b.mode == PASSES:
        return max(b1 - a0, a1 - b0)
    if a.mode == WITHIN and b.mode == WITHIN:
        return max(b0 - a1, a0 - b1, 0.0)
    p, w = (a, b) if a.mode == PASSES else (b, a)
    (p0, p1), (w0, w1) = p.t, w.t
    return max(w0 - p0, p1 - w1, 0.0)


def check_feasible(boxes: Iterable[Box], envelope: PriorEnvelope) -> None:
    """Raise :class:`InfeasibleBoxes` if the prior gives the hard boxes zero probability.

    Three ways a box set can ask for the impossible, each checked:

    1. a window outside ``[0, t_max]``;
    2. a range the walls exclude -- nowhere in the window for *passes*,
       somewhere in it for *within* (the present day's wall is 25 °C);
    3. two boxes, or a box and the present day, further apart in temperature
       than ``rate_max`` can cover in the time between them.

    Necessary conditions, not sufficient: a chain of boxes each reachable from
    the next can still be jointly out of reach, and a rate *near* the limit is
    possible but rare (the prior's ``tanh`` squashing compresses rates above
    ``0.3 · rate_max``).  Both show up as a low acceptance rate instead.  Soft
    boxes are never refused -- their weight is never zero.
    """
    hard = [b for b in boxes if b.hard]
    problems = []
    for b in hard:
        t0, t1 = b.t
        if t0 < 0.0 or t1 > envelope.t_max + 1e-9:
            problems.append(f"{b.describe()}: window outside the prior's 0–{envelope.t_max:g} Ma")
            continue
        lo, hi = envelope.walls_over(b.t)
        if b.mode == PASSES:
            ok = bool(np.any((lo <= b.T[1]) & (hi >= b.T[0])))
            where = "anywhere in the window"
        else:
            ok = bool(np.all((lo <= b.T[1]) & (hi >= b.T[0])))
            where = "throughout the window"
        if not ok:
            problems.append(f"{b.describe()}: the prior's walls ({lo.min():g}–{hi.max():g} °C, "
                            f"{envelope.hi[0]:g} °C at present) exclude the range {where}")
    present = Box((0.0, 0.0), (float(envelope.lo[0]), float(envelope.hi[0])), WITHIN, label="present day")
    pairs = [(present, b) for b in hard] + [(a, b) for i, a in enumerate(hard) for b in hard[i + 1:]]
    for a, b in pairs:
        gap_T = max(b.T[0] - a.T[1], a.T[0] - b.T[1], 0.0)
        if gap_T == 0.0:
            continue
        gap_t = _anchor_gap(a, b)
        if gap_t <= 0.0 or gap_T / gap_t > envelope.rate_max:
            need = "any time between them" if gap_t <= 0.0 else f"{gap_T / gap_t:.1f} K/Ma"
            problems.append(f"{a.describe()} → {b.describe()}: {gap_T:g} K apart needs {need}, "
                            f"above the prior's rate limit of {envelope.rate_max:g} K/Ma")
    if problems:
        raise InfeasibleBoxes("the prior gives these boxes zero probability:\n  " + "\n  ".join(problems))
