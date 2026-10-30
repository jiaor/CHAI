"""Thermal-history path representation.

Canonical convention (see PLAN.md §2): a path is given by control points with
``t[..., 0] == 0`` at the present day, ``t`` strictly increasing into the past
in Ma, and ``T`` in degrees Celsius.

A path is defined by a *small* number of control points (~10) and is understood
to be piecewise linear between them.  The forward models integrate it on a much
finer internal grid; that refinement is a quadrature choice, not part of the
model, and lives in :func:`refine`.
"""
from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor

KELVIN_AT_0C = 273.15
SECS_PER_MA = 3.1556736e13


def refine(t: Tensor, T: Tensor, n_sub: int) -> tuple[Tensor, Tensor]:
    """Subdivide each control-point segment into ``n_sub`` equal sub-intervals.

    Linear interpolation in ``T``, so the refined path traces exactly the same
    piecewise-linear history; only the sampling density changes.  The result is
    rectangular (every path in the batch gets the same number of nodes), which
    is what makes the batched forward models possible, and it is differentiable
    with respect to both ``t`` and ``T``.

    Bounding the temperature step matters more than bounding the time step: the
    Ketcham annealing model is calibrated for ~0.5% accuracy at temperature
    steps below ~8 K (3.5 K near the total-annealing temperature).  Subdividing
    *per segment* bounds the step at ``dT_segment / n_sub``, whereas a grid that
    is uniform in time would take huge temperature steps across a short, rapid
    segment.

    Args:
        t: ``(B, N)`` times before present in Ma, increasing, ``t[:, 0] == 0``.
        T: ``(B, N)`` temperatures in degC.
        n_sub: sub-intervals per segment.

    Returns:
        ``(t_fine, T_fine)``, each ``(B, (N - 1) * n_sub + 1)``, still ordered
        present -> past and still containing every original control point.
    """
    if t.shape != T.shape:
        raise ValueError(f"t and T must have the same shape, got {tuple(t.shape)}, {tuple(T.shape)}")
    if t.ndim != 2:
        raise ValueError(f"expected (B, N) tensors, got {t.ndim}-D")
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}")

    B = t.shape[0]
    frac = torch.arange(n_sub, dtype=t.dtype, device=t.device) / n_sub  # (n_sub,)

    t0, t1 = t[:, :-1, None], t[:, 1:, None]                           # (B, N-1, 1)
    T0, T1 = T[:, :-1, None], T[:, 1:, None]
    t_fine = (t0 + (t1 - t0) * frac).reshape(B, -1)                    # (B, (N-1)*n_sub)
    T_fine = (T0 + (T1 - T0) * frac).reshape(B, -1)

    # close the path with the oldest control point
    return (
        torch.cat([t_fine, t[:, -1:]], dim=1),
        torch.cat([T_fine, T[:, -1:]], dim=1),
    )


def validate(t: Tensor, T: Tensor) -> None:
    """Assert the canonical convention: present first, strictly increasing."""
    if t.shape != T.shape:
        raise ValueError(f"t and T must have the same shape, got {tuple(t.shape)}, {tuple(T.shape)}")
    if t.shape[-1] < 2:
        raise ValueError("need at least 2 control points")
    if not torch.all(t[..., 0] == 0):
        raise ValueError("t[..., 0] must be 0 (the present day)")
    if not torch.all(torch.diff(t, dim=-1) > 0):
        raise ValueError("t must be strictly increasing into the past")


def fixed_grid(
    n: int,
    t_max: float,
    spacing: str = "log",
    t_first: float | None = None,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """The canonical fixed time grid: ``n`` nodes from 0 to ``t_max`` Ma.

    The fixed-grid decision (PLAN.md §2) makes a thermal history a
    fixed-length vector, which is what lets Goal 2 use ordinary
    fixed-dimensional generative models.  This function defines that grid.

    ``spacing="log"`` places ``t[0] = 0`` and then ``n - 1`` geometrically
    spaced nodes from ``t_first`` to ``t_max``, so resolution concentrates on
    the recent past where the observables are most sensitive.  ``t_first``
    defaults to ``t_max / 1000``.  ``spacing="uniform"`` spaces them evenly.

    ``spacing="piecewise:64@100,32@250"`` is uniform within segments: 64 nodes
    over ``[0, 100]`` (both ends included), then 32 more over ``(100, 250]``.
    The counts must sum to ``n`` and the last break must be ``t_max``.  It is
    the 250 Ma member's grid: its first 96 nodes are the 100 Ma member's
    times exactly, and the coarseness goes only where it was measured to cost
    little.  A string rather than arrays, so that a
    :class:`~chai.data.simulate.CorpusSpec` still serialises to JSON and a
    file written before there was a second spacing reads back unchanged.

    Node placement is a real choice, not a detail: it must match the prior's
    roughness in time, or the grid under-resolves wherever the prior is rough.
    See ``docs/node-count-study.md``.
    """
    if n < 2:
        raise ValueError(f"need at least 2 nodes, got {n}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    if spacing == "uniform":
        return torch.linspace(0.0, t_max, n, dtype=dtype, device=device)
    if spacing.startswith("piecewise:"):
        counts, breaks = parse_piecewise(spacing)
        if sum(counts) != n:
            raise ValueError(f"{spacing!r} has {sum(counts)} nodes, not {n}")
        if abs(breaks[-1] - t_max) > 1e-9 * t_max:
            raise ValueError(f"{spacing!r} ends at {breaks[-1]:g} Ma, not t_max = {t_max:g}")
        parts = [torch.linspace(0.0, breaks[0], counts[0], dtype=dtype, device=device)]
        for lo, hi, k in zip(breaks[:-1], breaks[1:], counts[1:]):
            # (lo, hi]: the left end is the previous segment's last node.
            parts.append(lo + (hi - lo) * torch.arange(1, k + 1, dtype=dtype, device=device) / k)
        return torch.cat(parts)
    if spacing == "log":
        # With two nodes there is no room to place anything: the grid is the
        # time span itself, and `t_first` has nowhere to go.
        if n == 2:
            return torch.tensor([0.0, t_max], dtype=dtype, device=device)
        first = t_max / 1000.0 if t_first is None else t_first
        if not 0 < first < t_max:
            raise ValueError(f"t_first must be in (0, t_max), got {first}")
        tail = torch.logspace(
            float(torch.log10(torch.tensor(first))),
            float(torch.log10(torch.tensor(t_max))),
            n - 1,
            dtype=dtype,
            device=device,
        )
        return torch.cat([torch.zeros(1, dtype=dtype, device=device), tail])
    raise ValueError(f"spacing must be 'log', 'uniform' or 'piecewise:...', got {spacing!r}")


def parse_piecewise(spacing: str) -> tuple[list[int], list[float]]:
    """``"piecewise:64@100,32@250"`` -> ``([64, 32], [100.0, 250.0])``.

    Each ``k@t`` is a segment ending at ``t`` Ma with ``k`` nodes of its own.
    The first segment starts at the present and owns both its ends, so it needs
    at least 2; later ones own their right end only, so at least 1.
    """
    body = spacing.split(":", 1)[1] if spacing.startswith("piecewise:") else spacing
    try:
        pairs = [part.split("@") for part in body.split(",")]
        counts = [int(k) for k, _ in pairs]
        breaks = [float(t) for _, t in pairs]
    except ValueError as err:
        raise ValueError(f"bad piecewise spacing {spacing!r}; expected 'piecewise:k@t,k@t,...'") from err
    if counts[0] < 2 or any(k < 1 for k in counts[1:]):
        raise ValueError(f"{spacing!r}: the first segment needs >= 2 nodes, later ones >= 1")
    if breaks[0] <= 0 or any(b <= a for a, b in pairwise(breaks)):
        raise ValueError(f"{spacing!r}: breaks must be positive and increasing")
    return counts, breaks


def piecewise_spacing(counts: list[int], breaks: list[float]) -> str:
    """The inverse of :func:`parse_piecewise`."""
    return "piecewise:" + ",".join(f"{k}@{t:g}" for k, t in zip(counts, breaks))


def resample(t: Tensor, T: Tensor, t_query: Tensor) -> Tensor:
    """Evaluate the piecewise-linear path ``(t, T)`` at ``t_query``.

    This is how a path moves between grids: a path is *defined* as the
    piecewise-linear interpolant of its control points, so resampling is exact
    wherever the query grid is a superset of the control points and lossy
    otherwise.  Queries outside ``[t[0], t[-1]]`` are clamped to the end values,
    matching how both oracles treat the ends of a path.

    Args:
        t: ``(B, N)`` control-point times, increasing, ``t[:, 0] == 0``.
        T: ``(B, N)`` control-point temperatures in degC.
        t_query: ``(M,)`` or ``(B, M)`` times to evaluate at.

    Returns:
        ``(B, M)`` temperatures.  Differentiable with respect to ``T`` and,
        away from the knots, ``t``.
    """
    if t.shape != T.shape:
        raise ValueError(f"t and T must have the same shape, got {tuple(t.shape)}, {tuple(T.shape)}")
    if t.ndim != 2:
        raise ValueError(f"expected (B, N) tensors, got {t.ndim}-D")
    if t_query.ndim == 1:
        t_query = t_query.expand(t.shape[0], -1)

    # index of the segment containing each query point, clamped at both ends
    idx = torch.searchsorted(t.contiguous(), t_query.contiguous()).clamp(1, t.shape[1] - 1)
    t0 = torch.gather(t, 1, idx - 1)
    t1 = torch.gather(t, 1, idx)
    T0 = torch.gather(T, 1, idx - 1)
    T1 = torch.gather(T, 1, idx)
    w = ((t_query - t0) / (t1 - t0)).clamp(0.0, 1.0)
    return T0 + (T1 - T0) * w
