"""Batched tridiagonal solves for the diffusion models.

The Thomas algorithm is sequential in the unknown, which is fatal here: the
RDAAM Crank-Nicolson step is *already* inside a sequential loop over time, so a
further ``rdim``-long Python loop per step would cost ~10^5 kernel launches per
call.  Parallel cyclic reduction solves the same system in ``log2(rdim)``
vectorised steps instead, which is what makes the GPU port viable at all.
"""
from __future__ import annotations

import torch
from torch import Tensor


def solve_tridiagonal(sub: Tensor, diag: Tensor, sup: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``sub_i x_{i-1} + diag_i x_i + sup_i x_{i+1} = rhs_i``, batched.

    Parallel cyclic reduction.  Every step replaces each equation with a
    combination of itself and its two neighbours two rows away, halving the
    coupling distance; after ``ceil(log2(n))`` steps the system is diagonal.
    Work is ``O(n log n)`` against Thomas's ``O(n)``, but the depth is
    ``O(log n)`` rather than ``O(n)``, and every step is one vectorised
    operation over the whole batch.

    PCR is stable for diagonally dominant systems, which the Crank-Nicolson
    operator is (``|diag| = preBeta + |sub| + |sup|`` with ``preBeta > 0``).
    It is not pivoted, so do not use it on indefinite systems.

    Args:
        sub: ``(..., n)`` sub-diagonal; ``sub[..., 0]`` is ignored.
        diag: ``(..., n)`` diagonal.
        sup: ``(..., n)`` super-diagonal; ``sup[..., -1]`` is ignored.
        rhs: ``(..., n)`` right-hand side.

    Returns:
        ``(..., n)`` solution, differentiable with respect to every input.
    """
    if not (sub.shape == diag.shape == sup.shape == rhs.shape):
        raise ValueError(
            f"all four arrays must have the same shape, got {tuple(sub.shape)}, "
            f"{tuple(diag.shape)}, {tuple(sup.shape)}, {tuple(rhs.shape)}")
    n = diag.shape[-1]
    if n == 0:
        raise ValueError("system must have at least one row")

    # Out-of-range neighbours are zero, which is exactly the identity for this
    # elimination: a missing equation contributes no coupling.
    a = torch.zeros_like(sub)
    a[..., 1:] = sub[..., 1:]
    b = diag
    c = torch.zeros_like(sup)
    c[..., :-1] = sup[..., :-1]
    d = rhs

    step = 1
    while step < n:
        # Neighbour equations `step` rows away, zero-padded at the ends.
        a_up, b_up, c_up, d_up = (_shift(x, step) for x in (a, b, c, d))
        a_dn, b_dn, c_dn, d_dn = (_shift(x, -step) for x in (a, b, c, d))

        # Eliminate x_{i-step} and x_{i+step}.  Where a neighbour is missing its
        # b is 0, so the factor is 0 and the row is carried through unchanged.
        alpha = torch.where(b_up != 0, -a / torch.where(b_up != 0, b_up, torch.ones_like(b_up)),
                            torch.zeros_like(a))
        beta = torch.where(b_dn != 0, -c / torch.where(b_dn != 0, b_dn, torch.ones_like(b_dn)),
                           torch.zeros_like(c))

        a, b, c, d = (alpha * a_up,
                      b + alpha * c_up + beta * a_dn,
                      beta * c_dn,
                      d + alpha * d_up + beta * d_dn)
        step *= 2

    return d / b


def _shift(x: Tensor, k: int) -> Tensor:
    """``x`` shifted along the last axis by ``k``, zero-filled.

    ``k > 0`` fetches the row ``k`` earlier (``out[i] = x[i-k]``).
    """
    out = torch.zeros_like(x)
    if k > 0:
        out[..., k:] = x[..., :-k]
    elif k < 0:
        out[..., :k] = x[..., -k:]
    else:
        out = x
    return out
