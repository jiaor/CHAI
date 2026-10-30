"""Exact, batched, differentiable apatite (U-Th)/He model (RDAAM) in PyTorch.

A direct port of ``native/rdaam/RDAAM.cpp`` (Flowers et al. 2009) --
``RDAAM_PrepModel`` -> ``RDAAM_InitFTAnnealingTraps`` -> ``RDAAM_CalcHeAge`` ->
the age equation -- not a learned approximation.  Validated against that C++ in
``tests/test_torch_rdaam.py``.

The model is radial He diffusion in a sphere with a damage-dependent
diffusivity: fission tracks accumulate, anneal, and trap He.  Three nested
computations, in increasing cost:

1. **Grain setup** (no path dependence): radial grid, parent concentrations,
   alpha-ejection factors, and the Ft corrections.
2. **Annealing traps**: for every time step, the summed damage from every
   *earlier* cohort of tracks, each annealed to that moment.  This is the
   ``O(N^2)`` part and the reason the C's cost grows super-linearly in path
   length.  Here the outer loop over observation times runs in parallel and the
   inner annealing chain is the sequential axis, so the depth is ``O(N)``.
3. **Crank-Nicolson**: one tridiagonal solve per time step, over the radial
   grid.  Sequential in time, and solved with parallel cyclic reduction rather
   than the Thomas algorithm so the radial axis does not add depth.

Departures from the C, all documented in PLAN.md:

- **Internal grid.** ``RDAAM_InterpolateTTPath`` subdivides adaptively to bound
  the temperature step; that is data-dependent control flow, so we use
  :func:`chai.paths.refine` and verify convergence in ``n_sub`` instead.
- **No early exit.** Cohorts that anneal away are latched to zero rather than
  breaking the loop.
- **Age by Newton, not bisection.** The C bisects to a 100-year tolerance, which
  is both non-differentiable and a 5e-5 Ma quantisation of its own answer.
- **``annealingTraps`` at the first node.** The C reads that row before writing
  it (``dmatrix`` uses ``malloc``, not ``calloc``), so its value is whatever the
  allocator returned.  Zero -- no damage before the path starts -- is the
  intended value and is used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ...paths import refine
from .tridiag import solve_tridiagonal

# --- constants, all from native/rdaam/RDAAM.h -----------------------------
# Note this is NOT the value ketch.h uses (3.1556736e13); each C file has its
# own, and each port keeps its own file's.
SECS_PER_MA = 3.15569259747e13
SECS_PER_YR = 3.15569259747e7
KELVIN_AT_0C = 273.15
PI = 3.1415926535            # as truncated in RDAAM.h, which rounds the answer
UNIV_GAS_CONST = 1.987       # cal/K/mol

U238YR, U235YR, TH232YR, SM147YR = 1.55125e-10, 9.8485e-10, 4.9475e-11, 6.54e-12
U238SEC, U235SEC = 4.91575e-18, 3.12089e-17
TH232SEC, SM147SEC = 1.56781e-18, 2.072e-19
U238YR_SF = 8.46e-17

PPM_NMPG_U238, PPM_NMPG_U235 = 4.20168, 4.25532
PPM_NMPG_TH232, PPM_NMPG_SM147 = 4.31034, 6.80272
U238U235RATIO = 137.88
N_PER_NMOL = 6.02214199e14
DENSITY_APATITE = 3.19

#: He atoms produced per decay chain, in the C's isotope order.
YIELD = (8.0, 7.0, 6.0, 1.0)

# Alpha stopping distances in apatite, um (Ketcham et al. 2011)
AS_DIST = (18.81, 21.80, 22.25, 5.93)

# RDAAM (apatite) diffusion and trapping parameters, from RDAAM_Init
E_ACT = 29.23        # activation energy, kcal/mol
D_INF = 0.6071       # diffusivity at infinite temperature, cm^2/s
E_TRAP = 8.126       # trapping activation energy, kcal/mol
PSI = 1.0e-13
POLY_A = 1.0e-22
TRAP_RMR0 = 0.83

# Ketcham et al. (2007) annealing, as RDAAM_InitFTAnnealingTraps uses it.
# Same equations as torchimpl.ketcham's 2007 branch, but its own rmr0 (0.83),
# its own kappa offset, and an observational-bias threshold of 0.757 where
# KETCH.c uses 0.765.
AN_C0, AN_C1, AN_C2, AN_C3, AN_A = 0.39528, 0.01073, -65.12969, -7.91715, 0.04672
TRAP_TOT_ANNEAL_LEN = 0.55
ETA_Q = 0.91
ETCHABLE_RANGE = 0.000815    # cm, one fission fragment

#: Damage per unit He, converting the trap sum into the C's units.
DENSITY_CONV = (N_PER_NMOL * DENSITY_APATITE * (U238YR_SF / U238YR)
                * ETA_Q * ETCHABLE_RANGE / 8.0)

#: ``rdimLog2`` per HE_PREC_GOOD / BETTER / BEST.  The port takes ``rdim``
#: directly, but these are the oracle's settings.
PREC_RDIM = {0: 128, 1: 256, 2: 512}

_TINY = 1e-30


# --------------------------------------------------------------------------
# 1. Grain setup -- everything that does not depend on the thermal history
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GrainSetup:
    """Radial discretisation and composition of a batch of grains."""
    radius: Tensor        # (B,) um
    spacing: Tensor       # (B,) cm, the radial node spacing
    nmpg: Tensor          # (B, 4) nmol/g of 238U, 235U, 232Th, 147Sm
    a_depl: Tensor        # (B, 4, rdim) fraction of alphas retained per node
    totals: Tensor        # (B, 4) volume-integrated parent, He-yield-scaled
    ft: Tensor            # (B, 4) alpha-ejection correction per isotope
    rdim: int

    @property
    def device(self):
        return self.radius.device


def grain_setup(radius: Tensor, U: Tensor, Th: Tensor, Sm: Tensor,
                rdim: int = 512) -> GrainSetup:
    """``RDAAM_PrepModel`` and ``RDAAM_CalcAlphaCorrectionFactor``, batched.

    Args:
        radius: ``(B,)`` equivalent spherical radius in um.
        U, Th, Sm: ``(B,)`` parent concentrations in ppm.
        rdim: radial node count; the oracle uses 512 at ``PREC_BEST``.

    Returns:
        A :class:`GrainSetup`.  Differentiable with respect to all four inputs.
    """
    if rdim < 8:
        raise ValueError(f"rdim must be at least 8, got {rdim}")
    dtype, device = radius.dtype, radius.device
    nmpg = torch.stack([
        U * ((U238U235RATIO - 1) / U238U235RATIO) * PPM_NMPG_U238,
        U * (1.0 / U238U235RATIO) * PPM_NMPG_U235,
        Th * PPM_NMPG_TH232,
        Sm * PPM_NMPG_SM147,
    ], dim=1)                                                  # (B, 4)

    spacing = (radius / 1.0e4) / (rdim + 0.5)                  # (B,) cm
    i = torch.arange(rdim, dtype=dtype, device=device)

    # Alpha ejection: the fraction of alphas from node i that stop inside the
    # grain, for each isotope's stopping distance.
    xi = (i + 0.5) * spacing[:, None] * 1.0e4                  # (B, rdim) um
    rad = radius[:, None, None]
    s = torch.tensor(AS_DIST, dtype=dtype, device=device)[None, :, None]
    x = xi[:, None, :]
    xs = (x * x + rad * rad - s * s) / (2.0 * x)
    edge = (0.5 + (xs - x) / (2.0 * s)).clamp_min(0.0)
    a_depl = torch.where(x > rad - s, edge, torch.ones_like(edge))   # (B, 4, rdim)

    # Shell volumes, in units of r^3 -- the 4/3 pi cancels in every ratio and
    # is divided out of the He total, exactly as the C does it.
    outer = ((i + 1) * spacing[:, None]) ** 3
    inner = torch.cat([torch.zeros_like(outer[:, :1]), outer[:, :-1]], dim=1)
    vol = outer - inner                                        # (B, rdim)

    # Ft is the ejection-weighted mean over these shells, which reach only to
    # `rdim * spacing`.  RDAAM_CalcAlphaCorrectionFactor stops there; the parent
    # totals below do not, so the two use different volumes.  Keep it that way.
    denom = vol.sum(dim=1)[:, None]
    ft = (vol[:, None, :] * a_depl).sum(dim=2) / denom          # (B, 4)
    # The C reports 0 for an absent isotope rather than its geometric factor.
    # It never matters (the totals are zero too), but keep the two in step.
    ft = torch.where(nmpg == 0, torch.zeros_like(ft), ft)

    # The parent totals add a final half-node shell out to the true radius, so
    # they integrate the whole grain (RDAAM_PrepModel's "last drib").
    vol_total = vol.clone()
    vol_total[:, -1] = vol_total[:, -1] + ((rdim + 0.5) * spacing) ** 3 - outer[:, -1]
    yields = torch.tensor(YIELD, dtype=dtype, device=device)[None, :]
    totals = vol_total.sum(dim=1)[:, None] * nmpg * yields

    return GrainSetup(radius=radius, spacing=spacing, nmpg=nmpg, a_depl=a_depl,
                      totals=totals, ft=ft, rdim=rdim)


# --------------------------------------------------------------------------
# 2. Radiation damage: annealed fission-track density through time
# --------------------------------------------------------------------------

def _lag(x: Tensor, k: int) -> Tensor:
    """``y[..., m] = x[..., m - k]``, zero-filled where ``m < k``."""
    if k == 0:
        return x
    out = torch.zeros_like(x)
    if k < x.shape[-1]:
        out[..., k:] = x[..., :-k]
    return out


def annealing_traps(tau: Tensor, TK: Tensor, nmpg: Tensor) -> Tensor:
    """``RDAAM_InitFTAnnealingTraps``: trapped damage at every time node.

    For each time step ``m`` the C re-runs the Ketcham (2007) annealing chain
    from ``m`` back through every earlier cohort, giving an ``O(N^2)``
    triangular sweep.  Those ``N`` chains are independent of each other, so here
    they all advance together: step ``s`` of the loop handles cohort ``m - s``
    of chain ``m``, for every ``m`` at once.  Depth is ``O(N)``, work ``O(N^2)``.

    The result is radially uniform, because the C builds it from ``nmpg[0]``
    alone -- there is no zoning in this model.  Callers rely on that to keep the
    diffusivity a scalar per grain per step rather than a radial profile.

    Args:
        tau: ``(B, N)`` node times in seconds, oldest first.
        TK: ``(B, N)`` node temperatures in Kelvin, oldest first.
        nmpg: ``(B, 4)`` parent concentrations in nmol/g.

    Returns:
        ``(B, N)`` trap density at each node.  Entry 0 is zero: no damage has
        accumulated when the path begins.
    """
    N = tau.shape[1]
    kappa = min(1.04 - TRAP_RMR0, 1.0)
    equiv_tot = TRAP_TOT_ANNEAL_LEN ** (1.0 / kappa) * (1.0 - TRAP_RMR0) + TRAP_RMR0

    dt = tau[:, :-1] - tau[:, 1:]                               # (B, N-1) > 0
    # He (and so damage) produced in each interval, from RDAAM's baseline.
    decay = torch.tensor([U238SEC, U235SEC, TH232SEC, SM147SEC],
                         dtype=tau.dtype, device=tau.device)
    yields = torch.tensor(YIELD, dtype=tau.dtype, device=tau.device)
    # exp(l t_j) - exp(l t_{j+1}), written so it keeps its precision in float32.
    produced = (torch.exp(decay * tau[:, 1:, None])
                * torch.expm1(decay * dt[:, :, None]))          # (B, N-1, 4)
    he_state = DENSITY_CONV * (produced * yields * nmpg[:, None, :]).sum(dim=2)

    # `tc0[m]` is the chain's starting temperature term: -log(mean T over the
    # interval [m, m+1]).  The chain for m then walks to older intervals.
    tc_mid = -torch.log((TK[:, :-1] + TK[:, 1:]) / 2.0)         # (B, N-1)
    temp_calc = tc_mid
    equiv_time = torch.zeros_like(tc_mid)
    alive = torch.ones_like(tc_mid, dtype=torch.bool)
    trap = torch.zeros_like(tc_mid)

    for s in range(N - 1):
        # chain m is looking at cohort j = m - s
        started = torch.zeros_like(alive)
        started[:, s:] = True
        time_int = (_lag(dt, s) + equiv_time).clamp_min(_TINY)
        x1 = (torch.log(time_int) - AN_C2) / (temp_calc - AN_C3)
        # x2 < 0 is the no-annealing limit; the C takes a fractional power of it
        # and produces NaN.  Clamping to 0 gives f -> 1, which is that limit.
        x2 = (AN_C0 + AN_C1 * x1).clamp_min(0.0)
        f = 1.0 / (1.0 + x2 ** (1.0 / AN_A))
        f = torch.where(f < equiv_tot, torch.zeros_like(f), f)
        f = torch.where(alive & started, f, torch.zeros_like(f))

        # rmr0 conversion, then the track-density correction for observational
        # bias.  Threshold 0.757 here; KETCH.c uses 0.765 for the same formula.
        c = ((f - TRAP_RMR0) / (1.0 - TRAP_RMR0)).clamp_min(0.0) ** kappa
        dens = torch.where(c > 0.757, 1.600 * c - 0.599,
                           9.205 * c * c - 9.157 * c + 2.269)
        live = alive & started & (f > 0)
        trap = trap + torch.where(live, dens * _lag(he_state, s), torch.zeros_like(dens))

        alive = live
        if s + 1 < N - 1:
            # The C reuses x1 rather than inverting f; algebraically the same,
            # and it avoids a pow/inverse-pow round trip.
            # A chain whose oldest cohort this was has no next interval: its
            # `tc_new` is _lag's zero fill, not a temperature.  Updating it is
            # harmless in the forward pass (the chain is never read again) but
            # exp(x1 * (0 - AN_C3) + AN_C2) overflows float32, and that inf,
            # masked or not, sends NaN back into d/dT.  So such chains are not
            # updated, and x1 is taken as 0 wherever it is not used.
            has_next = torch.zeros_like(alive)
            has_next[:, s + 1:] = True
            update = live & (f < 0.999) & has_next
            tc_new = _lag(tc_mid, s + 1)
            x1_safe = torch.where(update, x1, torch.zeros_like(x1))
            equiv_time = torch.where(update, torch.exp(x1_safe * (tc_new - AN_C3) + AN_C2),
                                     equiv_time)
            temp_calc = torch.where(update, tc_new, temp_calc)

    # trap[m] is the damage present at node m+1; node 0 starts undamaged.
    return torch.cat([torch.zeros_like(trap[:, :1]), trap], dim=1)


# --------------------------------------------------------------------------
# 3. Crank-Nicolson diffusion
# --------------------------------------------------------------------------

def _cn_step(u, tau_n, tau_np1, TK_n, TK_np1, trap_n, trap_np1,
             spacing, a_depl, nmpg, exp_old):
    """One Crank-Nicolson step of ``RDAAM_CalcHeAge``'s inner loop."""
    dt = tau_n - tau_np1                                        # (B,) > 0
    h = spacing
    pre_beta = 2.0 * h * h / dt                                 # (B,)

    diff_n = D_INF * torch.exp(-E_ACT * 1000.0 / (UNIV_GAS_CONST * TK_n))
    diff_np1 = D_INF * torch.exp(-E_ACT * 1000.0 / (UNIV_GAS_CONST * TK_np1))
    trap_exp_n = torch.exp(E_TRAP * 1000.0 / (UNIV_GAS_CONST * TK_n))
    trap_exp_np1 = torch.exp(E_TRAP * 1000.0 / (UNIV_GAS_CONST * TK_np1))

    # Flowers et al. (2009) damage-diffusivity relation.  Radially uniform,
    # because `annealing_traps` is -- so these stay scalars per grain.
    d_n = diff_n / ((PSI * trap_n + POLY_A * trap_n ** 3) * trap_exp_n + 1.0)
    d_np1 = diff_np1 / ((PSI * trap_np1 + POLY_A * trap_np1 ** 3) * trap_exp_np1 + 1.0)
    d_n, d_np1 = d_n[:, None], d_np1[:, None]                   # (B, 1)

    # He produced in this interval, per radial node, already alpha-ejected.
    decay = torch.tensor([U238SEC, U235SEC, TH232SEC, SM147SEC],
                         dtype=u.dtype, device=u.device)
    exp_new = torch.exp(decay * tau_np1[:, None])               # (B, 4)
    yields = torch.tensor(YIELD, dtype=u.dtype, device=u.device)
    coeff = (exp_old - exp_new) * yields * nmpg                 # (B, 4)
    rdim = u.shape[1]
    r = (torch.arange(rdim, dtype=u.dtype, device=u.device) + 0.5) * h[:, None]
    prod = (coeff[:, :, None] * a_depl).sum(dim=1) * r * pre_beta[:, None]

    pb = pre_beta[:, None]
    diag = -(pb + 2.0 * d_np1).expand(-1, rdim).clone()
    # Neumann BC at the centre: the C doubles the inner coefficient there.
    diag[:, 0] = -(pb + 3.0 * d_np1)[:, 0]
    sub = d_np1.expand(-1, rdim)
    sup = d_np1.expand(-1, rdim).clone()
    sup[:, -1] = 0.0                                            # Dirichlet at the rim

    u_left = torch.cat([torch.zeros_like(u[:, :1]), u[:, :-1]], dim=1)
    u_right = torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], dim=1)
    rhs = (-pb + 2.0 * d_n) * u - d_n * u_right - d_n * u_left - prod
    rhs[:, 0] = ((-pb + 3.0 * d_n)[:, 0] * u[:, 0] - (d_n * u_right)[:, 0] - prod[:, 0])

    return solve_tridiagonal(sub, diag, sup, rhs), exp_new


# --------------------------------------------------------------------------
# 4. Integrating the He profile, and the age equation
# --------------------------------------------------------------------------

def _romberg_weights(rdim: int) -> list[float]:
    """Weights of ``RDAAM_ExtractHeProfile``'s Romberg rule, as a linear map.

    Every stage of the C's routine -- the trapezoid doubling, and the 5-point
    polynomial extrapolation of the refinement sequence to zero step size -- is
    linear in the integrand, and the ``gridSpacing*rdim`` factor is common to
    all of them.  So the whole thing collapses to one fixed weight per node,
    which is both exact and far cheaper than replaying the recursion.
    """
    import numpy as np

    level = rdim.bit_length() - 1
    if 1 << level != rdim:
        raise ValueError(f"rdim must be a power of two, got {rdim}")
    if level < 5:
        raise ValueError(f"rdim must be at least 32 for the 5-point extrapolation, got {rdim}")

    eye = np.eye(rdim)                       # basis vector per node
    step = [0.5 * eye[0]]                    # gam[rdim] is zero
    fact = start = rdim // 2
    it = 1
    for i in range(1, level + 1):
        total = eye[start:rdim:fact].sum(axis=0)
        step.append(0.5 * (step[i - 1] + total / it))
        it *= 2
        start //= 2
        if i > 1:
            fact //= 2

    # Extrapolate (h_k, step_k) to h = 0 over the last five refinements.
    # h_k = 0.25**k, so these Lagrange weights are pure constants.
    ks = list(range(level - 4, level + 1))
    xs = [0.25 ** k for k in ks]
    out = np.zeros(rdim)
    for p, k in enumerate(ks):
        w = 1.0
        for q in range(5):
            if q != p:
                w *= -xs[q] / (xs[p] - xs[q])
        out += w * step[k]
    return out.tolist()


def total_he(u: Tensor, spacing: Tensor) -> Tensor:
    """``RDAAM_ExtractHeProfile``: integrate the He profile over the grain.

    Args:
        u: ``(B, rdim)`` the transformed variable ``He * r``.
        spacing: ``(B,)`` radial node spacing in cm.

    Returns:
        ``(B,)`` total He in the C's internal units.
    """
    rdim = u.shape[1]
    r = (torch.arange(rdim, dtype=u.dtype, device=u.device) + 0.5) * spacing[:, None]
    he_profile = u / r
    gam = he_profile * 4.0 * PI * r * r

    w = torch.tensor(_romberg_weights(rdim), dtype=u.dtype, device=u.device)
    he = (gam * w).sum(dim=1) * (spacing * rdim)
    # Fill in the crystal centre, a half-node before the first node (NR 4.1.10).
    he = he + 0.5 * spacing * (gam[:, 0] * 55.0 / 24.0 - gam[:, 1] * 59.0 / 24.0
                               + gam[:, 2] * 37.0 / 24.0 - gam[:, 3] * 9.0 / 24.0)
    he = he / (1.333333 * PI)   # 1.333333, not 4/3: as written in the C

    # The C discards the result outright if the profile has gone appreciably
    # negative, which only happens when the solve has become unstable.
    bad = he_profile.min(dim=1).values < -0.05 * he_profile[:, 0]
    return torch.where(bad, torch.zeros_like(he), he)


def _solve_age(totals: Tensor, he: Tensor, max_age_yr: Tensor,
               iters: int = 12) -> Tensor:
    """Invert ``sum_i totals_i (exp(lambda_i a) - 1) = he`` for the age in Ma.

    The C bisects to a 100-year tolerance, which is neither differentiable nor
    better than a 5e-5 Ma quantisation of its own answer.  Newton from the
    linear-decay estimate converges here in a handful of steps because
    ``lambda * a`` stays below ~0.2 over any geological path, and it is exact
    at the fixed point, so autograd through the iterations gives the true
    implicit derivative.
    """
    lam = torch.tensor([U238YR, U235YR, TH232YR, SM147YR],
                       dtype=totals.dtype, device=totals.device)
    denom = (totals * lam).sum(dim=1)
    ok = denom > 0
    age = torch.where(ok, he / torch.where(ok, denom, torch.ones_like(denom)),
                      torch.zeros_like(denom))
    for _ in range(iters):
        g = (totals * torch.expm1(lam * age[:, None])).sum(dim=1) - he
        dg = (totals * lam * torch.exp(lam * age[:, None])).sum(dim=1)
        age = age - torch.where(ok, g / torch.where(ok, dg, torch.ones_like(dg)),
                                torch.zeros_like(dg))
    # The C brackets its bisection at the path duration, so it cannot return
    # more; match that, and never return a negative age.
    return age.clamp(torch.zeros_like(age), max_age_yr) / 1.0e6


# --------------------------------------------------------------------------
# 5. The model
# --------------------------------------------------------------------------

def rdaam_age(t: Tensor, T: Tensor, radius: Tensor, U: Tensor, Th: Tensor,
              Sm: Tensor | None = None, n_sub: int = 16, rdim: int = 512,
              use_checkpoint: bool = True) -> dict[str, Tensor]:
    """Apatite (U-Th)/He age with radiation-damage accumulation.

    Differentiable with respect to ``t``, ``T``, ``radius``, ``U``, ``Th``
    and ``Sm``.

    Args:
        t: ``(B, N)`` control-point times in Ma, ``t[:, 0] == 0``, increasing.
        T: ``(B, N)`` control-point temperatures in degC.
        radius: ``(B,)`` equivalent spherical radius in um.
        U, Th, Sm: ``(B,)`` parent concentrations in ppm.
        n_sub: sub-intervals per control-point segment.  The cost is
            ``O((N*n_sub)^2)`` in the damage sweep, so raise it deliberately.
        rdim: radial nodes; the oracle uses 512 at its BEST setting.
        use_checkpoint: recompute the diffusion steps during the backward pass
            instead of storing them.  Without it the activations are
            ``O(B * rdim * n_steps * log2(rdim))``, which is gigabytes at any
            useful batch size.  Set False only if gradients are not needed and
            the extra speed matters.

    Returns:
        ``{"age", "corrected_age", "total_he"}``; ages in Ma, matching
        ``HeResult.age`` and ``.corrected_age`` from the native binding.
    """
    if Sm is None:
        Sm = torch.zeros_like(radius)
    setup = grain_setup(radius, U, Th, Sm, rdim=rdim)

    t_fine, T_fine = refine(t, T, n_sub)
    tau = t_fine.flip(-1) * SECS_PER_MA                # oldest first, seconds
    TK = T_fine.flip(-1) + KELVIN_AT_0C
    traps = annealing_traps(tau, TK, setup.nmpg)

    decay = torch.tensor([U238SEC, U235SEC, TH232SEC, SM147SEC],
                         dtype=tau.dtype, device=tau.device)
    u = torch.zeros(tau.shape[0], rdim, dtype=tau.dtype, device=tau.device)
    exp_old = torch.exp(decay * tau[:, :1])            # (B, 4)

    for node in range(tau.shape[1] - 1):
        args = (u, tau[:, node], tau[:, node + 1], TK[:, node], TK[:, node + 1],
                traps[:, node], traps[:, node + 1], setup.spacing, setup.a_depl,
                setup.nmpg, exp_old)
        if use_checkpoint and torch.is_grad_enabled():
            u, exp_old = checkpoint(_cn_step, *args, use_reentrant=False)
        else:
            u, exp_old = _cn_step(*args)

    he = total_he(u, setup.spacing)
    # No parent, no He, no age -- and the age equation is degenerate.
    live = (setup.totals.sum(dim=1) > 0)
    he = torch.where(live, he, torch.zeros_like(he))

    max_age = tau[:, 0] / SECS_PER_YR
    return {
        "age": _solve_age(setup.totals, he, max_age),
        "corrected_age": _solve_age(setup.totals * setup.ft, he, 1.2 * max_age),
        "total_he": he,
    }
