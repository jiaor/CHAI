"""The flow head: a conditional normalizing flow in the whitened latent.

Stage 12 measured what the Gaussian head cannot do (``docs/calibration.md``):
its posterior has the right second moments and the wrong shape -- too light
in the tails, unable to hold the atoms the prior's walls put on 38% of node
values, and rougher along the path than any real history.  This module puts
a density with a shape behind the same encoder, in the same coordinates, in
the same slot: :class:`FlowHead` returns a :class:`ConditionalFlow` that has
``log_prob`` and ``sample`` like the ``MultivariateNormal`` the Gaussian head
returns, and :class:`chai.models.gaussian.AmortizedPosterior` does not know
which of the two it is holding.

The flow, from the base to ``z``:

1. **Base.** ``u ~ N(0, I)`` or, with ``base="student_t"``, independent
   Student-t coordinates with one learned degrees-of-freedom per coordinate.
   The spline layers below are the identity outside ``[-B, B]``, so the tails
   of the posterior *are* the tails of the base -- which is why a heavier
   base is the first thing the plan says to try when the ranks come out
   U-shaped, and why it is a switch here rather than a rewrite.
2. **Coupling layers.** ``n_layers`` rational-quadratic spline couplings
   (Durkan et al. 2019): half the coordinates pass through a monotone spline
   whose knots are read off an MLP of the other half and the encoder's
   embedding; the halves alternate by parity and by position so every
   coordinate is transformed by every other layer.  The splines are the
   identity at initialisation.
3. **Affine.** ``z = mu(x) + L(x) v`` with the full Cholesky factor of the
   Gaussian head -- literally :class:`chai.models.gaussian.GaussianHead`,
   kept as the last layer.  With the couplings at the identity and a Gaussian
   base the flow *is* the Gaussian head, so a run starts from the Stage 11
   baseline (``N(0, I)`` for every input) and a bad flow can do no worse than
   fall back to it.

Densities are exact in both directions -- the spline inverts analytically --
and cost the same, so ``log_prob`` for training and ``sample`` for reporting
are one network pass each.  ``standardise`` pulls a true ``z`` back to the
base: under a calibrated head it is distributed as the base is, whatever the
head learned, which is what the training loop's residual diagnostics read.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chai.models.encoder import mlp
from chai.models.gaussian import GaussianHead, HeadConfig

__all__ = ["ConditionalFlow", "FlowConfig", "FlowHead", "rational_quadratic_spline"]


@dataclass(frozen=True)
class FlowConfig:
    """
    Args:
        n_layers: coupling layers.  Six means every coordinate is transformed
            three times.
        n_bins: spline bins across ``[-tail_bound, tail_bound]``.
        d_hidden, n_hidden: the conditioner MLP in each layer.
        tail_bound: the splines act inside ``[-B, B]`` in base units and are the
            identity outside, so the base's tails are the posterior's.
        base: ``"gaussian"`` or ``"student_t"`` (per-coordinate learned df).
        df_init, df_min: the Student-t degrees of freedom start at ``df_init``
            -- 30 is Gaussian to within a percent -- and cannot go below
            ``df_min``; 2.5 keeps the variance finite.
        affine_hidden, min_scale: the affine layer, as :class:`HeadConfig`.
    """

    kind: str = "flow"
    n_layers: int = 6
    n_bins: int = 8
    d_hidden: int = 256
    n_hidden: int = 2
    tail_bound: float = 5.0
    base: str = "gaussian"
    df_init: float = 30.0
    df_min: float = 2.5
    affine_hidden: int = 256
    min_scale: float = 1e-3

    def __post_init__(self) -> None:
        if self.kind != "flow":
            raise ValueError(f"FlowConfig.kind must be 'flow', got {self.kind!r}")
        if self.base not in ("gaussian", "student_t"):
            raise ValueError(f"base must be 'gaussian' or 'student_t', got {self.base!r}")

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# The spline
# ---------------------------------------------------------------------------

_MIN_BIN = 1e-3
_MIN_DERIV = 1e-3
#: ``softplus(c) + _MIN_DERIV == 1``: the raw value at which a knot's slope is one.
_UNIT_SLOPE = math.log(math.expm1(1.0 - _MIN_DERIV))


def rational_quadratic_spline(x: Tensor, raw_w: Tensor, raw_h: Tensor, raw_d: Tensor, *,
                              inverse: bool, tail_bound: float) -> tuple[Tensor, Tensor]:
    """Monotone rational-quadratic spline on ``[-B, B]``, identity outside.

    ``x`` is ``(...)``; ``raw_w`` and ``raw_h`` are ``(..., K)`` unnormalised
    bin widths and heights, ``raw_d`` ``(..., K - 1)`` unnormalised slopes at
    the interior knots (the boundary slopes are one, so the map is C1 with
    its identity tails).  All-zero raw parameters give the identity.
    Returns ``(y, log|dy/dx|)``; with ``inverse=True`` the map is inverted and
    the log-derivative is that of the inverse.
    """
    B = tail_bound
    K = raw_w.shape[-1]
    inside = (x > -B) & (x < B)
    # Everything is computed for every point with x pulled into the interval, and the
    # tails are put back with a ``where``: no boolean gather/scatter, no host sync.
    xi = x.clamp(-B, B)
    w = _MIN_BIN + (1.0 - _MIN_BIN * K) * torch.softmax(raw_w, dim=-1)
    h = _MIN_BIN + (1.0 - _MIN_BIN * K) * torch.softmax(raw_h, dim=-1)
    d = _MIN_DERIV + F.softplus(raw_d + _UNIT_SLOPE)
    d = F.pad(d, (1, 1), value=1.0)                                # unit slopes at the ends

    cw = F.pad(torch.cumsum(w, dim=-1), (1, 0), value=0.0) * (2 * B) - B
    ch = F.pad(torch.cumsum(h, dim=-1), (1, 0), value=0.0) * (2 * B) - B
    cw[..., -1] = B
    ch[..., -1] = B
    w = cw[..., 1:] - cw[..., :-1]
    h = ch[..., 1:] - ch[..., :-1]

    knots = ch if inverse else cw
    # The number of knots at or below x, less one: ``searchsorted(right=True) - 1``, the
    # same index, written with ops ONNX has (it has no searchsorted; PLAN.md Stage 17b).
    idx = (knots <= xi.unsqueeze(-1)).sum(-1, keepdim=True) - 1
    idx = idx.clamp(0, K - 1)

    def pick(a: Tensor) -> Tensor:
        return a.gather(-1, idx).squeeze(-1)

    x0, wk = pick(cw), pick(w)
    y0, hk = pick(ch), pick(h)
    d0, d1 = pick(d[..., :-1]), pick(d[..., 1:])
    s = hk / wk                                                    # the bin's mean slope
    if inverse:
        r = xi - y0
        a = hk * (s - d0) + r * (d0 + d1 - 2 * s)
        b = hk * d0 - r * (d0 + d1 - 2 * s)
        c = -s * r
        disc = (b * b - 4 * a * c).clamp_min(0.0)
        theta = (2 * c) / (-b - torch.sqrt(disc))
        theta = theta.clamp(0.0, 1.0)
        yi = theta * wk + x0
    else:
        theta = ((xi - x0) / wk).clamp(0.0, 1.0)
    t1 = theta * (1.0 - theta)
    denom = s + (d0 + d1 - 2 * s) * t1
    if not inverse:
        yi = y0 + hk * (s * theta * theta + d0 * t1) / denom
    dnum = s * s * (d1 * theta * theta + 2 * s * t1 + d0 * (1.0 - theta) ** 2)
    ld = torch.log(dnum) - 2.0 * torch.log(denom)
    y = torch.where(inside, yi, x)
    logdet = torch.where(inside, -ld if inverse else ld, torch.zeros_like(x))
    return y, logdet


# ---------------------------------------------------------------------------
# Coupling layers
# ---------------------------------------------------------------------------

def _masks(dim: int, n_layers: int) -> list[Tensor]:
    """Which coordinates each layer *conditions on* (True) and transforms (False).

    Cycles parity, position, and their complements, so that neighbours in
    the latent's ordering -- adjacent nodes -- and the two halves of the
    path each get to see one another.
    """
    idx = torch.arange(dim)
    patterns = [idx % 2 == 0, idx < dim // 2, idx % 2 == 1, idx >= dim // 2]
    return [patterns[i % len(patterns)] for i in range(n_layers)]


class CouplingLayer(nn.Module):
    """One spline coupling: the free half is splined with knots from the masked half."""

    def __init__(self, dim: int, d_ctx: int, mask: Tensor, config: FlowConfig):
        super().__init__()
        self.config = config
        self.register_buffer("mask", mask.clone())
        self.register_buffer("ix_masked", torch.nonzero(mask).squeeze(-1))
        self.register_buffer("ix_free", torch.nonzero(~mask).squeeze(-1))
        self.n_masked = int(mask.sum())
        self.n_free = dim - self.n_masked
        K = config.n_bins
        self.n_params = 3 * K - 1
        self.net = mlp(self.n_masked + d_ctx, config.d_hidden, self.n_free * self.n_params,
                       n_hidden=config.n_hidden)
        # ``cat([masked, free])`` back into the latent's order.
        order = torch.cat([self.ix_masked, self.ix_free])
        self.register_buffer("unshuffle", torch.argsort(order))
        out: nn.Linear = self.net[-1]
        with torch.no_grad():                                  # start at the identity
            out.weight.mul_(1e-3)
            out.bias.zero_()

    def _params(self, x_masked: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        K = self.config.n_bins
        p = self.net(torch.cat([x_masked, ctx], dim=-1))
        p = p.reshape(*p.shape[:-1], self.n_free, self.n_params)
        return p[..., :K], p[..., K:2 * K], p[..., 2 * K:]

    def transform(self, x: Tensor, ctx: Tensor, *, inverse: bool) -> tuple[Tensor, Tensor]:
        """``(y, sum log|dy/dx|)`` for ``x`` ``(..., dim)`` and ``ctx`` ``(..., d_ctx)``."""
        xm = x.index_select(-1, self.ix_masked)
        xf = x.index_select(-1, self.ix_free)
        raw_w, raw_h, raw_d = self._params(xm, ctx)
        yf, ld = rational_quadratic_spline(xf, raw_w, raw_h, raw_d, inverse=inverse,
                                           tail_bound=self.config.tail_bound)
        y = torch.cat([xm, yf], dim=-1).index_select(-1, self.unshuffle)
        return y, ld.sum(-1)


# ---------------------------------------------------------------------------
# The head and what it returns
# ---------------------------------------------------------------------------

def _gamma(alpha: Tensor, shape: tuple[int, ...], generator: torch.Generator | None
           ) -> Tensor:
    """``Gamma(alpha, 1)`` draws of ``shape`` (broadcast over ``alpha``) from a generator.

    Marsaglia & Tsang (2000) for ``alpha >= 1``, vectorised: a few rounds of
    rejection fill essentially every slot (acceptance is above 95%), and the
    rare remainder is filled with the mean rather than looping forever.
    ``torch``'s own gamma sampler takes no generator, and the SBC loop needs
    its draws reproducible.
    """
    alpha = alpha.expand(shape)
    d = alpha - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = d.clone()                                            # the fallback: the mean
    todo = torch.ones(shape, dtype=torch.bool, device=alpha.device)
    kw = {"generator": generator, "device": alpha.device, "dtype": alpha.dtype}
    for _ in range(8):
        x = torch.randn(shape, **kw)
        v = (1.0 + c * x) ** 3
        u = torch.rand(shape, **kw)
        ok = (v > 0) & (torch.log(u.clamp_min(1e-30)) < 0.5 * x * x + d - d * v + d * torch.log(v.clamp_min(1e-30)))
        take = todo & ok
        out = torch.where(take, d * v, out)
        todo = todo & ~ok
        if not bool(todo.any()):
            break
    return out


class ConditionalFlow:
    """``q(z | x)`` for one batch: what :class:`FlowHead` returns.

    Holds the head, the embedding and the affine layer's ``(loc, L)`` for
    the batch.  ``batch_shape`` is ``loc.shape[:-1]``, ``event_shape``
    ``(dim,)``, like a ``torch.distributions`` object.
    """

    def __init__(self, head: FlowHead, embed: Tensor, loc: Tensor, scale_tril: Tensor):
        self.head = head
        self.embed = embed
        self.loc = loc
        self.scale_tril = scale_tril

    @property
    def batch_shape(self) -> torch.Size:
        return self.loc.shape[:-1]

    def _ctx(self, like: Tensor) -> Tensor:
        return self.embed.expand(*like.shape[:-1], self.embed.shape[-1])

    def standardise(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """``(u, log|du/dz|)``: ``z`` pulled back to the base."""
        v = torch.linalg.solve_triangular(self.scale_tril, (z - self.loc).unsqueeze(-1),
                                          upper=False).squeeze(-1)
        ld = -torch.log(torch.diagonal(self.scale_tril, dim1=-2, dim2=-1)).sum(-1)
        ctx = self._ctx(v)
        for layer in reversed(self.head.layers):
            v, l_ = layer.transform(v, ctx, inverse=True)
            ld = ld + l_
        return v, ld

    def push(self, u: Tensor) -> Tensor:
        """``z`` for base draws ``u`` ``(..., dim)``; the embedding broadcasts."""
        ctx = self._ctx(u)
        v = u
        for layer in self.head.layers:
            v, _ = layer.transform(v, ctx, inverse=False)
        return self.loc + torch.einsum("...ij,...j->...i", self.scale_tril, v)

    def log_prob(self, z: Tensor) -> Tensor:
        u, ld = self.standardise(z)
        return self.head.base_log_prob(u) + ld

    def sample(self, n: int, generator: torch.Generator | None = None, *,
               max_rows: int = 131_072) -> Tensor:
        """``(n, *batch_shape, dim)`` draws, in chunks that fit alongside the training step."""
        rows = int(np.prod(self.batch_shape)) if len(self.batch_shape) else 1
        per = max(1, max_rows // rows)
        out = []
        for s in range(0, n, per):
            k = min(per, n - s)
            u = self.head.base_sample((k, *self.batch_shape, self.loc.shape[-1]), generator,
                                      like=self.loc)
            out.append(self.push(u))
        return torch.cat(out, dim=0)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        n = int(np.prod(sample_shape)) if sample_shape else 1
        z = self.sample(n)
        return z.reshape(*sample_shape, *z.shape[1:]) if sample_shape else z[0]

    def residual(self, z: Tensor) -> Tensor:
        """``Phi^-1(F_base(u))`` per coordinate: ``N(0, 1)`` under a calibrated head.

        For a Gaussian base this is ``u`` itself.  For a Student-t base the
        CDF goes through SciPy, so this is a diagnostic, not a training term.
        """
        u, _ = self.standardise(z)
        return self.head.base_to_normal(u)


class FlowHead(nn.Module):
    """``(n, d_in) -> ConditionalFlow`` over ``dim`` coordinates."""

    def __init__(self, d_in: int, dim: int, config: FlowConfig | None = None):
        super().__init__()
        self.config = config or FlowConfig()
        self.dim = dim
        c = self.config
        self.affine = GaussianHead(d_in, dim, HeadConfig(d_hidden=c.affine_hidden,
                                                         min_scale=c.min_scale))
        self.layers = nn.ModuleList([CouplingLayer(dim, d_in, m, c) for m in _masks(dim, c.n_layers)])
        if c.base == "student_t":
            raw = math.log(math.expm1(c.df_init - c.df_min))
            self.df_raw = nn.Parameter(torch.full((dim,), raw))
        else:
            self.df_raw = None

    # -- the base --------------------------------------------------------------

    @property
    def df(self) -> Tensor | None:
        """Degrees of freedom per coordinate, or ``None`` for a Gaussian base."""
        if self.df_raw is None:
            return None
        return self.config.df_min + F.softplus(self.df_raw)

    def base_log_prob(self, u: Tensor) -> Tensor:
        if self.df_raw is None:
            return -0.5 * u.pow(2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        nu = self.df
        lp = (torch.lgamma(0.5 * (nu + 1)) - torch.lgamma(0.5 * nu) - 0.5 * torch.log(nu * math.pi)
              - 0.5 * (nu + 1) * torch.log1p(u.pow(2) / nu))
        return lp.sum(-1)

    def base_sample(self, shape: tuple[int, ...], generator: torch.Generator | None, *,
                    like: Tensor) -> Tensor:
        eps = torch.randn(shape, generator=generator, device=like.device, dtype=like.dtype)
        if self.df_raw is None:
            return eps
        nu = self.df.detach().to(like.dtype)
        g = _gamma(0.5 * nu, shape, generator)                     # chi2_nu / 2
        return eps / torch.sqrt(2.0 * g / nu)

    def base_to_normal(self, u: Tensor) -> Tensor:
        if self.df_raw is None:
            return u
        from scipy.special import ndtri, stdtr
        nu = self.df.detach().to(torch.float64).cpu().numpy()
        p = stdtr(nu, u.detach().to(torch.float64).cpu().numpy())
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return torch.as_tensor(ndtri(p), dtype=u.dtype, device=u.device)

    # -- the head ----------------------------------------------------------------

    def forward(self, embed: Tensor) -> ConditionalFlow:
        out = self.affine.net(embed)
        loc, raw = out[..., : self.dim], out[..., self.dim:]
        return ConditionalFlow(self, embed, loc, self.affine.scale_tril(raw))
