"""The coordinates the Goal 2 network works in: the prior, whitened.

A thermal history is a 64-vector of temperatures on the fixed grid, and the
prior that drew it stores three hyperparameters beside it (``tau``,
``rate_sd``, ``drift``; the fourth, ``T_0``, *is* the present-day node, bit for
bit).  This module maps that joint ``(h, T)`` to a vector ``z`` whose
distribution under the prior is as close to ``N(0, I)`` as an affine map can
make it, and back.  Nothing is trained here.  Every number is estimated
exactly from the corpus, once, in seconds.

Why whiten at all, when a network could learn the scale?  Because the prior's
own structure -- the 0.41 correlation between the temperature at 10 Ma and at
60 Ma, the 79 K deep-past spread, the drift that ties the whole path to one
number -- is then *in the base distribution* rather than something the head
must re-derive from data that (PLAN.md Stage 1, audit finding 2) carries no
gradient past 67 Ma.  A conditional head that outputs ``N(0, I)`` where it has
learned nothing is already correct out there.

The map, in order:

1. **Hyperparameters to the real line.**  Each stored ``h`` is either
   log-uniform or uniform on a known range.  The default map is the
   *probit* of its prior CDF, ``y = Phi^-1(F(h))``, which makes each one
   exactly ``N(0, 1)`` under the prior, and -- because the three are drawn
   independently -- exactly ``N(0, I_3)`` jointly.  The alternative
   ``"loglogit"`` (log for the log-uniform ones, logit for the uniform ones) is
   what the Stage 10 review proposed; it leaves a uniform or a logistic, and
   it is kept so the two can be measured against each other.
2. **Nodes, optionally, to the real line.**  ``box_logit=True`` maps each node
   through the logit of its position in the temperature box, widened by a
   margin so the wall itself is finite.  Off by default: it fixes a Gaussian
   head's *leak* past the walls but not the *atoms* on them, and the study in
   ``docs/latent.md`` measures which of the two costs more.
3. **Whitening.**  ``z = W (x - mu)`` with ``mu`` and ``Sigma = W^-1 W^-T`` the
   mean and covariance of the joint vector over the train split.  With
   ``method="cholesky"`` (default) ``W = L^-1`` for the lower Cholesky factor
   of ``Sigma``, and because the vector is ordered hyperparameters first and
   then nodes from the present into the past, ``z_k`` is the *innovation* at
   node ``k`` given everything more recent -- so "the deep past follows the
   prior's conditional given the recent past", which is the Stage 12
   acceptance test, reads as "``z`` is standard normal past the last node the
   data can see".  ``method="pca"`` rotates onto the eigenbasis instead, for
   the spectrum.

4. **The present day, optionally, as an input** (``present_input=True``, Stage
   19c).  The present-day node leaves the vector and every other node enters as
   its excess over it, ``T_k - T_0``: a shift, Jacobian one.  ``T_0`` is then the
   user's surface temperature, handed to :meth:`inverse` rather than drawn, and
   the density is ``p(T_1.. | T_0)``.  Under the pulse prior the data barely see ``T_0``
   (``U(0, 25)``), so a head that drew it returned the prior's hard-edged
   uniform with soft edges and over-covered the recent nodes; and because no
   path is ever colder than today, ``T_0`` is also every other node's floor,
   which :meth:`inverse` applies when it clamps.

The whole map is a bijection with a tractable Jacobian, so a density in ``z``
is a density in ``(h, T)`` -- which is what the Stage 13 importance-sampling
correction needs from any head that sits in these coordinates.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor, nn

__all__ = ["HYPER_NAMES", "HyperSpec", "Latent", "LatentConfig"]

#: The hyperparameters that enter the latent, in the order they occupy it.
#: ``T_0`` is deliberately absent: it is ``T[..., 0]`` exactly, and a
#: covariance that carried both would be singular.
HYPER_NAMES: tuple[str, ...] = ("tau", "rate_sd", "drift")

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
#: How close to a range's edge a value may sit before the real-line map clips
#: it.  1e-7 in the unit interval is below float32 resolution of the stored
#: hyperparameters, so no corpus row is moved by it.
_EDGE = 1e-7


@dataclass(frozen=True)
class HyperSpec:
    """One stored hyperparameter: its range as stored, and its prior's shape."""

    name: str
    lo: float
    hi: float
    log_axis: bool

    @staticmethod
    def from_prior_config(cfg: dict[str, Any]) -> tuple[HyperSpec, ...]:
        """The specs, from the ``prior_config`` a corpus records.

        A GP prior's three.  ``drift`` is stored divided by ``t_max`` (K/Ma, not K
        over the time span), the same trap
        :func:`chai.eval.sensitivity.hyper_log_density` guards.  A pulse prior
        (``PulsePrior``) has none in the latent: two of its latents are counts,
        which no density can hold, and its paths are a density on their own
        (``docs/prior-v2.md``), so the latent is the path alone.
        """
        if "tau_range" not in cfg:
            return ()
        t_max = float(cfg["t_max"])
        d_lo, d_hi = cfg["drift_total_range"]
        return (
            HyperSpec("tau", *map(float, cfg["tau_range"]), log_axis=True),
            HyperSpec("rate_sd", *map(float, cfg["rate_sd_range"]), log_axis=True),
            HyperSpec("drift", float(d_lo) / t_max, float(d_hi) / t_max, log_axis=False),
        )


@dataclass(frozen=True)
class LatentConfig:
    """Everything that determines the map other than the fitted moments."""

    hyper_map: str = "probit"
    box_logit: bool = False
    box_margin: float = 1.0
    method: str = "cholesky"
    present_input: bool = False

    def __post_init__(self) -> None:
        if self.hyper_map not in ("probit", "loglogit"):
            raise ValueError(f"hyper_map must be 'probit' or 'loglogit', got {self.hyper_map!r}")
        if self.method not in ("cholesky", "pca"):
            raise ValueError(f"method must be 'cholesky' or 'pca', got {self.method!r}")
        if self.box_margin <= 0:
            raise ValueError(f"box_margin must be positive, got {self.box_margin}")
        if self.present_input and self.box_logit:
            raise ValueError("present_input takes the nodes as excesses over T_0; the box logit is not defined on them")


# ---------------------------------------------------------------------------
# Elementwise maps to the real line, with their Jacobians
# ---------------------------------------------------------------------------

def _unit(x: Tensor, lo: Tensor, hi: Tensor, log_axis: Tensor) -> tuple[Tensor, Tensor]:
    """Position of ``x`` in its range on the prior's own axis, and ``d u / d x``.

    Returns ``u`` in the open unit interval and the derivative, both ``(..., k)``.
    """
    g_lo = torch.where(log_axis, torch.log(lo), lo)
    g_hi = torch.where(log_axis, torch.log(hi), hi)
    span = g_hi - g_lo
    x_safe = torch.clamp(x, min=torch.finfo(x.dtype).tiny)
    g = torch.where(log_axis, torch.log(x_safe), x)
    u = ((g - g_lo) / span).clamp(_EDGE, 1.0 - _EDGE)
    du = torch.where(log_axis, 1.0 / (x_safe * span), 1.0 / span)
    return u, du


def _from_unit(u: Tensor, lo: Tensor, hi: Tensor, log_axis: Tensor) -> Tensor:
    g_lo = torch.where(log_axis, torch.log(lo), lo)
    g_hi = torch.where(log_axis, torch.log(hi), hi)
    g = g_lo + u * (g_hi - g_lo)
    return torch.where(log_axis, torch.exp(g), g)


def _probit(u: Tensor) -> Tensor:
    return torch.special.ndtri(u)


def _probit_inv(y: Tensor) -> Tensor:
    return torch.special.ndtr(y)


def _log_dprobit(y: Tensor) -> Tensor:
    """``log d Phi^-1(u) / d u`` evaluated at ``y = Phi^-1(u)``: ``-log phi(y)``."""
    return 0.5 * y * y + _LOG_SQRT_2PI


def _logit(u: Tensor) -> Tensor:
    return torch.log(u) - torch.log1p(-u)


def _logit_inv(y: Tensor) -> Tensor:
    return torch.sigmoid(y)


def _log_dlogit(u: Tensor) -> Tensor:
    return -torch.log(u) - torch.log1p(-u)


def _moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and (unbiased) covariance of the rows of ``x``."""
    mean = x.mean(axis=0)
    xc = x - mean
    return mean, (xc.T @ xc) / (x.shape[0] - 1)


# ---------------------------------------------------------------------------
# The latent
# ---------------------------------------------------------------------------

class Latent(nn.Module):
    """The bijection ``(h, T) <-> z`` described in the module docstring.

    Build one with :meth:`fit` (from a corpus file) or :meth:`fit_arrays`
    (from arrays already in memory), save it with :meth:`save`, and reload it
    with :meth:`load`.  Parameters live in buffers, so ``.to(device)`` and
    ``state_dict`` behave as for any module, and the shape of the problem --
    ``n_grid``, the box, the hyperparameter ranges -- is read from the corpus
    and never assumed.

    Shapes: ``T`` is ``(..., n_grid)``, hyperparameters are a dict of ``(...)``
    arrays keyed by name (``T_0`` may be present and is checked against
    ``T[..., 0]``), ``z`` is ``(..., n_hyper + n_grid)``.  Inputs may be any
    float dtype; the elementwise maps are evaluated in float64 and the result
    is returned in the input's dtype.
    """

    def __init__(self, config: LatentConfig, hyper: tuple[HyperSpec, ...],
                 t_grid: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray,
                 mean: np.ndarray, cov: np.ndarray, meta: dict[str, Any] | None = None,
                 path_mean: np.ndarray | None = None, path_cov: np.ndarray | None = None):
        super().__init__()
        self.config = config
        self.hyper = tuple(hyper)
        self.meta: dict[str, Any] = dict(meta or {})
        n_h, n_grid = len(hyper), int(np.asarray(t_grid).size)
        d = n_h + n_grid - int(config.present_input)
        if mean.shape != (d,) or cov.shape != (d, d):
            raise ValueError(f"mean/cov are {mean.shape}/{cov.shape}; expected {d}")
        f64 = {"dtype": torch.float64}
        self.register_buffer("t_grid", torch.as_tensor(np.asarray(t_grid), **f64))
        self.register_buffer("box_lo", torch.as_tensor(np.asarray(box_lo), **f64))
        self.register_buffer("box_hi", torch.as_tensor(np.asarray(box_hi), **f64))
        self.register_buffer("h_lo", torch.tensor([s.lo for s in hyper], **f64))
        self.register_buffer("h_hi", torch.tensor([s.hi for s in hyper], **f64))
        self.register_buffer("h_log", torch.tensor([s.log_axis for s in hyper], dtype=torch.bool))
        self.register_buffer("mean", torch.as_tensor(mean, **f64))
        self.register_buffer("cov", torch.as_tensor(cov, **f64))
        # The path's own moments in degrees, all nodes: what :meth:`path_conditional`
        # regresses on.  Without present_input (and no box logit) they are the
        # node block of ``mean`` / ``cov``, and older latents never stored them.
        if path_mean is None:
            if config.present_input or config.box_logit:
                raise ValueError("this latent's coordinates are not degrees: pass path_mean and path_cov")
            path_mean, path_cov = np.asarray(mean)[n_h:], np.asarray(cov)[n_h:, n_h:]
        self.register_buffer("path_mean", torch.as_tensor(np.asarray(path_mean), **f64))
        self.register_buffer("path_cov", torch.as_tensor(np.asarray(path_cov), **f64))
        W, W_inv, logdet = self._whitening(self.cov, config.method)
        self.register_buffer("W", W)
        self.register_buffer("W_inv", W_inv)
        self.register_buffer("logdet_W", logdet)

    # -- construction -------------------------------------------------------

    @staticmethod
    def _whitening(cov: Tensor, method: str) -> tuple[Tensor, Tensor, Tensor]:
        d = cov.shape[0]
        eye = torch.eye(d, dtype=cov.dtype)
        if method == "cholesky":
            L = torch.linalg.cholesky(cov)
            W = torch.linalg.solve_triangular(L, eye, upper=False)
            return W, L, -torch.log(torch.diagonal(L)).sum()
        evals, evecs = torch.linalg.eigh(cov)
        if evals.min() <= 0:
            raise ValueError("covariance is not positive definite")
        # Columns ordered largest variance first, so z[0] is the leading mode.
        evals, evecs = evals.flip(0), evecs.flip(1)
        W = (evecs / evals.sqrt()).T
        W_inv = evecs * evals.sqrt()
        return W, W_inv, -0.5 * torch.log(evals).sum()

    @classmethod
    def fit_arrays(cls, T: np.ndarray, hyper: dict[str, np.ndarray], *, t_grid: np.ndarray,
                   prior_config: dict[str, Any], config: LatentConfig | None = None,
                   meta: dict[str, Any] | None = None) -> Latent:
        """Estimate the map from arrays: ``T`` is ``(n, n_grid)``, ``hyper`` per row.

        The box is the prior's: ``[T_min, T_max]`` at every node except the
        present-day one, which the prior confines to ``T_present_range``.
        """
        config = config or LatentConfig()
        specs = HyperSpec.from_prior_config(prior_config)
        t_grid = np.asarray(t_grid, dtype=np.float64)
        n_grid = t_grid.size
        # A pulse prior has no walls; its support is [coolest T_0, T_max] (T_top in checkpoints before 2026-10-23).
        t_lo = prior_config.get("T_min", prior_config["T_present_range"][0])
        t_hi = prior_config.get("T_max", prior_config.get("T_top"))
        box_lo = np.full(n_grid, float(t_lo))
        box_hi = np.full(n_grid, float(t_hi))
        box_lo[0], box_hi[0] = map(float, prior_config["T_present_range"])
        # Moments are estimated with a provisional module whose whitening is
        # the identity, so the same elementwise code runs in both places.
        d = len(specs) + n_grid - int(config.present_input)
        T64 = np.asarray(T, dtype=np.float64)
        proto = cls(config, specs, t_grid, box_lo, box_hi, np.zeros(d), np.eye(d),
                    path_mean=np.zeros(n_grid), path_cov=np.eye(n_grid))
        x = proto._to_real_line(torch.as_tensor(T64), proto._stack_hyper(hyper, T64))[0].numpy()
        mean, cov = _moments(x)
        path_mean, path_cov = _moments(T64)
        info = {"n_fit": int(x.shape[0]), "prior_config": dict(prior_config), **(meta or {})}
        return cls(config, specs, t_grid, box_lo, box_hi, mean, cov, meta=info,
                   path_mean=path_mean, path_cov=path_cov)

    @classmethod
    def fit(cls, corpus: str | Path, *, split: str = "train", config: LatentConfig | None = None,
            max_rows: int | None = None, max_reheat: float | None = None) -> Latent:
        """Estimate the map from one split of a corpus file.

        Reads the whole ``T`` and ``hyper/*`` columns and takes the split's rows:
        the paths are 512 MB in float32, well inside memory.
        ``max_rows`` takes the first rows of the split, for quick looks.
        ``max_reheat`` keeps the rows that reheat by at most that many kelvin,
        as :class:`chai.train.data.CorpusSplit` does: a cooling-only member's
        latent is fitted on its own rows (under a pulse prior, the ``N = 0`` rows).
        """
        corpus = Path(corpus)
        with h5py.File(corpus, "r") as f:
            names = list(f.attrs["split_names"])
            idx = np.flatnonzero(f["split"][:] == names.index(split))
            if max_rows is not None:
                idx = idx[:max_rows]
            # Whole-column reads then a numpy take: h5py's fancy indexing walks
            # the selection element by element and is ~100x slower on 10^6 rows.
            T = f["T"][:][idx]
            if max_reheat is not None:
                keep = (np.maximum.accumulate(T, axis=1) - T).max(axis=1) <= max_reheat
                idx, T = idx[keep], T[keep]
            prior_cfg = json.loads(f.attrs["prior_config"])
            names = [s.name for s in HyperSpec.from_prior_config(prior_cfg)]
            hyper = {k: f[f"hyper/{k}"][:][idx] for k in names}
            t_grid = f["t_grid"][:]
            meta = {"corpus": str(corpus), "split": split, "max_reheat": max_reheat,
                    "corpus_git_sha": str(f.attrs.get("git_sha", ""))}
        return cls.fit_arrays(T, hyper, t_grid=t_grid, prior_config=prior_cfg,
                              config=config, meta=meta)

    # -- shape --------------------------------------------------------------

    @property
    def n_hyper(self) -> int:
        return len(self.hyper)

    @property
    def n_grid(self) -> int:
        return int(self.t_grid.numel())

    @property
    def present_input(self) -> bool:
        """Whether ``T_0`` is an input (handed to :meth:`inverse`) rather than a coordinate."""
        return self.config.present_input

    @property
    def dim(self) -> int:
        return self.n_hyper + self.n_grid - int(self.present_input)

    @property
    def hyper_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.hyper)

    def node_index(self, k: int) -> int:
        """Position of node ``k`` in the joint vector (the hyperparameters come first).

        With ``present_input`` node 0 is not in the vector, and node ``k`` is
        its excess over node 0.
        """
        if self.present_input:
            if k == 0:
                raise IndexError("node 0 is an input of this latent, not a coordinate")
            return self.n_hyper + k - 1
        return self.n_hyper + k

    # -- the elementwise half -----------------------------------------------

    def _stack_hyper(self, hyper: dict[str, Any], T: Any) -> Tensor:
        missing = set(self.hyper_names) - set(hyper)
        if missing:
            raise KeyError(f"hyper is missing {sorted(missing)}")
        cols = [torch.as_tensor(np.asarray(hyper[k]) if not torch.is_tensor(hyper[k]) else hyper[k],
                                dtype=torch.float64) for k in self.hyper_names]
        if cols:
            h = torch.stack(cols, dim=-1)
        else:
            shape = (np.asarray(T) if not torch.is_tensor(T) else T).shape[:-1]
            h = torch.zeros(*shape, 0, dtype=torch.float64)
        if "T_0" in hyper:
            T0 = torch.as_tensor(np.asarray(hyper["T_0"]) if not torch.is_tensor(hyper["T_0"])
                                 else hyper["T_0"], dtype=torch.float64)
            Tt = torch.as_tensor(np.asarray(T) if not torch.is_tensor(T) else T, dtype=torch.float64)
            if not torch.allclose(T0, Tt[..., 0], atol=1e-4, rtol=0):
                raise ValueError("hyper['T_0'] disagrees with T[..., 0]; the present-day node "
                                 "is the hyperparameter, and the latent carries it once")
        return h

    def _to_real_line(self, T: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """``(x, log|dx/d(h,T)|)`` for ``T`` ``(..., n_grid)`` and ``h`` ``(..., n_hyper)``."""
        u, du = _unit(h, self.h_lo, self.h_hi, self.h_log)
        if self.config.hyper_map == "probit":
            y_h = _probit(u)
            ld = (_log_dprobit(y_h) + torch.log(du)).sum(-1)
        else:
            lin = ~self.h_log
            y_h = torch.where(lin, _logit(u), torch.log(torch.clamp(h, min=torch.finfo(h.dtype).tiny)))
            ld = torch.where(lin, _log_dlogit(u) + torch.log(du), -torch.log(torch.clamp(h, min=1e-300))).sum(-1)
        if self.config.box_logit:
            m = self.config.box_margin
            lo, hi = self.box_lo - m, self.box_hi + m
            u_T = ((T - lo) / (hi - lo)).clamp(_EDGE, 1.0 - _EDGE)
            y_T = _logit(u_T)
            ld = ld + (_log_dlogit(u_T) - torch.log(hi - lo)).sum(-1)
        elif self.present_input:
            y_T = T[..., 1:] - T[..., :1]
        else:
            y_T = T
        return torch.cat([y_h, y_T], dim=-1), ld

    def _from_real_line(self, x: Tensor, T0: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """``(h, T, log|d(h,T)/dx|)`` -- the inverse of :meth:`_to_real_line`; ``T0`` with ``present_input``."""
        y_h, y_T = x[..., : self.n_hyper], x[..., self.n_hyper:]
        span = torch.where(self.h_log, torch.log(self.h_hi) - torch.log(self.h_lo),
                           self.h_hi - self.h_lo)
        if self.config.hyper_map == "probit":
            u = _probit_inv(y_h).clamp(_EDGE, 1.0 - _EDGE)
            h = _from_unit(u, self.h_lo, self.h_hi, self.h_log)
            # d h / d y = (d h / d u) phi(y): span * h on a log axis, span on a linear one.
            ld_h = -_log_dprobit(y_h) + torch.log(torch.where(self.h_log, span * h, span))
        else:
            u = _logit_inv(y_h).clamp(_EDGE, 1.0 - _EDGE)
            h = torch.where(self.h_log, torch.exp(y_h), self.h_lo + u * span)
            ld_h = torch.where(self.h_log, y_h, -_log_dlogit(u) + torch.log(span))
        ld = ld_h.sum(-1)
        if self.config.box_logit:
            m = self.config.box_margin
            lo, hi = self.box_lo - m, self.box_hi + m
            u_T = _logit_inv(y_T).clamp(_EDGE, 1.0 - _EDGE)
            T = lo + u_T * (hi - lo)
            ld = ld + (-_log_dlogit(u_T) + torch.log(hi - lo)).sum(-1)
        elif self.present_input:
            T0 = T0.to(y_T).unsqueeze(-1).expand(*y_T.shape[:-1], 1)
            T = torch.cat([T0, T0 + y_T], dim=-1)
        else:
            T = y_T
        return h, T, ld

    # -- the public bijection -----------------------------------------------

    def forward(self, T: Tensor | np.ndarray, hyper: dict[str, Any]) -> Tensor:
        """``z`` for paths ``T`` ``(..., n_grid)`` and their hyperparameters."""
        return self.forward_with_logdet(T, hyper)[0]

    def forward_with_logdet(self, T: Tensor | np.ndarray,
                            hyper: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """``(z, log|dz/d(h,T)|)``.  A density ``q(z)`` is ``q(z) * exp(logdet)`` in ``(h, T)``."""
        Tt = torch.as_tensor(np.asarray(T) if not torch.is_tensor(T) else T)
        out_dtype = Tt.dtype if Tt.is_floating_point() else torch.float32
        Tt = Tt.to(torch.float64)
        if Tt.shape[-1] != self.n_grid:
            raise ValueError(f"T has {Tt.shape[-1]} nodes; this latent was fitted on {self.n_grid}")
        h = self._stack_hyper(hyper, Tt).to(Tt.device)
        x, ld = self._to_real_line(Tt, h)
        z = (x - self.mean) @ self.W.T
        return z.to(out_dtype), (ld + self.logdet_W).to(out_dtype)

    def inverse(self, z: Tensor, *, clamp: bool = False, T0: Tensor | float | None = None
                ) -> tuple[Tensor, dict[str, Tensor]]:
        """``(T, hyper)`` for ``z`` ``(..., dim)``; ``hyper`` includes ``T_0 = T[..., 0]``.

        ``clamp=True`` projects the returned path onto the temperature box,
        which is what the prior's own walls do to a path that presses on them
        (``docs/prior.md``, "the wall is a projection").  It is not a bijection
        any more, so no Jacobian is offered for it; use it for reporting paths,
        not for densities.

        ``T0`` is required with ``present_input`` and refused without: the
        present-day temperature, a number or anything broadcastable to
        ``z.shape[:-1]`` (one per row, shared by the draws in front).  Clamping
        then also holds every node at or above it.
        """
        return self.inverse_with_logdet(z, clamp=clamp, T0=T0)[:2]

    def inverse_with_logdet(self, z: Tensor, *, clamp: bool = False, T0: Tensor | float | None = None
                            ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        z = torch.as_tensor(z)
        out_dtype = z.dtype if z.is_floating_point() else torch.float32
        if z.shape[-1] != self.dim:
            raise ValueError(f"z has {z.shape[-1]} coordinates; this latent has {self.dim}")
        if self.present_input and T0 is None:
            raise ValueError("this latent takes the present-day temperature as an input: pass T0")
        if not self.present_input and T0 is not None:
            raise ValueError("this latent draws the present-day temperature; it takes no T0")
        x = self.mean + z.to(torch.float64) @ self.W_inv.T
        if T0 is not None:
            T0 = torch.as_tensor(T0, dtype=torch.float64, device=x.device).expand(x.shape[:-1])
        h, T, ld = self._from_real_line(x, T0)
        if clamp:
            T = torch.maximum(torch.minimum(T, self.box_hi), self.box_lo)
            if T0 is not None:
                T = torch.maximum(T, T0.unsqueeze(-1))
        hyper = {name: h[..., i].to(out_dtype) for i, name in enumerate(self.hyper_names)}
        hyper["T_0"] = T[..., 0].to(out_dtype)
        return T.to(out_dtype), hyper, (ld - self.logdet_W).to(out_dtype)

    def sample_prior(self, n: int, generator: torch.Generator | None = None, *,
                     clamp: bool = False, T0: Tensor | float | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """``n`` draws from the Gaussian that ``z ~ N(0, I)`` implies in ``(h, T)``.

        This is *not* the prior; it is what a head that has learned nothing
        would return, and the gap between the two is the leak the study
        measures.
        """
        z = torch.randn(n, self.dim, generator=generator, dtype=torch.float64, device=self.mean.device)
        return self.inverse(z, clamp=clamp, T0=T0)

    # -- what the covariance knows ------------------------------------------

    def conditional(self, target: np.ndarray | list[int], given: np.ndarray | list[int]
                    ) -> tuple[Tensor, Tensor, Tensor]:
        """The prior's Gaussian conditional of some coordinates given others.

        Returns ``(A, b, S)`` such that ``E[x_target | x_given] = A x_given + b`` and
        ``Cov = S``, on the real-line vector ``x`` (indices via :meth:`node_index`).
        With ``box_logit=False`` and node indices only, ``x`` is temperature and
        this is the Stage 12 deep-past test's reference: the regression of the
        deep nodes on the recent ones, estimated from the corpus and nowhere else.
        """
        t = torch.as_tensor(np.asarray(target), dtype=torch.long)
        g = torch.as_tensor(np.asarray(given), dtype=torch.long)
        S_tg = self.cov[t][:, g]
        S_gg = self.cov[g][:, g]
        A = torch.linalg.solve(S_gg, S_tg.T).T
        b = self.mean[t] - A @ self.mean[g]
        S = self.cov[t][:, t] - A @ S_tg.T
        return A, b, S

    def path_conditional(self, target: np.ndarray | list[int], given: np.ndarray | list[int]
                         ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`conditional` on the path in degrees, by node number rather than vector position.

        The deep-past test's reference.  Read from the path's own moments, so it
        is the same regression whatever coordinates the head works in.
        """
        t = torch.as_tensor(np.asarray(target), dtype=torch.long)
        g = torch.as_tensor(np.asarray(given), dtype=torch.long)
        S_tg = self.path_cov[t][:, g]
        A = torch.linalg.solve(self.path_cov[g][:, g], S_tg.T).T
        b = self.path_mean[t] - A @ self.path_mean[g]
        S = self.path_cov[t][:, t] - A @ S_tg.T
        return A, b, S

    def innovation_sd(self) -> Tensor:
        """``sd(x_k | x_<k)`` for every coordinate: the diagonal of the Cholesky factor.

        Independent of ``method`` -- computed from the covariance directly.
        """
        return torch.diagonal(torch.linalg.cholesky(self.cov))

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            "config": asdict(self.config),
            "hyper": [asdict(s) for s in self.hyper],
            "meta": self.meta,
            "t_grid": self.t_grid.numpy(), "box_lo": self.box_lo.numpy(),
            "box_hi": self.box_hi.numpy(), "mean": self.mean.numpy(), "cov": self.cov.numpy(),
            "path_mean": self.path_mean.numpy(), "path_cov": self.path_cov.numpy(),
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> Latent:
        d = torch.load(path, weights_only=False)
        return cls(LatentConfig(**d["config"]), tuple(HyperSpec(**s) for s in d["hyper"]),
                   d["t_grid"], d["box_lo"], d["box_hi"], d["mean"], d["cov"], meta=d["meta"],
                   path_mean=d.get("path_mean"), path_cov=d.get("path_cov"))

    def extra_repr(self) -> str:
        return (f"dim={self.dim} (n_hyper={self.n_hyper}, n_grid={self.n_grid}), "
                f"hyper_map={self.config.hyper_map!r}, box_logit={self.config.box_logit}, "
                f"method={self.config.method!r}, present_input={self.present_input}")
