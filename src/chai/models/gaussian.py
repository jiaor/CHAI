"""The first head: a conditional Gaussian in the whitened latent.

Given the encoder's summary of a sample, the head returns a multivariate
normal over ``z`` -- the 67 coordinates of :class:`chai.models.latent.Latent`
-- with a full covariance through its Cholesky factor.  It is the baseline
the plan asks for, and it is not a toy: a full covariance *is* correlated
path structure, which is what the Stage 1 audit's finding 3 ruled a per-node
head out for, and in these coordinates a head that has learned nothing is
already the prior's second moments (``mu = 0``, ``L = I``), which is where
it is initialised.

What it cannot do is also known in advance (``docs/latent.md``): the prior in
``z`` has excess kurtosis of 40--160 per node and 38% of its node values sit
exactly on a wall, and a Gaussian has neither tails nor atoms.  A *posterior*
is far narrower than the prior and may be Gaussian enough; Stage 12's SBC
decides.  Paths are reported through ``Latent.inverse(z, clamp=True)``,
densities never are.

:class:`AmortizedPosterior` is the whole model -- encoder, head and the
latent it was trained in -- so that a saved file is enough to go from a
laboratory's numbers to paths in degrees.  The flow head of the plan's next
step replaces :class:`GaussianHead` in the one slot and nothing else moves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import MultivariateNormal

from chai.models.encoder import AliquotEncoder, EncoderConfig, mlp
from chai.models.latent import Latent

if TYPE_CHECKING:
    from chai.models.flow import FlowConfig

__all__ = ["AmortizedPosterior", "GaussianHead", "HeadConfig", "draw_z", "head_config_from",
           "remove_floor", "residual_of", "sample_mvn"]


@dataclass(frozen=True)
class HeadConfig:
    """
    Args:
        d_hidden: width of the one hidden layer between embedding and outputs.
        min_scale: floor on the Cholesky diagonal, in whitened units (the prior's
            innovation is 1).  Keeps the log-density finite when a coordinate is
            pinned by the data to better than the floor.
    """

    d_hidden: int = 256
    min_scale: float = 1e-3

    def to_config(self) -> dict[str, Any]:
        return {"kind": "gaussian", **asdict(self)}


def head_config_from(cfg: dict[str, Any] | HeadConfig | FlowConfig) -> HeadConfig | FlowConfig:
    """The head config a dict names: ``kind`` is ``"gaussian"`` (the default) or ``"flow"``."""
    if not isinstance(cfg, dict):
        return cfg
    cfg = dict(cfg)
    kind = cfg.pop("kind", "gaussian")
    if kind == "gaussian":
        return HeadConfig(**cfg)
    if kind == "flow":
        from chai.models.flow import FlowConfig
        return FlowConfig(**cfg)
    raise ValueError(f"unknown head kind {kind!r}")


def sample_mvn(q: MultivariateNormal, n: int, generator: torch.Generator | None = None) -> Tensor:
    """``(n, *batch, dim)`` draws from ``q`` without materialising ``n`` copies of ``L``.

    ``MultivariateNormal.rsample`` broadcasts the Cholesky factor against the
    noise inside a ``matmul``, which for 100 draws of 4 096 rows in 67
    dimensions is a 7 GB temporary.  ``loc + einsum(L, eps)`` is the same
    draw in the memory of the draws themselves.
    """
    eps = torch.randn((n, *q.loc.shape), generator=generator, device=q.loc.device,
                      dtype=q.loc.dtype)
    return q.loc + torch.einsum("...ij,s...j->s...i", q.scale_tril, eps)


def remove_floor(T: Tensor, sd: float, *, first: int = 0) -> Tensor:
    """Take a known white noise floor back out of a head's draws.

    A head trained on ``T + N(0, sd^2 I)`` (``WallDequantiser.noise_sd``) draws
    ``y = T + e`` with ``e`` independent of ``T``, so each row's draw covariance is
    the posterior's plus ``sd^2 I``.  Each eigen-direction of that covariance is
    shrunk about the draws' mean to the variance left once the floor is taken
    out.  A direction that is the floor alone goes to zero; that is where a
    thin posterior's node-to-node noise lives.  Sample eigenvalues are read as
    a spiked covariance with ``p / S`` the aspect ratio (``p`` nodes, ``S``
    draws): below the Marchenko--Pastur edge ``sd^2 (1 + sqrt(p / S))^2`` a
    direction is floor, above it the population eigenvalue is recovered from
    the sample one before the floor is subtracted.

    Linear in the draws, and exact in the first two moments for a Gaussian
    posterior; for any other, the shape along each direction is kept and only
    its scale changes.  Nothing is fitted: the floor is the training config's.

    Args:
        T: ``(S, ..., n_grid)`` draws in kelvin, before any clamp; ``S`` in front.
        sd: the floor, kelvin.
        first: nodes before it carry no floor (``T_0`` given as an input) and are
            returned as they are.
    """
    X = T[..., first:].to(torch.float64)
    S, p = X.shape[0], X.shape[-1]
    if S < 2 * p:
        raise ValueError(f"{S} draws cannot resolve a {p}-node covariance per row; take at least {2 * p}")
    m = X.mean(0)
    D = X - m
    C = torch.einsum("s...i,s...j->...ij", D, D) / ((S - 1) * sd ** 2)
    lam, V = torch.linalg.eigh(C)
    g = p / S
    b = lam + 1 - g
    ell = 0.5 * (b + torch.sqrt((b * b - 4 * lam).clamp_min(0)))
    signal = torch.where(lam > (1 + g ** 0.5) ** 2, ell - 1, torch.zeros_like(lam))
    a = torch.sqrt(signal / lam.clamp_min(1e-12))
    A = V @ (a.unsqueeze(-1) * V.transpose(-1, -2))
    out = T.clone()
    out[..., first:] = (m + torch.einsum("s...i,...ji->s...j", D, A)).to(T.dtype)
    return out


def draw_z(q: Any, n: int, generator: torch.Generator | None = None) -> Tensor:
    """``(n, *batch, dim)`` draws from whatever a head returned."""
    if isinstance(q, MultivariateNormal):
        return sample_mvn(q, n, generator)
    return q.sample(n, generator)


def residual_of(q: Any, z: Tensor) -> Tensor:
    """``(rows, dim)`` standardised residual of ``z`` under ``q``: ``N(0, I)`` if calibrated.

    For a Gaussian, ``L^-1 (z - mu)``; for a flow, ``z`` pulled back to the
    base and mapped through the base's CDF to a normal quantile.  Either way
    the rank of the truth among draws from ``q`` is the normal CDF of it
    coordinate by coordinate in the head's own frame.
    """
    if isinstance(q, MultivariateNormal):
        r = torch.linalg.solve_triangular(q.scale_tril, (z - q.loc).unsqueeze(-1), upper=False)
        return r.squeeze(-1)
    return q.residual(z)


class GaussianHead(nn.Module):
    """``(n, d_in) -> MultivariateNormal`` over ``dim`` coordinates, full covariance.

    The Cholesky factor's diagonal is ``softplus(raw) + min_scale``; the
    strict lower triangle is read straight off the network.  At
    initialisation the output layer's weights are scaled down so that the
    head starts at ``N(0, I)`` for every input.
    """

    def __init__(self, d_in: int, dim: int, config: HeadConfig | None = None):
        super().__init__()
        self.config = config or HeadConfig()
        self.dim = dim
        self.n_tril = dim * (dim + 1) // 2
        self.net = mlp(d_in, self.config.d_hidden, dim + self.n_tril)
        idx = torch.tril_indices(dim, dim)
        self.register_buffer("tril_rows", idx[0])
        self.register_buffer("tril_cols", idx[1])
        self.register_buffer("is_diag", idx[0] == idx[1])
        # softplus(b) = 1 - min_scale on the diagonal, everything else zero.
        b = torch.log(torch.expm1(torch.tensor(1.0 - self.config.min_scale)))
        out: nn.Linear = self.net[-1]
        with torch.no_grad():
            out.weight.mul_(1e-2)
            out.bias.zero_()
            out.bias[dim:][self.is_diag] = b

    def scale_tril(self, raw: Tensor) -> Tensor:
        n = raw.shape[:-1]
        vals = torch.where(self.is_diag, F.softplus(raw) + self.config.min_scale, raw)
        L = raw.new_zeros(*n, self.dim, self.dim)
        L[..., self.tril_rows, self.tril_cols] = vals
        return L

    def forward(self, embed: Tensor) -> MultivariateNormal:
        out = self.net(embed)
        loc, raw = out[..., : self.dim], out[..., self.dim:]
        return MultivariateNormal(loc, scale_tril=self.scale_tril(raw))


class AmortizedPosterior(nn.Module):
    """Encoder, head and latent together: ``batch -> q(z | x)`` and paths in degrees.

    Args:
        latent: the fitted :class:`Latent`; its ``dim`` sets the head's, its
            ``n_grid`` the paths'.  Held as a submodule, so it saves and moves
            with the model.
        n_aft, n_bins, n_grain, aft_flags: the input shape, from the data
            module's :data:`AFT_INPUTS`, the corpus's bins and :data:`GRAIN_INPUTS`.
    """

    def __init__(self, latent: Latent, *, n_aft: int, n_bins: int, n_grain: int,
                 aft_flags: tuple[int, ...] = (), encoder: EncoderConfig | None = None,
                 head: HeadConfig | FlowConfig | dict[str, Any] | None = None,
                 meta: dict[str, Any] | None = None):
        super().__init__()
        self.latent = latent
        self.encoder = AliquotEncoder(n_aft, n_bins, n_grain, aft_flags=aft_flags,
                                      config=encoder, present_input=latent.present_input)
        head = head_config_from(head or HeadConfig())
        if isinstance(head, HeadConfig):
            self.head: nn.Module = GaussianHead(self.encoder.d_embed, latent.dim, config=head)
        else:
            from chai.models.flow import FlowHead
            self.head = FlowHead(self.encoder.d_embed, latent.dim, config=head)
        self.meta: dict[str, Any] = dict(meta or {})
        self.latent.requires_grad_(False)
        # A floor, kelvin, for paths() to take back out of the draws (remove_floor); not saved.
        self.floor_sd = 0.0

    @property
    def dim(self) -> int:
        return self.latent.dim

    @property
    def is_gaussian(self) -> bool:
        """Whether the head is the closed-form Gaussian (conditionals, exact marginals)."""
        return isinstance(self.head, GaussianHead)

    def fit_normalisation(self, batch: dict[str, Tensor]) -> None:
        self.encoder.fit_normalisation(batch)

    def forward(self, batch: dict[str, Tensor]) -> Any:
        """``q(z | x)``: a ``MultivariateNormal`` or a :class:`~chai.models.flow.ConditionalFlow`.

        Both have ``log_prob``; draw from either with :func:`draw_z`.
        """
        return self.head(self.encoder(batch))

    def nll(self, batch: dict[str, Tensor]) -> Tensor:
        """``-log q(z_true | x)`` per row, nats, in whitened coordinates."""
        return -self(batch).log_prob(batch["z"])

    @staticmethod
    def prior_nll(z: Tensor) -> Tensor:
        """``-log N(z; 0, I)`` per row: what a head that learned nothing scores.

        The reference every NLL here is read against.  Not the prior's own
        density -- the prior is not Gaussian in ``z`` -- but the density of
        the head at initialisation.
        """
        d = z.shape[-1]
        return 0.5 * (z.pow(2).sum(-1) + d * torch.log(torch.tensor(2 * torch.pi, device=z.device)))

    @torch.no_grad()
    def sample_z(self, batch: dict[str, Tensor], n: int) -> Tensor:
        """``(n, rows, dim)`` draws from ``q(z | x)``."""
        return draw_z(self(batch), n)

    def paths(self, z: Tensor, batch: dict[str, Tensor], *, clamp: bool = True
              ) -> tuple[Tensor, dict[str, Tensor]]:
        """``z`` ``(..., rows, dim)`` to paths in degrees C, with the batch's ``T0`` if the latent takes one.

        With ``floor_sd`` set, ``z`` must be ``(S, rows, dim)`` draws, and the floor is
        taken out of them (:func:`remove_floor`) before any clamp.
        """
        T0 = batch["T0"] if self.latent.present_input else None
        if not self.floor_sd:
            return self.latent.inverse(z, clamp=clamp, T0=T0)
        T, hyper = self.latent.inverse(z, T0=T0)
        if self.latent.present_input:
            # The floor was on each node's excess over the draw's own T_0 (NoisyBatches): take it out there.
            T = remove_floor(T - T[..., :1] * (torch.arange(T.shape[-1], device=T.device) > 0), self.floor_sd, first=1)
            T = T + T[..., :1] * (torch.arange(T.shape[-1], device=T.device) > 0)
        else:
            T = remove_floor(T, self.floor_sd)
        if clamp:
            T = torch.maximum(torch.minimum(T, self.latent.box_hi.to(T)), self.latent.box_lo.to(T))
            if T0 is not None:
                T = torch.maximum(T, torch.as_tensor(T0, device=T.device).to(T).expand(T.shape[:-1]).unsqueeze(-1))
        hyper["T_0"] = T[..., 0]
        return T, hyper

    @torch.no_grad()
    def sample_paths(self, batch: dict[str, Tensor], n: int, *, clamp: bool = True
                     ) -> tuple[Tensor, dict[str, Tensor]]:
        """``(n, rows, n_grid)`` paths in degrees C, and their hyperparameters.

        ``clamp=True`` projects onto the prior's box for reporting; pass
        ``False`` for anything that needs the density.
        """
        return self.paths(self.sample_z(batch, n), batch, clamp=clamp)

    @torch.no_grad()
    def standardised_residual(self, batch: dict[str, Tensor]) -> Tensor:
        """``(rows, dim)`` residual of the truth in the head's own frame: ``N(0, I)`` if calibrated.

        ``L^-1 (z_true - mu)`` for the Gaussian, for which this is the whole
        of simulation-based calibration in one number per coordinate; the
        base-space pullback for a flow.  See :func:`residual_of`.
        """
        return residual_of(self(batch), batch["z"])

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            "shape": {"n_aft": self.encoder.n_aft, "n_bins": self.encoder.n_bins,
                      "n_grain": self.encoder.n_grain, "aft_flags": list(self.encoder.aft_flags)},
            "encoder": self.encoder.config.to_config(),
            "head": self.head.config.to_config(),
            "meta": self.meta,
            "latent": _latent_payload(self.latent),
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }, path)

    @classmethod
    def load(cls, path: str | Path, map_location: Any = "cpu") -> AmortizedPosterior:
        d = torch.load(path, map_location=map_location, weights_only=False)
        latent = _latent_from_payload(d["latent"])
        shape = dict(d["shape"])
        shape["aft_flags"] = tuple(shape["aft_flags"])
        model = cls(latent, encoder=EncoderConfig(**d["encoder"]), head=head_config_from(d["head"]),
                    meta=d["meta"], **shape)
        state = dict(d["state_dict"])
        # Saved before the latent kept the path's moments: they are its node block, rebuilt above.
        for k in ("latent.path_mean", "latent.path_cov"):
            state.setdefault(k, model.state_dict()[k])
        model.load_state_dict(state)
        return model.to(map_location)


def _latent_payload(latent: Latent) -> dict[str, Any]:
    from dataclasses import asdict as _asdict
    return {
        "config": _asdict(latent.config), "hyper": [_asdict(s) for s in latent.hyper],
        "meta": latent.meta, "t_grid": latent.t_grid.cpu().numpy(),
        "box_lo": latent.box_lo.cpu().numpy(), "box_hi": latent.box_hi.cpu().numpy(),
        "mean": latent.mean.cpu().numpy(), "cov": latent.cov.cpu().numpy(),
        "path_mean": latent.path_mean.cpu().numpy(), "path_cov": latent.path_cov.cpu().numpy(),
    }


def _latent_from_payload(d: dict[str, Any]) -> Latent:
    from chai.models.latent import HyperSpec, LatentConfig
    return Latent(LatentConfig(**d["config"]), tuple(HyperSpec(**s) for s in d["hyper"]),
                  d["t_grid"], d["box_lo"], d["box_hi"], d["mean"], d["cov"], meta=d["meta"],
                  path_mean=d.get("path_mean"), path_cov=d.get("path_cov"))
