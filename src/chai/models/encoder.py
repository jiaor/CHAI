"""From a sample to a vector: the set encoder over an aliquot.

An aliquot is three things of three kinds: a handful of scalars (the AFT age,
its error, the track count, Dpar), a 200-bin histogram (the confined-length
distribution), and a *set* of helium grains, each a 4-vector, in no order and
of no fixed size.  The encoder gives each its own path and joins them:

* **Scalars** are standardised against moments fitted from one training batch
  (:class:`Standardize`; the 0/1 presence flags are left alone) and pass
  straight into the trunk.  Standardisation lives here, in the model, because
  it is a property of what the model was trained on and must travel with it.
* **The histogram** goes through a small MLP.  It arrives as bin probabilities
  and is multiplied by the number of bins so that a flat distribution is one
  everywhere -- a fixed rescaling, not a fitted one, so a histogram with no
  counts (nothing measured) stays exactly zero.
* **The grains** go through Deep Sets (Zaheer et al. 2017): the same MLP on
  every grain, then a symmetric pool -- masked mean and masked max, side by
  side -- and the count.  Any permutation of the grain axis gives the same
  vector bit for bit, up to floating-point summation order, which
  ``tests/test_gaussian_head.py`` checks against :meth:`Aliquot.permute_grains`.
  The mask multiplies the *embeddings*, not the features, so a bias term in
  the grain MLP cannot leak padding into the pool.

* **The present-day temperature**, with ``present_input``, is one
  more scalar beside the AFT ones: ``batch["T0"]``, standardised the same way.
  It is not data but the user's surface temperature, and the head's latent
  then carries the path's excess over it (:class:`~chai.models.latent.Latent`).

Nothing here knows how many grains a corpus stores or how many bins the
histogram has: both are read off the first batch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["AliquotEncoder", "EncoderConfig", "Standardize", "mlp"]


@dataclass(frozen=True)
class EncoderConfig:
    """Widths.  Small on purpose: the input is a few hundred numbers.

    ``hist_pool`` sums each run of that many adjacent length bins before the
    histogram MLP, so the network reads a coarser grid while its input stays
    the forward model's 200 bins (5 reads 40 bins of 0.5 um).  It must divide
    the bin count.
    """

    d_embed: int = 256
    d_hidden: int = 256
    d_grain: int = 128
    d_hist: int = 64
    n_trunk: int = 3
    hist_pool: int = 1

    def to_config(self) -> dict[str, Any]:
        return asdict(self)


def mlp(d_in: int, d_hidden: int, d_out: int, n_hidden: int = 1) -> nn.Sequential:
    """``d_in -> d_hidden x n_hidden -> d_out`` with GELU and LayerNorm between."""
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.LayerNorm(d_hidden), nn.GELU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class Standardize(nn.Module):
    """``(x - mean) / sd`` per feature, with moments fitted once and stored.

    ``skip`` names features left untouched (flags).  ``fit`` takes an optional
    row mask so that absent rows -- zero-filled by construction -- do not pull
    the moments of the quantities they do not have.
    """

    def __init__(self, n_features: int, skip: tuple[int, ...] = ()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        keep = torch.ones(n_features, dtype=torch.bool)
        keep[list(skip)] = False
        self.register_buffer("keep", keep)
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: Tensor, mask: Tensor | None = None) -> None:
        x = x.reshape(-1, x.shape[-1]).to(self.mean.dtype)
        if mask is not None:
            x = x[mask.reshape(-1)]
        mean = x.mean(0)
        sd = x.std(0).clamp_min(1e-3)
        self.mean.copy_(torch.where(self.keep, mean, torch.zeros_like(mean)))
        self.sd.copy_(torch.where(self.keep, sd, torch.ones_like(sd)))
        self.fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.sd


class AliquotEncoder(nn.Module):
    """``batch -> (n, d_embed)`` for the batches :mod:`chai.train.data` makes.

    Args:
        n_aft: columns of the ``aft`` input.
        n_bins: bins of the length histogram.
        n_grain: columns of the per-grain input.
        aft_flags: which ``aft`` columns are 0/1 flags.
        config: widths.
        present_input: read ``batch["T0"]``, the present-day temperature, as
            one more input.  Set from the head's latent, not by hand.
    """

    def __init__(self, n_aft: int, n_bins: int, n_grain: int, *,
                 aft_flags: tuple[int, ...] = (), config: EncoderConfig | None = None,
                 present_input: bool = False):
        super().__init__()
        self.config = config or EncoderConfig()
        c = self.config
        self.n_aft, self.n_bins, self.n_grain = n_aft, n_bins, n_grain
        self.aft_flags = tuple(aft_flags)
        self.norm_aft = Standardize(n_aft, skip=self.aft_flags)
        self.norm_grain = Standardize(n_grain)
        if n_bins % c.hist_pool:
            raise ValueError(f"hist_pool {c.hist_pool} does not divide {n_bins} length bins")
        self.hist = mlp(n_bins // c.hist_pool, c.d_hist, c.d_hist)
        self.grain = mlp(n_grain, c.d_grain, c.d_grain)
        # Two pools and the count, then a mixer so the trunk sees one vector.
        self.pool = mlp(2 * c.d_grain + 2, c.d_grain, c.d_grain)
        self.present_input = bool(present_input)
        if self.present_input:
            self.norm_T0 = Standardize(1)
        d_in = n_aft + c.d_hist + c.d_grain + int(self.present_input)
        self.trunk = mlp(d_in, c.d_hidden, c.d_embed, n_hidden=c.n_trunk)

    @property
    def d_embed(self) -> int:
        return self.config.d_embed

    @torch.no_grad()
    def fit_normalisation(self, batch: dict[str, Tensor]) -> None:
        """Set the standardisation moments from one (large) training batch."""
        aft = batch["aft"]
        # Rows with an AFT analysis, for the AFT columns; the flags are skipped anyway.
        has_aft = aft[:, self.aft_flags[-1]] > 0.5 if self.aft_flags else None
        self.norm_aft.fit(aft, has_aft)
        self.norm_grain.fit(batch["grains"], batch["grain_mask"])
        if self.present_input:
            self.norm_T0.fit(batch["T0"].reshape(-1, 1))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        aft = self.norm_aft(batch["aft"])
        pdf = batch["length_pdf"]
        if self.config.hist_pool > 1:
            pdf = pdf.reshape(*pdf.shape[:-1], -1, self.config.hist_pool).sum(-1)
        hist = self.hist(pdf * pdf.shape[-1])

        g = self.norm_grain(batch["grains"])
        mask = batch["grain_mask"]
        m = mask.unsqueeze(-1).to(g.dtype)
        e = self.grain(g) * m                                 # (n, G, d)
        count = mask.sum(-1, keepdim=True).to(g.dtype)        # (n, 1)
        any_grain = (count > 0).to(g.dtype)
        mean = e.sum(1) / count.clamp_min(1.0)
        # Max over present grains only; an empty set pools to zero, not -inf.
        neg = torch.finfo(e.dtype).min
        mx = torch.where(mask.unsqueeze(-1), e, torch.full_like(e, neg)).amax(1)
        mx = torch.where(count > 0, mx, torch.zeros_like(mx))
        pooled = self.pool(torch.cat([mean, mx, torch.log1p(count), any_grain], dim=-1))

        parts = [aft, hist, pooled]
        if self.present_input:
            parts.append(self.norm_T0(batch["T0"].reshape(-1, 1).to(aft.dtype)))
        return self.trunk(torch.cat(parts, dim=-1))
