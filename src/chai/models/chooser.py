"""Which member answers: ``p(member | sample)``, amortised.

The family has two members, a head trained under the 100 Ma prior and one under
the 250 Ma prior, and the two priors are **different models** of the same
rock -- not a short and a long view of one (Stage 16 decision 2: net cooling is
0-200 K over each time span, so over 0-100 Ma they disagree too).  Choosing
between them is model comparison, and the quantity that does it is the
Bayes factor ``p(x | 250) / p(x | 100)``.

A classifier trained on observed samples drawn from **both corpora in equal
numbers** learns exactly that.  Its logit is ``log p(x | 250) - log p(x | 100)``
when the classes are balanced, because the optimum of the cross-entropy is the
posterior class probability, and at 50/50 prior odds the posterior odds are
the likelihood ratio.  Other prior odds shift the logit by their log and
nothing else, which is why the user's odds (a craton, an active orogen) enter
at inference and not in training.

The heads' own densities cannot stand in for this: each is a density in its
own latent, and Stage 13 measured 6.0 nats of per-draw disagreement between
two equally calibrated heads on the *same* prior, which would swamp any real
difference between the members.  The classifier never touches a density.

The network is the heads' own :class:`AliquotEncoder` (so it reads exactly
the inputs a member reads, with the same standardisation fitted on the mixed
batch) and a small MLP to one logit.  Two members only: the logit is of the
second (``members[1]``, the longer time span) against the first.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chai.models.encoder import AliquotEncoder, EncoderConfig, mlp

__all__ = ["MemberChooser"]


class MemberChooser(nn.Module):
    """``batch -> log p(x | members[1]) - log p(x | members[0])``.

    Args:
        n_aft, n_bins, n_grain, aft_flags: the encoder's input shape, as for a head.
        members: the two members' names, in logit order (``members[1]`` is the
            positive class).  Time spans in Ma for this family: ``(100, 250)``.
        encoder: widths.
        present_input: read ``batch["T0"]``, the present-day temperature, as the
            members of a family given it do (Stage 19c): the Bayes factor is then
            ``p(x | T_0, 250) / p(x | T_0, 100)``.
        d_hidden: the logit MLP's width.
        meta: provenance, saved with the weights.
    """

    def __init__(self, n_aft: int, n_bins: int, n_grain: int, *, aft_flags: tuple[int, ...] = (),
                 members: tuple[Any, Any] = (100, 250), encoder: EncoderConfig | None = None,
                 present_input: bool = False, d_hidden: int = 256, meta: dict[str, Any] | None = None):
        super().__init__()
        self.encoder = AliquotEncoder(n_aft, n_bins, n_grain, aft_flags=aft_flags, config=encoder,
                                      present_input=present_input)
        self.head = mlp(self.encoder.d_embed, d_hidden, 1, n_hidden=2)
        self.members = tuple(members)
        self.d_hidden = d_hidden
        self.meta: dict[str, Any] = dict(meta or {})
        # Start at a Bayes factor of one for every input.
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()

    @property
    def present_input(self) -> bool:
        return self.encoder.present_input

    def fit_normalisation(self, batch: dict[str, Tensor]) -> None:
        """Standardisation moments from one batch holding both members' samples."""
        self.encoder.fit_normalisation(batch)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """``(n,)`` the log Bayes factor of ``members[1]`` against ``members[0]``."""
        return self.head(self.encoder(batch)).squeeze(-1)

    def log_odds(self, batch: dict[str, Tensor], prior_odds: float = 1.0) -> Tensor:
        """``(n,)`` posterior log odds of ``members[1]`` under the given prior odds."""
        if not prior_odds > 0:
            raise ValueError(f"prior_odds must be positive, got {prior_odds}")
        return self(batch) + math.log(prior_odds)

    def prob(self, batch: dict[str, Tensor], prior_odds: float = 1.0) -> Tensor:
        """``(n,)`` ``p(members[1] | x)`` under the given prior odds."""
        return torch.sigmoid(self.log_odds(batch, prior_odds))

    def loss(self, batch: dict[str, Tensor], label: Tensor) -> Tensor:
        """``(n,)`` cross-entropy against the member each row was drawn from (1 for ``members[1]``)."""
        return F.binary_cross_entropy_with_logits(self(batch), label.to(torch.float32), reduction="none")

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        enc = self.encoder
        torch.save({"state": self.state_dict(), "members": list(self.members), "d_hidden": self.d_hidden,
                    "n_aft": enc.n_aft, "n_bins": enc.n_bins, "n_grain": enc.n_grain,
                    "aft_flags": list(enc.aft_flags), "encoder": enc.config.to_config(),
                    "present_input": enc.present_input, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: Any = "cpu") -> MemberChooser:
        d = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(d["n_aft"], d["n_bins"], d["n_grain"], aft_flags=tuple(d["aft_flags"]),
                  members=tuple(d["members"]), encoder=EncoderConfig(**d["encoder"]),
                  present_input=d.get("present_input", False), d_hidden=d["d_hidden"], meta=d["meta"])
        net.load_state_dict(d["state"])
        return net.to(map_location).eval()
