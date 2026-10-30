"""A sample, its constraints and the run's settings, saved to one file and loaded back.

    from chai import api, inputs

    inp = inputs.Input(sample, boxes=[api.Box(t=(55, 60), T=(0, 25))], cooling_only=True, seed=3)
    inp.save("KL-07.chai.json")
    ...
    inp = inputs.load("KL-07.chai.json")
    post = inp.infer(api.load())

The file is JSON, and the web page saves and loads the same one ("Save input"
and "Load input" in its Sample box), so a sample entered in the browser can be
rerun here and the other way round::

    {"format": "chai-input", "version": 1, "chai": "0.15.0",
     "sample": {"name": ..., "aft": {...} | null, "grains": [...]},
     "constraints": {"boxes": [{"t": [young, old], "T": [lo, hi], "mode": "passes", ...}],
                     "cooling_only": false},
     "settings": {"member": "auto", "n": 2000, "seed": 0, "prior_odds": 1.0, "T_present": [5.0, 15.0]}}

``sample`` is :func:`sample_to_dict`'s, the JSON the browser port reads.  A key
left out takes its default, so a hand-written file needs only ``format`` and
``sample``.  ``T_present`` (the present-day range, °C, for a family given it)
is written only when set, so a file for a family that draws the present day is
the same as before it existed.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chai
from chai import api
from chai.constraints import Box
from chai.data.projection import PROJECTIONS

__all__ = ["FORMAT", "VERSION", "Input", "load", "sample_from_dict", "sample_to_dict"]

FORMAT = "chai-input"
VERSION = 1


def sample_from_dict(d: dict[str, Any]) -> api.Sample:
    """An :class:`api.Sample` from the JSON the browser port reads (the same field names)."""
    a = d.get("aft")
    aft = None if a is None else api.AFT(**{k: v for k, v in a.items() if v is not None})
    grains = [api.Grain(**{k: v for k, v in g.items() if v is not None}) for g in d.get("grains", ())]
    return api.Sample(aft=aft, grains=grains, name=d.get("name", ""))


def sample_to_dict(s: api.Sample) -> dict[str, Any]:
    a = s.aft
    aft = None if a is None else {
        "age": a.age, "n_spontaneous": a.n_spontaneous, "age_sd": a.age_sd, "lengths": list(a.lengths),
        "dpar": a.dpar, "angles": None if a.angles is None else list(a.angles),
        "projection": None if a.angles is None else next(k for k, p in PROJECTIONS.items() if p == a.projection),
    }
    return {"name": s.name, "aft": aft,
            "grains": [{"age": g.age, "radius": g.radius, "eU": g.eU, "age_sd": g.age_sd,
                        **({} if g.corrected else {"corrected": False})} for g in s.grains]}


def _box_to_dict(b: Box) -> dict[str, Any]:
    return {"t": list(b.t), "T": list(b.T), "mode": b.mode, "soft_sd": b.soft_sd, "label": b.label}


def _box_from_dict(d: dict[str, Any]) -> Box:
    return Box(**{k: tuple(v) if k in ("t", "T") else v for k, v in d.items()})


@dataclass(frozen=True)
class Input:
    """Everything a run reads: the sample, the constraints and the settings.

    Args:
        sample: the measurements.
        boxes: :class:`~chai.constraints.Box` constraints.
        cooling_only: keep only paths that do not reheat (:meth:`api.Model.infer`).
        member: ``"auto"``, ``100`` or ``250`` (:meth:`api.Family.infer`).
        n: paths wanted.
        seed: for the draws; ``None`` for fresh ones each run.
        prior_odds: the long member's prior odds.
        T_present: the present-day (surface) temperature range ``(lo, hi)``, °C, for
            a model given it (:meth:`api.Model.infer`); ``None`` for one that draws it.
    """

    sample: api.Sample
    boxes: Sequence[Box] = field(default=())
    cooling_only: bool = False
    member: str | int = "auto"
    n: int = 2000
    seed: int | None = 0
    prior_odds: float = 1.0
    T_present: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", tuple(self.boxes))
        if self.T_present is not None:
            if len(self.T_present) != 2:
                raise ValueError(f"T_present must be a range (lo, hi) in °C, got {self.T_present!r}")
            object.__setattr__(self, "T_present", tuple(float(v) for v in self.T_present))
        if self.member != "auto" and not isinstance(self.member, int):
            raise ValueError(f"member must be 'auto' or a time span in Ma (100 or 250), got {self.member!r}")
        if not (isinstance(self.n, int) and self.n > 0):
            raise ValueError(f"n must be a positive whole number, got {self.n!r}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be a whole number or None, got {self.seed!r}")
        if not self.prior_odds > 0:
            raise ValueError(f"prior_odds must be positive, got {self.prior_odds}")

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT, "version": VERSION, "chai": chai.__version__,
                "sample": sample_to_dict(self.sample),
                "constraints": {"boxes": [_box_to_dict(b) for b in self.boxes], "cooling_only": self.cooling_only},
                "settings": {"member": self.member, "n": self.n, "seed": self.seed, "prior_odds": self.prior_odds,
                             **({} if self.T_present is None else {"T_present": list(self.T_present)})}}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Input:
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            raise ValueError(f"not a CHAI input file (no \"format\": \"{FORMAT}\")")
        if d.get("version", VERSION) > VERSION:
            raise ValueError(f"input file version {d['version']} is newer than this chai reads ({VERSION}); "
                             "update chai-thermo")
        c, s = d.get("constraints") or {}, d.get("settings") or {}
        member = s.get("member", "auto")
        return cls(sample=sample_from_dict(d["sample"]), boxes=[_box_from_dict(b) for b in c.get("boxes", ())],
                   cooling_only=bool(c.get("cooling_only", False)),
                   member=member if member == "auto" else int(member), n=int(s.get("n", 2000)),
                   seed=s.get("seed", 0), prior_odds=float(s.get("prior_odds", 1.0)), T_present=s.get("T_present"))

    def save(self, path: str | Path) -> Path:
        """Write the file (JSON, indented); returns its path."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def infer(self, model: api.Family | api.Model) -> api.PathPosterior:
        """The run the file describes: :meth:`api.Family.infer`, or one member's (``member``, ``prior_odds`` unused)."""
        kw: dict[str, Any] = {"boxes": self.boxes, "cooling_only": self.cooling_only, "seed": self.seed,
                              "T_present": self.T_present}
        if isinstance(model, api.Family):
            kw.update(member=self.member, prior_odds=self.prior_odds)
        return model.infer(self.sample, self.n, **kw)


def load(path: str | Path) -> Input:
    """An :class:`Input` from a file written by :meth:`Input.save` or the web page."""
    return Input.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
