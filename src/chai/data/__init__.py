"""Priors, the simulation corpus, the observation model, and the datasets built from them.

The corpus names (``CorpusSpec``, ``reproduce``, ``simulate``) are imported on first
use: :mod:`.simulate` runs the compiled oracles, and an installed wheel, which does
not build them, must still be able to read data and run :mod:`chai.api`.
"""
import importlib
from typing import Any

from .aliquot import Aliquot
from .observe import AFTNoise, AHeNoise, ObservationModel, observe_corpus, truth_from_corpus
from .priors import GrainPrior, HistoryBatch, ThermalHistoryPrior

_NEEDS_ORACLES = ("CorpusSpec", "reproduce", "simulate")


def __getattr__(name: str) -> Any:
    if name in _NEEDS_ORACLES:
        module = importlib.import_module(f"{__name__}.simulate")
        names = {n: getattr(module, n) for n in _NEEDS_ORACLES}
        globals().update(names)
        return names[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AFTNoise",
    "AHeNoise",
    "Aliquot",
    "CorpusSpec",
    "GrainPrior",
    "HistoryBatch",
    "ObservationModel",
    "ThermalHistoryPrior",
    "observe_corpus",
    "reproduce",
    "simulate",
    "truth_from_corpus",
]
