"""ctypes bindings to the compiled ground-truth forward models.

All of these are *oracles*: exact, one path per call, non-differentiable.  They
define the targets the PyTorch ports in ``chai.forward.torchimpl`` are
validated against, and they generate the simulation corpus.  Not thread-safe:
parallelise with processes.
"""
from .ketcham import (
                      KETCHAM_1999,
                      KETCHAM_2007,
                      LENGTH_BINS,
                      AFTResult,
                      calculate_aft,
                      set_time_step_pct,
)
from .madhe import calculate_madhe
from .rdaam import (
                      PREC_BEST,
                      PREC_BETTER,
                      PREC_GOOD,
                      HeDebugResult,
                      HeResult,
                      calculate_rdaam,
                      calculate_rdaam_debug,
                      eU,
)

__all__ = [
                      "KETCHAM_1999",
                      "KETCHAM_2007",
                      "LENGTH_BINS",
                      "PREC_BEST",
                      "PREC_BETTER",
                      "PREC_GOOD",
                      "AFTResult",
                      "HeDebugResult",
                      "HeResult",
                      "calculate_aft",
                      "calculate_madhe",
                      "calculate_rdaam",
                      "calculate_rdaam_debug",
                      "eU",
                      "set_time_step_pct",
]
