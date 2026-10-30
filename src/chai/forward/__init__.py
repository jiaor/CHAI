"""Forward thermochronology models.

``native``: the compiled oracles (``KETCH.c``, ``RDAAM.cpp``, ``Mad_He.f90``) --
exact, one path per call, CPU, no gradients.  ``torchimpl``: exact PyTorch ports
of the first two -- batched, any device, differentiable.  The learned
surrogates once planned here were descoped (PLAN.md Stage 4).  See ``docs/api.md``.
"""
