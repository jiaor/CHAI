"""Effective closure temperature: the one number the literature reports.

Every forward model here returns an *age*.  The thermochronology literature
mostly reports a **closure temperature**, so an age cannot be checked against a
paper directly.  This module supplies the missing conversion, in both
directions, which is what makes an external check possible at all:

* :func:`dodson_tc` -- the analytical estimate, Dodson (1973, 1979) as written
  in Reiners & Brandon (2006) eq. 7.  It depends only on published constants,
  so it knows nothing about this repository and can disagree with it.
* :func:`effective_tc` -- the *measured* closure temperature of any forward
  model, by the definition Reiners & Brandon give: cool a rock at a steady rate
  through the whole partial-retention zone, ask the model for an age, and read
  off the temperature the rock was at when that age says it closed.

The two are independent -- one is an asymptotic formula in the kinetic
constants, the other a simulation -- and Reiners & Brandon report that they
agree to within +-1 degC for the systems in :data:`REFERENCE`.  That +-1 degC is
the tolerance every check in ``scripts/validate_published.py`` is held to.

Closure temperature is only defined for steady monotonic cooling, and
:func:`effective_tc` builds exactly that path.  Nothing here is a statement
about the wiggly histories the rest of the project cares about.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "GAS_CONSTANT",
    "REFERENCE",
    "SEC_PER_MYR",
    "ClosureReference",
    "dodson_tc",
    "effective_tc",
    "omega_from_diffusion",
]

#: J / (mol K).  CODATA; the papers use 8.314 and the difference is ~1e-5 K.
GAS_CONSTANT = 8.314462618
#: Seconds in one million years, on the 365.25-day year Mad_He.f90 also uses.
SEC_PER_MYR = 3600.0 * 24.0 * 365.25e6

#: Dodson's geometry factor for a sphere.  27 is a cylinder, 8.7 a plane sheet.
SPHERE_A = 55.0


def omega_from_diffusion(d0_cm2_s: float, radius_um: float, A: float = SPHERE_A) -> float:
    """``Omega = A * D0 / a^2`` in 1/s, the frequency factor of eq. 7.

    Reiners & Brandon tabulate ``Omega`` directly for fission track, where it is
    fitted to the 50% retention isopleth rather than derived from a diffusion
    experiment, so this conversion applies to the He and Ar systems only.
    """
    return A * d0_cm2_s / (radius_um * 1e-4) ** 2


def dodson_tc(ea_j_mol: float, omega_per_s: float, rate_k_per_myr: float) -> float:
    """Closure temperature in degC for steady cooling at ``rate_k_per_myr``.

    Reiners & Brandon (2006) eq. 7, with the pressure term dropped as they
    recommend::

        dT/dt = (R Tc^2 / Ea) Omega exp(-Ea / (R Tc))

    solved for ``Tc``.  The right-hand side rises monotonically in ``Tc`` over
    any range of geological interest, so the bracketing root find is safe.
    """
    if rate_k_per_myr <= 0:
        raise ValueError(f"cooling rate must be positive, got {rate_k_per_myr}")
    dtdt = rate_k_per_myr / SEC_PER_MYR  # K/s

    def residual(tc_k: float) -> float:
        return dtdt - (GAS_CONSTANT * tc_k**2 / ea_j_mol) * omega_per_s * np.exp(
            -ea_j_mol / (GAS_CONSTANT * tc_k)
        )

    return float(brentq(residual, 100.0, 1500.0, xtol=1e-8)) - 273.15


def effective_tc(
    age_fn: Callable[[np.ndarray, np.ndarray], float],
    rate_k_per_myr: float,
    t_start_c: float = 300.0,
    t_now_c: float = 1.0,
    n_nodes: int = 1201,
) -> float:
    """Closure temperature of a forward model, measured rather than derived.

    Builds the steady cooling path ``T(t) = t_now + rate * t`` from
    ``t_start_c`` down to ``t_now_c``, hands it to ``age_fn``, and returns the
    temperature the rock was at that age -- Reiners & Brandon's definition,
    "the temperature of the dated mineral at the time indicated by the
    thermochronometer".

    ``t_start_c`` has to sit above the top of the partial-retention zone or the
    model never closes and the answer is just the start of the path; 300 degC
    clears every apatite system here.  ``t_now_c`` is held slightly above zero
    because the path has to be strictly monotonic for the bindings to accept it.
    """
    if t_start_c <= t_now_c:
        raise ValueError("t_start_c must be above t_now_c")
    t_max = (t_start_c - t_now_c) / rate_k_per_myr
    t = np.linspace(0.0, t_max, n_nodes)
    T = t_now_c + rate_k_per_myr * t
    age = float(age_fn(t, T))
    # A model that never closed returns essentially the whole path.  The margin
    # is 5% rather than machine epsilon because a fully retentive Mad_He run
    # still comes back ~1.5% short, from its own radial quadrature; a genuine
    # apatite closure temperature sits far below a sensibly chosen t_start_c,
    # so nothing real is anywhere near this boundary.
    if age >= 0.95 * t_max:
        raise ValueError(
            f"model never closed: age {age:.4g} Ma against a {t_max:.4g} Ma path; "
            f"raise t_start_c above {t_start_c:g} degC"
        )
    return t_now_c + rate_k_per_myr * age


@dataclass(frozen=True)
class ClosureReference:
    """One published row: what the literature says, and how to reproduce it.

    ``ea_j_mol`` and ``omega_per_s`` are the paper's own closure parameters, so
    :func:`dodson_tc` on them is an *independent* prediction -- it uses no code
    from this repository.  ``kinetic`` says how to ask our oracle for the same
    thing.
    """

    label: str
    source: str
    ea_j_mol: float
    omega_per_s: float
    tc10_c: float
    system: str  # "ahe" or "aft"
    kinetic: dict
    note: str = ""


#: Reiners & Brandon (2006), Table 2, apatite rows.
#:
#: The AHe row is Farley (2000) Durango: Ea 138 kJ/mol, D0 50 cm^2/s at an
#: equivalent spherical radius of 60 um.  ``Mad_He.f90`` hardwires exactly
#: those, and ``omega_from_diffusion(50.0, 60.0)`` returns 7.639e7 against the
#: table's 7.64e7 -- the same kinetics, not a coincidence.
#:
#: The FT rows are Ketcham et al. (1999), with ``rmr0`` as the table's
#: footnotes report it, and Omega fitted to the 50% retention isopleth rather
#: than to a diffusion experiment.  They are therefore checks on the **1999**
#: calibration; the project default is Ketcham et al. (2007) as published, which
#: is measurably less retentive and is reported alongside.
REFERENCE: tuple[ClosureReference, ...] = (
    ClosureReference(
        label="(U-Th)/He apatite",
        source="Farley (2000); Reiners & Brandon (2006) Table 2",
        ea_j_mol=138e3,
        omega_per_s=7.64e7,
        tc10_c=67.0,
        system="ahe",
        kinetic={"radius": 60.0},
        note="D0 = 50 cm^2/s, a_s = 60 um; Mad_He.f90's hardwired kinetics",
    ),
    ClosureReference(
        label="FT apatite, average composition",
        source="Ketcham et al. (1999); Reiners & Brandon (2006) Table 2",
        ea_j_mol=147e3,
        omega_per_s=2.05e6,
        tc10_c=116.0,
        system="aft",
        kinetic={"rmr0": 0.810},
        note="rmr0 from Carlson et al. (1999) table 4 composition",
    ),
    ClosureReference(
        label="FT Renfrew apatite, low retentivity",
        source="Ketcham et al. (1999); Reiners & Brandon (2006) Table 2",
        ea_j_mol=138e3,
        omega_per_s=5.08e5,
        tc10_c=104.0,
        system="aft",
        kinetic={"rmr0": 0.8464},
    ),
    ClosureReference(
        label="FT Tioga apatite, high retentivity",
        source="Ketcham et al. (1999); Reiners & Brandon (2006) Table 2",
        ea_j_mol=187e3,
        omega_per_s=1.57e8,
        tc10_c=177.0,
        system="aft",
        kinetic={"rmr0": 0.1398},
    ),
)
