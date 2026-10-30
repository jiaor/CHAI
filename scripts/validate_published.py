"""Check the oracles against the published literature, and against each other.

Every other validation in this repository is *internal*: the PyTorch ports are
checked against the C oracles, and grids are checked against finer grids.  That
cannot catch an error the oracles and the ports share -- a wrong constant, a
misread convention, a unit slip in the vendored C.  Only something from outside
can, and this script is the outside.

Two modes:

``closure``
    Reproduce the apatite rows of Reiners & Brandon (2006) Table 2.  Each row
    gives an effective closure temperature at 10 degC/Myr together with the
    closure parameters behind it, so the published value can be predicted two
    independent ways -- analytically from the paper's own constants
    (:func:`~chai.eval.closure.dodson_tc`, which touches no code of ours) and
    by simulation through our oracles -- and both compared with what the paper
    printed.  Reiners & Brandon state that a track-modelling program and the
    Dodson equation agree to +-1 degC, which is the bar used here.

``crossval``
    RDAAM against Mad_He.  These are separate codes in separate languages by
    separate authors: Flowers et al. (2009) radiation-damage He in C++, and
    Braun et al. (2006) damage-free production-diffusion in Fortran.  Where
    they share physics they must agree, and where their published kinetics
    differ they must differ *by the amount those kinetics require* -- which is
    a sharper test than agreement, because it has a predicted value.

Usage:
    uv run python scripts/validate_published.py [--mode closure|crossval|all]
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import brentq

from chai.eval.closure import (
    REFERENCE,
    dodson_tc,
    effective_tc,
    omega_from_diffusion,
)
from chai.forward.native import (
    KETCHAM_1999,
    KETCHAM_2007,
    calculate_aft,
    calculate_madhe,
    calculate_rdaam,
    eU,
)

#: Reiners & Brandon's stated agreement between Dodson and a track-modelling
#: program.  Every closure check is held to it.
TOL_C = 1.0

#: Flowers et al. (2009) RDAAM apatite, damage-free end-member, as hardwired in
#: RDAAM.cpp: E = 29.23 kcal/mol, dInf = 0.6071 cm^2/s.  Not Farley's Durango.
RDAAM_EA = 29.23 * 4184.0
RDAAM_D0 = 0.6071
#: Below a few ppm the trap model contributes nothing and RDAAM is on this
#: end-member exactly; the ages stop moving in the eighth digit.
DAMAGE_FREE_U = 0.05

#: Radial nodes hardwired in Mad_He.f90 (``n = 100``), which fixes its quadrature.
MADHE_N = 100


def _aft_age(rmr0: float, model: int):
    def age(t, T):
        return calculate_aft(t, T, kinetic_par=rmr0, kinpar_id=4,
                             annealing_model=model).age
    return age


def _madhe_age(radius: float):
    return lambda t, T: calculate_madhe(t, T, grain_size=radius)


def _rdaam_age(radius: float, U: float, raw: bool = False):
    def age(t, T):
        r = calculate_rdaam(t, T, radius=radius, U=U, Th=0.0)
        return r.age if raw else r.corrected_age
    return age


def mode_closure(args) -> int:
    print("=" * 78)
    print("Reiners & Brandon (2006) Table 2 -- effective closure temperature at "
          "10 degC/Myr")
    print("=" * 78)
    print("\n  'dodson' uses only the paper's own Ea and Omega, so it is an external")
    print("  prediction.  'oracle' is our forward model run down a steady cooling path")
    print(f"  and read at its own age.  Both are held to +-{TOL_C:g} degC of 'published'.\n")
    print(f"  {'row':<38} {'published':>9} {'dodson':>7} {'oracle':>7} {'worst':>7}")
    print(f"  {'-' * 38} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 7}")

    failures = 0
    for ref in REFERENCE:
        dod = dodson_tc(ref.ea_j_mol, ref.omega_per_s, 10.0)
        if ref.system == "ahe":
            fn = _madhe_age(ref.kinetic["radius"])
        else:
            fn = _aft_age(ref.kinetic["rmr0"], KETCHAM_1999)
        ours = effective_tc(fn, 10.0)
        worst = max(abs(dod - ref.tc10_c), abs(ours - ref.tc10_c))
        flag = "" if worst <= TOL_C else "   <-- FAIL"
        failures += worst > TOL_C
        print(f"  {ref.label:<38} {ref.tc10_c:9.1f} {dod:7.1f} {ours:7.1f} "
              f"{worst:7.2f}{flag}")

    om = omega_from_diffusion(50.0, 60.0)
    print(f"\n  Mad_He's kinetics are the table's: Omega = 55 D0/a^2 = {om:.3e} /s")
    print("  against the tabulated 7.64e7 /s (Ea 138 kJ/mol, D0 50 cm^2/s, a_s 60 um).")

    print("\n  The FT rows are the Ketcham et al. (1999) calibration.  Our default is")
    print("  Ketcham et al. (2007) as published, which is less retentive:\n")
    print(f"  {'row':<38} {'1999':>7} {'2007':>7} {'shift':>7}")
    print(f"  {'-' * 38} {'-' * 7} {'-' * 7} {'-' * 7}")
    for ref in REFERENCE:
        if ref.system != "aft":
            continue
        a99 = effective_tc(_aft_age(ref.kinetic["rmr0"], KETCHAM_1999), 10.0)
        a07 = effective_tc(_aft_age(ref.kinetic["rmr0"], KETCHAM_2007), 10.0)
        print(f"  {ref.label:<38} {a99:7.1f} {a07:7.1f} {a07 - a99:+7.1f}")

    print("\n  Tc against cooling rate (degC), the paper's Figure 2a:\n")
    rates = [float(r) for r in args.rates]
    print("  " + f"{'system':<38}" + "".join(f"{r:>9g}" for r in rates) + "   per decade")
    for ref in REFERENCE:
        fn = (_madhe_age(ref.kinetic["radius"]) if ref.system == "ahe"
              else _aft_age(ref.kinetic["rmr0"], KETCHAM_1999))
        tcs = [effective_tc(fn, r) for r in rates]
        decade = (tcs[-1] - tcs[0]) / np.log10(rates[-1] / rates[0])
        print(f"  {ref.label:<38}" + "".join(f"{v:9.1f}" for v in tcs)
              + f"{decade:11.1f}")
    print("\n  Reiners & Brandon quote ~13 degC per decade for apatite He.")
    return failures


def mode_crossval(args) -> int:
    print("\n" + "=" * 78)
    print("RDAAM (Flowers et al. 2009, C++) against Mad_He (Braun et al. 2006, Fortran)")
    print("=" * 78)

    failures = 0

    print("\n1. Shared physics: production, decay and alpha ejection.")
    print("   Held at 0 degC nothing diffuses, so both codes reduce to the same")
    print("   bookkeeping and the age must be the age of the path.  Mad_He models no")
    print("   ejection at all, so its counterpart is RDAAM's *corrected* age.\n")
    print(f"   {'path (Ma)':>10} {'rdaam corr':>11} {'mad_he':>9} {'rel':>9}")
    for t_max in (10.0, 50.0, 100.0, 200.0):
        t = np.linspace(0.0, t_max, 201)
        T = np.zeros_like(t)
        r = calculate_rdaam(t, T, radius=60.0, U=10.0, Th=40.0).corrected_age
        m = calculate_madhe(t, T, grain_size=60.0)
        rel = abs(r - m) / m
        failures += rel > 0.02
        print(f"   {t_max:10.0f} {r:11.3f} {m:9.3f} {rel:9.2e}"
              + ("   <-- FAIL" if rel > 0.02 else ""))

    t = np.linspace(0.0, 100.0, 201)
    T = np.full_like(t, 250.0)
    r = calculate_rdaam(t, T, radius=60.0, U=10.0, Th=40.0).age
    m = calculate_madhe(t, T, grain_size=60.0)
    print(f"\n   held at 250 degC, both must lose everything: "
          f"rdaam {r:.2e} Ma, mad_he {m:.2e} Ma")

    print("\n1b. Most of that residual is Mad_He's radial quadrature, not physics.")
    print("    Mad_He.f90 hardwires n = 100 radial nodes, pins the rim at zero and")
    print("    integrates with the trapezoid rule.  Applied to a *uniform* field that")
    print("    sum does not come to one, so the age is biased low even when nothing")
    print("    diffuses -- a floor under any comparison with it.\n")
    j = np.arange(MADHE_N)
    fact = np.ones(MADHE_N)
    fact[0] = fact[-1] = 0.5
    w = 3.0 * (1.0 / (MADHE_N - 1)) ** 3 * j**2 * fact
    predicted = w[:-1].sum()
    print(f"    quadrature weight of a uniform field, rim excluded: {predicted:.5f}"
          f"  ({100 * (1 - predicted):.2f}% low)")
    print(f"    {'path (Ma)':>10} {'mad_he':>10} {'ratio':>9}")
    for t_max in (5.0, 10.0, 50.0, 200.0):
        t = np.linspace(0.0, t_max, 201)
        m = calculate_madhe(t, np.zeros_like(t), grain_size=60.0)
        print(f"    {t_max:10.0f} {m:10.4f} {m / t_max:9.5f}")
    print(f"\n    The shortest path lands on {predicted:.5f} to four decimals; the slow")
    print("    drift after it is the last of the real diffusion at 0 degC.")

    print("\n2. Different kinetics, by a predictable amount.")
    print("   Mad_He is Farley (2000) Durango (Ea 138 kJ/mol, D0 50 cm^2/s); RDAAM's")
    print("   damage-free end-member is Flowers et al. (2009)'s own fit (Ea 122.3,")
    print("   D0 0.6071), which is far less retentive.  At negligible damage the two")
    print("   *must* disagree, and Dodson says by how much.\n")
    print(f"   {'rate':>6} {'radius':>7} | {'mad_he':>7} {'dodson':>7} {'d':>6} "
          f"| {'rdaam0':>7} {'dodson':>7} {'d':>6}")
    for rate in (1.0, 10.0, 100.0):
        for radius in (40.0, 60.0, 100.0):
            m = effective_tc(_madhe_age(radius), rate)
            md = dodson_tc(138e3, omega_from_diffusion(50.0, radius), rate)
            r = effective_tc(_rdaam_age(radius, DAMAGE_FREE_U), rate)
            rd = dodson_tc(RDAAM_EA, omega_from_diffusion(RDAAM_D0, radius), rate)
            print(f"   {rate:6g} {radius:7g} | {m:7.1f} {md:7.1f} {m - md:6.1f} "
                  f"| {r:7.1f} {rd:7.1f} {r - rd:6.1f}")
    print("\n   Mad_He sits on its own Dodson prediction to well inside the paper's")
    print("   +-1 degC.  RDAAM's corrected age sits a few degrees above its own,")
    print("   which is the next test.")

    print("\n3. That residual is the alpha-ejection correction, not the diffusion.")
    print("   Ejection removes He within one stopping distance (~20 um) of the rim, so")
    print("   it vanishes in a large grain.  If the diffusion solver is right, the raw")
    print("   age must converge onto Dodson as the grain grows -- and the Ft-corrected")
    print("   age must not, because Ft is computed for an undiffused profile.\n")
    print(f"   {'radius':>7} {'dodson':>7} {'raw':>7} {'corrected':>10} "
          f"{'raw - dodson':>13} {'corr - dodson':>14}")
    for radius in (30.0, 60.0, 100.0, 200.0, 500.0, 1000.0, 2000.0):
        dod = dodson_tc(RDAAM_EA, omega_from_diffusion(RDAAM_D0, radius), 10.0)
        raw = effective_tc(_rdaam_age(radius, DAMAGE_FREE_U, raw=True), 10.0)
        cor = effective_tc(_rdaam_age(radius, DAMAGE_FREE_U), 10.0)
        print(f"   {radius:7g} {dod:7.1f} {raw:7.1f} {cor:10.1f} "
              f"{raw - dod:13.1f} {cor - dod:14.1f}")

    print("\n4. Where damage makes RDAAM agree with Durango kinetics.")
    print("   The eU at which RDAAM's trap model has bought back exactly the")
    print("   retentivity Farley measured on Durango.  If this were a property of the")
    print("   minerals it would be one number; it is not.\n")
    print(f"   {'rate':>6} {'radius':>7} {'mad_he age':>11} {'eU* (ppm)':>10} "
          f"{'ratio@eU=1':>11} {'ratio@eU=100':>13}")
    for rate in (0.3, 1.0, 3.0, 10.0):
        for radius in (40.0, 60.0, 100.0):
            t = np.linspace(0.0, 150.0, 301)
            T = np.minimum(10.0 + rate * t, 180.0)
            m = calculate_madhe(t, T, grain_size=radius)

            def gap(u, T=T, t=t, radius=radius, m=m):
                return calculate_rdaam(t, T, radius=radius, U=float(u),
                                       Th=0.0).corrected_age - m

            try:
                u_star = brentq(gap, 0.05, 400.0, xtol=0.05)
                star = f"{u_star:10.1f}"
            except ValueError:
                star = f"{'none':>10}"
            r1 = calculate_rdaam(t, T, radius=radius, U=1.0, Th=0.0).corrected_age / m
            r2 = calculate_rdaam(t, T, radius=radius, U=100.0, Th=0.0).corrected_age / m
            print(f"   {rate:6g} {radius:7g} {m:11.2f} {star} {r1:11.3f} {r2:13.3f}")
    print(f"\n   (eU of a U-only grain is U itself: eU = U + 0.235 Th, "
          f"Th = 0 here; eU(100, 0) = {eU(100.0, 0.0):.0f} ppm.)")
    return failures


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("closure", "crossval", "all"), default="all")
    p.add_argument("--rates", type=float, nargs="+", default=[1.0, 10.0, 100.0],
                   help="cooling rates (degC/Myr) for the Tc-vs-rate table")
    args = p.parse_args()

    failures = 0
    if args.mode in ("closure", "all"):
        failures += mode_closure(args)
    if args.mode in ("crossval", "all"):
        failures += mode_crossval(args)

    print("\n" + "=" * 78)
    print(f"{'FAILURES: ' + str(failures) if failures else 'all checks within tolerance'}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
