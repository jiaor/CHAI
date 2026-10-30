"""The external checks: our oracles against numbers somebody else published.

Every other test here is internal -- ports against oracles, coarse grids against
fine ones -- and so is blind to anything the whole stack gets wrong together.
These pin the reproduction of Reiners & Brandon (2006) Table 2, which is the
only test in the suite that would fail if a vendored constant were wrong.

Full report: ``uv run python scripts/validate_published.py``, docs/validation.md.
"""
import itertools

import numpy as np
import pytest

from chai.eval.closure import (
    REFERENCE,
    dodson_tc,
    effective_tc,
    omega_from_diffusion,
)
from chai.forward.native import (
    KETCHAM_1999,
    calculate_aft,
    calculate_madhe,
    calculate_rdaam,
)

#: Reiners & Brandon's own stated agreement between the Dodson equation and a
#: track-modelling program, and so the tolerance they license us to use.
TOL_C = 1.0


def _aft_tc(rmr0, rate=10.0, model=KETCHAM_1999):
    return effective_tc(
        lambda t, T: calculate_aft(t, T, kinetic_par=rmr0, kinpar_id=4,
                                   annealing_model=model).age,
        rate,
    )


class TestDodson:
    """The analytical side, which uses no code from this repository."""

    @pytest.mark.parametrize("ref", REFERENCE, ids=lambda r: r.label)
    def test_reproduces_published_tc(self, ref):
        assert dodson_tc(ref.ea_j_mol, ref.omega_per_s, 10.0) == pytest.approx(
            ref.tc10_c, abs=TOL_C
        )

    def test_madhe_kinetics_are_the_published_ones(self):
        """Mad_He hardwires D0 = 50 cm^2/s, Ea = 138 kJ/mol at a_s = 60 um.

        Table 2 tabulates Omega = 7.64e7 /s for that row, so this is a check
        that the Fortran carries Farley (2000)'s constants and not some other
        calibration.
        """
        assert omega_from_diffusion(50.0, 60.0) == pytest.approx(7.64e7, rel=1e-3)

    def test_tc_rises_with_cooling_rate(self):
        tc = [dodson_tc(138e3, 7.64e7, r) for r in (0.1, 1.0, 10.0, 100.0)]
        assert all(b > a for a, b in itertools.pairwise(tc))

    def test_rejects_nonpositive_rate(self):
        with pytest.raises(ValueError, match="positive"):
            dodson_tc(138e3, 7.64e7, 0.0)


class TestOracleClosure:
    """The simulated side: our oracles cooled steadily, read at their own age."""

    def test_madhe_apatite_he(self):
        tc = effective_tc(lambda t, T: calculate_madhe(t, T, grain_size=60.0), 10.0)
        assert tc == pytest.approx(67.0, abs=TOL_C)

    @pytest.mark.parametrize(
        "rmr0, published",
        [(0.810, 116.0), (0.8464, 104.0), (0.1398, 177.0)],
        ids=["average-composition", "renfrew", "tioga"],
    )
    def test_aft_ketcham_1999(self, rmr0, published):
        assert _aft_tc(rmr0) == pytest.approx(published, abs=TOL_C)

    def test_rmr0_reaches_the_1999_model(self):
        """``CalcModelLengthsKet`` had no ``case RMR0:`` in its switch.

        Every rmr0 fell through to the initialiser of 0 -- the most resistant
        apatite there is -- so all three published apatites returned one
        identical closure temperature.  See docs/inherited.md.
        """
        tcs = [_aft_tc(r) for r in (0.1398, 0.810, 0.8464)]
        assert len({round(v, 3) for v in tcs}) == 3
        # More resistant apatite (lower rmr0) closes hotter.
        assert tcs[0] > tcs[1] > tcs[2]

    def test_ketcham_2007_is_less_retentive_than_1999(self):
        """The calibration change, in the units the literature reports."""
        from chai.forward.native import KETCHAM_2007

        assert _aft_tc(0.810, model=KETCHAM_2007) < _aft_tc(0.810) - 5.0

    def test_refuses_a_path_that_never_closes(self):
        with pytest.raises(ValueError, match="never closed"):
            effective_tc(lambda t, T: calculate_madhe(t, T, grain_size=60.0),
                         10.0, t_start_c=20.0)


class TestCrossValidation:
    """RDAAM against Mad_He: separate codes, separate languages, separate authors."""

    def test_agree_under_total_retention(self):
        """Held cold, both reduce to the same bookkeeping.

        Mad_He models no alpha ejection, so its counterpart is RDAAM's
        *corrected* age.  The 2% tolerance is set by Mad_He's own radial
        quadrature, not by the comparison -- see
        :meth:`test_madhe_quadrature_bias`.
        """
        t = np.linspace(0.0, 100.0, 201)
        T = np.zeros_like(t)
        r = calculate_rdaam(t, T, radius=60.0, U=10.0, Th=40.0).corrected_age
        m = calculate_madhe(t, T, grain_size=60.0)
        assert r == pytest.approx(m, rel=0.02)
        assert r == pytest.approx(100.0, rel=0.01)

    def test_agree_when_fully_open(self):
        t = np.linspace(0.0, 100.0, 201)
        T = np.full_like(t, 250.0)
        assert calculate_rdaam(t, T, radius=60.0, U=10.0, Th=40.0).age < 1e-3
        assert calculate_madhe(t, T, grain_size=60.0) < 1e-3

    def test_madhe_quadrature_bias(self):
        """Mad_He is ~1.5% low even where nothing diffuses.

        ``n = 100`` radial nodes, a rim pinned to zero and the trapezoid rule
        do not integrate a uniform field to one.  This is the floor under any
        comparison against Mad_He, so it is pinned rather than tolerated
        silently.
        """
        n = 100
        j = np.arange(n)
        fact = np.ones(n)
        fact[0] = fact[-1] = 0.5
        predicted = (3.0 * (1.0 / (n - 1)) ** 3 * j**2 * fact)[:-1].sum()
        assert predicted == pytest.approx(0.98490, abs=1e-5)

        t = np.linspace(0.0, 5.0, 201)
        measured = calculate_madhe(t, np.zeros_like(t), grain_size=60.0) / 5.0
        assert measured == pytest.approx(predicted, abs=2e-4)

    def test_rdaam_damage_free_is_not_durango(self):
        """RDAAM's zero-damage end-member is deliberately a different mineral.

        Flowers et al. (2009) fitted Ea = 122.3 kJ/mol and D0 = 0.6071 cm^2/s,
        far less retentive than Farley's Durango.  Agreement here would mean
        one of the two codes had the wrong constants.
        """
        t = np.linspace(0.0, 150.0, 301)
        T = np.minimum(10.0 + 1.0 * t, 180.0)
        m = calculate_madhe(t, T, grain_size=60.0)
        r = calculate_rdaam(t, T, radius=60.0, U=0.05, Th=0.0).corrected_age
        assert r / m == pytest.approx(0.83, abs=0.03)

    def test_damage_makes_rdaam_more_retentive(self):
        t = np.linspace(0.0, 150.0, 301)
        T = np.minimum(10.0 + 1.0 * t, 180.0)
        ages = [
            calculate_rdaam(t, T, radius=60.0, U=u, Th=0.0).corrected_age
            for u in (0.05, 10.0, 50.0, 150.0)
        ]
        assert all(b > a for a, b in itertools.pairwise(ages))
