"""Regression tests pinning the native oracles to values produced by the
earlier prototype of this project (Dec 2025), so that recompilation, flag changes or
refactors cannot silently move the ground truth.

Reference values are taken from the prototype notebooks:
  aft_age_conv.ipynb cell 13, he_rdaam_age_conv.ipynb cell 13,
  he_mad_age_conv.ipynb cell 12  (see docs/inherited.md).
"""
import numpy as np
import pytest

from chai.forward.native import KETCHAM_1999, calculate_aft, calculate_madhe, calculate_rdaam, eU


def test_aft_age_matches_prototype():
    """The prototype ran the 1999 model, so pin that explicitly.

    `calculate_aft` now defaults to Ketcham et al. (2007) -- the project-wide
    default -- so this value is only reproduced by asking for 1999 by name.
    """
    t = np.array([0.0, 1.3284607, 23.9213091, 26.86698, 72.74038])
    T = np.array([30.0, 45.791878, 24.81271, 100.09523, 120.393248])
    r = calculate_aft(t, T, kinetic_par=2.5, annealing_model=KETCHAM_1999)
    assert r.age == pytest.approx(55.19537951287892, rel=1e-12)
    assert r.oldest_age == pytest.approx(72.74038, rel=1e-5)
    assert r.length_pdf.shape == (200,)
    assert np.all(r.length_pdf >= 0)


def test_rdaam_age_matches_prototype():
    t = np.array([0.0, 1.3284607, 3.9213091, 16.86698, 72.74038])
    T = np.array([30.0, 25.791878, 15.81271, 96.09523, 120.393248])
    h = calculate_rdaam(t, T, radius=50.0, U=50.0, Th=100.0)
    # -O3 reassociation moves these at the 1e-7 level relative to the -O0 prototype.
    assert h.age == pytest.approx(7.7456819200997415, rel=1e-6)
    assert h.corrected_age == pytest.approx(10.984104175031822, rel=1e-6)
    assert h.corrected_age > h.age  # alpha-ejection correction always increases age


def test_madhe_age_matches_prototype():
    t = np.array([0.0, 1.3284607, 13.9213091, 85.86698, 102.74038])
    T = np.array([30.0, 45.791878, 105.81271, 106.09523, 120.393248])
    assert calculate_madhe(t, T, grain_size=50.0) == pytest.approx(4.732378005981445, rel=1e-6)


def test_eu():
    assert eU(50.0, 100.0) == pytest.approx(73.5)


@pytest.mark.parametrize("fn", [
    lambda t, T: calculate_aft(t, T, kinetic_par=2.0),
    lambda t, T: calculate_rdaam(t, T, 60.0, 40.0, 80.0),
    lambda t, T: calculate_madhe(t, T, 60.0),
])
class TestSharedContract:
    """Every oracle must enforce the same canonical input convention."""

    def test_rejects_decreasing_time(self, fn):
        with pytest.raises(ValueError, match="increasing"):
            fn(np.array([10.0, 0.0]), np.array([100.0, 20.0]))

    def test_rejects_shape_mismatch(self, fn):
        with pytest.raises(ValueError):
            fn(np.array([0.0, 1.0]), np.array([20.0, 50.0, 80.0]))

    def test_rejects_single_node(self, fn):
        with pytest.raises(ValueError):
            fn(np.array([0.0]), np.array([20.0]))

    def test_monotone_in_burial(self, fn):
        """A hotter history must not give an older age than a cooler one."""
        t = np.array([0.0, 20.0, 40.0, 100.0])
        cool = fn(t, np.array([10.0, 30.0, 50.0, 60.0]))
        hot = fn(t, np.array([10.0, 60.0, 100.0, 120.0]))
        age_of = lambda r: r if isinstance(r, float) else getattr(r, "age", None)
        assert age_of(hot) <= age_of(cool) + 1e-6


def test_aft_rejects_bad_annealing_model():
    """KETCH.c leaves `annealmodel` uninitialised outside {1, 2} -- guard it."""
    with pytest.raises(ValueError):
        calculate_aft(np.array([0.0, 50.0]), np.array([20.0, 100.0]), annealing_model=3)
