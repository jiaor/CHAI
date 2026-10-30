"""C-axis projection: the models' curves against the papers' figures, their inverse, and the API's use of them."""
from __future__ import annotations

import numpy as np
import pytest

from chai.data.projection import (
    DONELICK_1999,
    KETCHAM_2007_B03,
    KETCHAM_2007_C99,
    PROJECTIONS,
    ProjectionModel,
    project_lengths,
)

MODELS = list(PROJECTIONS.values())
ids = list(PROJECTIONS)


def _corner(m: ProjectionModel, lc: float) -> tuple[float, float, float]:
    """``(x, y)`` of the ellipse's point at ``phi_alr``, and the segment's c-perpendicular intercept."""
    th = m.theta_alr(lc)
    r = m.ellipse_radius(lc, th)
    return r * np.cos(np.deg2rad(th)), r * np.sin(np.deg2rad(th)), float(m.intercept(lc, th))


@pytest.mark.parametrize("m", MODELS, ids=ids)
def test_a_track_along_the_c_axis_is_its_own_projection(m):
    L = np.array([6.0, 9.5, 12.0, 14.5, 16.8])
    np.testing.assert_allclose(project_lengths(L, np.zeros_like(L), m), L, rtol=0, atol=0)


@pytest.mark.parametrize("m", MODELS, ids=ids)
def test_without_accelerated_reduction_a_perpendicular_track_is_the_ellipse_s_minor_radius(m):
    lc = np.array([13.5, 15.0, 16.5])                     # above every model's lc_alr: ellipse only
    np.testing.assert_allclose(project_lengths(m.la(lc), np.full_like(lc, 90.0), m), lc, atol=1e-10)


@pytest.mark.parametrize("m", MODELS, ids=ids)
def test_every_point_on_the_model_s_curve_projects_back_to_its_lc(m):
    rng = np.random.default_rng(0)
    lc = rng.uniform(7.5, 16.5, 20_000)
    phi = rng.uniform(0.0, 90.0, 20_000)
    L = m.radius(lc, phi)
    ok = L > 0.5                                          # rays past a collapsed segment have no tracks
    alr = (lc < m.lc_alr) & (phi > m.theta_alr(lc)) & ok
    assert alr.sum() > 2000                               # the segment branch is exercised
    np.testing.assert_allclose(project_lengths(L[ok], phi[ok], m), lc[ok], atol=1e-9)


@pytest.mark.parametrize("m", MODELS, ids=ids)
def test_the_model_s_length_is_continuous_at_the_threshold_angle_and_grows_with_lc(m):
    for lc in (10.5, 11.5, m.lc_alr - 0.3):
        th = m.theta_alr(lc)
        assert abs(m.radius(lc, th - 1e-7) - m.radius(lc, th + 1e-7)) < 1e-5
    lc = np.linspace(m.lc_min + 0.01, 17.0, 2000)
    for phi in np.linspace(0.0, 90.0, 91):
        r = m.radius(lc, np.full_like(lc, phi))
        assert (np.diff(r)[r[1:] > 0] > -1e-9).all(), phi


@pytest.mark.parametrize("m", MODELS, ids=ids)
def test_projection_lengthens_oblique_tracks_and_keeps_order_at_one_angle(m):
    L = np.linspace(8.0, 16.0, 50)
    for phi in (30.0, 60.0, 85.0):
        lc = project_lengths(L, np.full_like(L, phi), m)
        assert (lc > L).all() and (np.diff(lc) > 0).all()


def test_donelick_1999_matches_figure_7_of_the_paper():
    """Corners and c-perpendicular intercepts digitised from Donelick et al. (1999) Fig. 7, ±0.25 µm."""
    m = DONELICK_1999
    for lc, (px, py, a1) in {12.5: (2.8, 9.25, 5.35), 12.0: (4.8, 8.0, 3.9), 11.5: (6.2, 6.65, 2.65)}.items():
        x, y, a = _corner(m, lc)
        assert abs(x - px) < 0.25 and abs(y - py) < 0.25 and abs(a - a1) < 0.25, lc
        assert abs(m.radius(lc, 90.0) - a) < 1e-9        # the segment ends on the c-perpendicular axis
    assert abs(m.lc_alr - 12.96) < 0.01                   # "not expected for lc values > 12.96 µm"
    assert abs(0.304 * np.exp(0.439 * 11.0) - m.theta_alr(11.0)) < 1e-9   # their Eq. 2 as written


def test_ketcham_2007_matches_figure_5a_of_the_paper_and_the_scale_is_pi_over_two():
    """Figure 5a (analyst 1): two segments, corner and intercept digitised to ±0.3 µm.

    Ketcham et al. (2007a) write ``s0 = π / exp(lc_alr s1)``.  With π the corners
    of Figure 5a cannot be reproduced for any lc; with π/2 both are.
    """
    digitised = [(5.8, 6.9, 5.0), (7.0, 4.8, 2.1)]
    m = KETCHAM_2007_B03
    literal = ProjectionModel(e1=m.e1, e0=m.e0, s1=m.s1, lc_alr=m.lc_alr - np.log(2.0) / m.s1)  # π, rewritten
    lcs = np.linspace(9.0, m.lc_alr - 1e-6, 4000)
    for px, py, a1 in digitised:
        for model, should_match in ((m, True), (literal, False)):
            c = np.array([_corner(model, lc) for lc in lcs])
            miss = np.max(np.abs(c - [px, py, a1]), axis=1).min()
            assert (miss < 0.3) == should_match, (px, py, should_match, miss)


def test_the_c99_set_repeats_donelick_s_ellipse():
    assert abs(KETCHAM_2007_C99.e1 - DONELICK_1999.e1) < 1e-9 and abs(KETCHAM_2007_C99.e0 - DONELICK_1999.e0) < 2e-3


def test_models_are_chosen_by_name_and_bad_inputs_are_refused():
    L, phi = [11.0, 13.0], [60.0, 20.0]
    np.testing.assert_array_equal(project_lengths(L, phi), project_lengths(L, phi, "ketcham2007_b03"))
    assert not np.allclose(project_lengths(L, phi, "ketcham2007_c99"), project_lengths(L, phi))
    with pytest.raises(ValueError, match="unknown projection"):
        project_lengths(L, phi, "hefty")
    with pytest.raises(ValueError, match="one angle per track"):
        project_lengths([12.0, 13.0], [10.0])
    with pytest.raises(ValueError, match="0, 90"):
        project_lengths([12.0], [95.0])
    with pytest.raises(ValueError, match="positive"):
        project_lengths([-1.0], [10.0])
    assert project_lengths([], []).shape == (0,)


def test_the_api_projects_lengths_given_with_angles():
    from chai import api

    raw, phi = [11.0, 12.5, 14.0], [70.0, 45.0, 5.0]
    a = api.AFT(age=40.0, n_spontaneous=200, lengths=raw, angles=phi)
    np.testing.assert_allclose(a.projected_lengths, project_lengths(raw, phi, KETCHAM_2007_B03))
    assert a.lengths == tuple(raw) and a.projection is KETCHAM_2007_B03
    c99 = api.AFT(age=40.0, n_spontaneous=200, lengths=raw, angles=phi, projection="ketcham2007_c99")
    np.testing.assert_allclose(c99.projected_lengths, project_lengths(raw, phi, KETCHAM_2007_C99))
    plain = api.AFT(age=40.0, n_spontaneous=200, lengths=raw)
    assert plain.projected_lengths == plain.lengths       # no angles: read as already projected
    with pytest.raises(ValueError, match="one angle per track"):
        api.AFT(age=40.0, n_spontaneous=200, lengths=raw, angles=[10.0])
    with pytest.raises(ValueError, match="0, 90"):
        api.AFT(age=40.0, n_spontaneous=200, lengths=raw, angles=[10.0, 20.0, 100.0])
    with pytest.raises(ValueError, match="unknown projection"):
        api.AFT(age=40.0, n_spontaneous=200, lengths=raw, angles=phi, projection="hefty")
