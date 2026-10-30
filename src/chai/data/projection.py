"""C-axis projection of confined fission-track lengths.

The heads read confined lengths **c-axis projected**: the Ketcham (2007) forward
model that made the corpus predicts the length of a track parallel to the c axis
(``doProject = 1`` in the oracle), and that is what the length histogram means to
the network.  A laboratory measures a length ``l`` and the track's angle ``phi``
to the c axis.  This module turns one into the other.

Every model here has one geometry (Donelick et al. 1999; Ketcham 2003; Ketcham
et al. 2007a).  On a polar plot of ``(l cos phi, l sin phi)``:

1. **Ellipse.**  The tracks of one annealing state lie on an ellipse with radii
   ``lc`` along the c axis and ``la`` across it, ``la = e1 lc + e0``.
2. **Accelerated length reduction.**  Once ``lc`` falls below ``lc_alr``, tracks
   above a threshold angle ``phi_alr = 90° exp(s1 (lc - lc_alr))`` shorten faster
   than the ellipse says.  They lie on a straight segment from the ellipse's point
   at ``phi_alr`` to the c-perpendicular axis, which it meets at ``I_a``.

The models differ in their constants and in ``I_a``:

* :data:`KETCHAM_2007_B03`, :data:`KETCHAM_2007_B03_A3`, :data:`KETCHAM_2007_C99`
  -- Ketcham et al. (2007a) Table 1, with ``I_a = la sin^2(phi_alr)`` (their
  Eq. 2).  Fitted to Barbarand et al. (2003) analysts 1 and 3 (5.0 M HNO3) and
  to Carlson et al. (1999) (5.5 M HNO3).  Ketcham et al. (2007b) fitted the
  annealing model the heads' forward model uses to Carlson data projected with
  the C99 set and Barbarand data projected with the analyst 1 set.
* :data:`DONELICK_1999` -- Donelick et al. (1999) Eqs. 1-3, the model the 2007
  sets refine, with ``I_a = 0.1035 phi_alr - 2.250`` µm, which can be negative:
  then no track of that state survives above ``phi_alr``.

**The default is** :data:`KETCHAM_2007_B03`.  Ketcham et al. (2015) projected
the lengths of an inter-laboratory comparison (55 analysts in 30 laboratory
groups, etchants from under 5 M to 7 M HNO3) with both 2007 sets: B03 matched
the fitted ellipses with a mean residual of -0.02 µm, C99 with -0.30 µm, and
etchant strength did not explain which set suits an analyst.  A laboratory that has calibrated its own analysts against one set
should pass that set.

The paper writes ``s0 = π / exp(lc_alr s1)`` for the scale of ``phi_alr`` in
radians.  That puts ``phi_alr`` at 180° where accelerated length reduction
begins, so it would begin only at ``phi_alr = 90°``, 1.4-1.6 µm below the tabulated
``lc_alr``.  The paper's Figure 5a is reproduced by ``π/2`` and not by ``π``
(``tests/test_projection.py``), so ``π/2`` it is: 90° at ``lc_alr``, as the
table's caption defines it.

A measured ``(l, phi)`` is projected by finding the ``lc`` whose curve passes
through it: the ellipse first, then the segment if the ellipse's answer puts the
track past ``phi_alr``.  There is no closed form, so both are solved by bisection,
exact to the tolerance and vectorised over tracks.  The uncertainty of each
projected length is not computed, because the head reads a histogram of
projected lengths and not their standard errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DONELICK_1999", "KETCHAM_2007_B03", "KETCHAM_2007_B03_A3", "KETCHAM_2007_C99", "PROJECTIONS",
           "ProjectionModel", "project_lengths", "projection_model"]


@dataclass(frozen=True)
class ProjectionModel:
    """The constants of an ellipse-plus-segment c-axis projection.

    Args:
        e1, e0: ``la = e1 * lc + e0``, µm.
        s1: rate of ``phi_alr = 90° * exp(s1 * (lc - lc_alr))``, 1/µm.
        lc_alr: the ``lc`` at which accelerated length reduction begins, µm.
        a1_slope, a1_intercept: the segment's c-perpendicular intercept as
            ``a1_slope * phi_alr_degrees + a1_intercept`` (Donelick et al. 1999).
            Both ``None`` for Ketcham et al. (2007a) Eq. 2, ``la * sin^2(phi_alr)``.
        name: for notes and reprs.
    """

    e1: float
    e0: float
    s1: float
    lc_alr: float
    a1_slope: float | None = None
    a1_intercept: float | None = None
    name: str = ""

    @property
    def lc_min(self) -> float:
        """The ``lc`` at which ``la`` reaches zero: the ellipse's lower limit."""
        return -self.e0 / self.e1

    def la(self, lc: np.ndarray) -> np.ndarray:
        return self.e1 * lc + self.e0

    def theta_alr(self, lc: np.ndarray) -> np.ndarray:
        """``phi_alr``, degrees, capped at 90."""
        return np.minimum(90.0 * np.exp(self.s1 * (np.asarray(lc, float) - self.lc_alr)), 90.0)

    def intercept(self, lc: np.ndarray, theta_alr: np.ndarray) -> np.ndarray:
        """Where the segment meets the c-perpendicular axis, µm."""
        if self.a1_slope is None:
            return self.la(lc) * np.sin(np.deg2rad(theta_alr)) ** 2
        return self.a1_slope * theta_alr + self.a1_intercept

    def ellipse_radius(self, lc: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
        """The ellipse's radius at angle ``phi`` for c-axis radius ``lc``."""
        p = np.deg2rad(phi_deg)
        la = np.maximum(self.la(lc), 1e-12)
        return 1.0 / np.sqrt((np.cos(p) / lc) ** 2 + (np.sin(p) / la) ** 2)

    def radius(self, lc: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
        """The model's track length at angle ``phi`` for c-axis length ``lc``: ellipse, then segment.

        A ray that misses the segment (a negative Donelick intercept) has length
        zero: no track of that state survives at that angle.
        """
        lc, phi = np.broadcast_arrays(np.asarray(lc, float), np.asarray(phi_deg, float))
        r = self.ellipse_radius(lc, phi)
        th = self.theta_alr(lc)
        alr = (lc < self.lc_alr) & (phi > th)
        if not alr.any():
            return r
        rp = self.ellipse_radius(lc, th)
        t = np.deg2rad(th)
        px, py = rp * np.cos(t), rp * np.sin(t)              # the ellipse's point at phi_alr
        qx, qy = np.zeros_like(px), self.intercept(lc, th)   # the c-perpendicular intercept
        ex, ey = qx - px, qy - py
        p = np.deg2rad(phi)
        dx, dy = np.cos(p), np.sin(p)
        # rho d = P + s (Q - P): Cramer's rule on the 2x2 system.
        det = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (px * ey - py * ex) / det
            s = (px * dy - py * dx) / det
        hit = np.isfinite(rho) & (rho >= 0) & (s >= -1e-12) & (s <= 1 + 1e-12)
        return np.where(alr, np.where(hit, rho, 0.0), r)


#: Ketcham et al. (2007a) Table 1, Barbarand et al. (2003) analyst 1 (5.0 M HNO3).  The default.
KETCHAM_2007_B03 = ProjectionModel(e1=1.470, e0=-8.029, s1=0.4947, lc_alr=12.241,
                                   name="Ketcham et al. (2007a), Barbarand et al. (2003) analyst 1")
#: Ketcham et al. (2007a) Table 1, Barbarand et al. (2003) analyst 3 (5.0 M HNO3).
KETCHAM_2007_B03_A3 = ProjectionModel(e1=1.457, e0=-7.905, s1=0.4773, lc_alr=12.521,
                                      name="Ketcham et al. (2007a), Barbarand et al. (2003) analyst 3")
#: Ketcham et al. (2007a) Table 1, Carlson et al. (1999) (5.5 M HNO3).
KETCHAM_2007_C99 = ProjectionModel(e1=1.632, e0=-10.880, s1=0.4221, lc_alr=13.090,
                                   name="Ketcham et al. (2007a), Carlson et al. (1999)")
#: Donelick et al. (1999) Eqs. 1-3, from the Carlson et al. (1999) experiments:
#: ``phi_alr = 0.304 exp(0.439 lc)`` degrees, which is 90° at ``lc = ln(90 / 0.304) / 0.439``.
DONELICK_1999 = ProjectionModel(e1=1.632, e0=-10.879, s1=0.439, lc_alr=float(np.log(90.0 / 0.304) / 0.439),
                                a1_slope=0.1035, a1_intercept=-2.250, name="Donelick et al. (1999)")

#: The names :func:`projection_model` and ``api.AFT(projection=...)`` accept.
PROJECTIONS: dict[str, ProjectionModel] = {
    "ketcham2007_b03": KETCHAM_2007_B03,
    "ketcham2007_b03_analyst3": KETCHAM_2007_B03_A3,
    "ketcham2007_c99": KETCHAM_2007_C99,
    "donelick1999": DONELICK_1999,
}


def projection_model(model: str | ProjectionModel) -> ProjectionModel:
    """A :class:`ProjectionModel` by name (:data:`PROJECTIONS`), or the one given."""
    if isinstance(model, ProjectionModel):
        return model
    try:
        return PROJECTIONS[model]
    except KeyError:
        raise ValueError(f"unknown projection {model!r}; one of {', '.join(PROJECTIONS)}") from None


def _bisect(f, lo: np.ndarray, hi: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """The root of ``f``, increasing on ``[lo, hi]`` with ``f(lo) <= 0 <= f(hi)``, elementwise."""
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


def project_lengths(lengths, angles, model: str | ProjectionModel = KETCHAM_2007_B03) -> np.ndarray:
    """C-axis projected lengths for measured confined lengths and their angles to the c axis.

    Args:
        lengths: measured confined track lengths, µm.
        angles: each track's angle to the crystallographic c axis, degrees in [0, 90].
        model: a :class:`ProjectionModel` or its name in :data:`PROJECTIONS`.

    Returns:
        ``lc`` per track, µm, the same shape as ``lengths``.

    A track along the c axis is its own projection.  Very short tracks at a
    nonzero angle project to just above :attr:`ProjectionModel.lc_min` (5.5 µm for
    the default), where the ellipse degenerates; the models were fitted to states
    with ``lc`` down to about 9 µm, so such tracks are outside them.
    """
    model = projection_model(model)
    L = np.asarray(lengths, dtype=float)
    phi = np.asarray(angles, dtype=float)
    if L.shape != phi.shape:
        raise ValueError(f"{L.size} lengths but {phi.size} angles: give one angle per track")
    if not np.isfinite(L).all() or (L <= 0).any():
        raise ValueError("track lengths must be positive and finite")
    if not np.isfinite(phi).all() or (phi < 0).any() or (phi > 90).any():
        raise ValueError("angles to the c axis must be in [0, 90] degrees")
    if L.size == 0:
        return L.copy()

    x = L * np.cos(np.deg2rad(phi))
    y = L * np.sin(np.deg2rad(phi))
    on_axis = y <= 1e-12 * L

    # 1. The ellipse through (x, y): 1 - (x/lc)^2 - (y/la)^2 increases in lc, from -inf at
    #    lc_min (la = 0) to +1 as lc grows.  Double the upper bracket until it holds.
    lo = np.full_like(L, model.lc_min * (1 + 1e-12))
    hi = np.maximum(L, model.lc_min) * 2.0 + 1.0

    def ellipse_gap(lc: np.ndarray) -> np.ndarray:
        la = np.maximum(model.la(lc), 1e-300)
        return 1.0 - (x / lc) ** 2 - (y / la) ** 2

    while (bad := ellipse_gap(hi) < 0).any():
        hi = np.where(bad, hi * 2.0, hi)
    lc = _bisect(ellipse_gap, lo, hi)
    lc = np.where(on_axis, L, lc)

    # 2. Tracks the ellipse's answer puts past phi_alr lie on the segment instead.  The
    #    segment is a chord inside that ellipse, so the model's length there at the ellipse's
    #    lc is at most L; at the lc where phi_alr reaches phi it is the ellipse again, at
    #    least L.  The root lies between.
    alr = (~on_axis) & (lc < model.lc_alr) & (phi > model.theta_alr(lc))
    if alr.any():
        La, pa = L[alr], phi[alr]
        lo_a = lc[alr]
        hi_a = np.minimum(model.lc_alr + np.log(pa / 90.0) / model.s1, model.lc_alr)
        hi_a = np.maximum(hi_a, lo_a)
        lc[alr] = _bisect(lambda c: model.radius(c, pa) - La, lo_a, hi_a)
    return lc
