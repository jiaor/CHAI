"""Constraint boxes: exact extremes on the piecewise-linear path, the two semantics, soft weights, refusals."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chai.constraints import (
    PASSES,
    WITHIN,
    Box,
    InfeasibleBoxes,
    PriorEnvelope,
    box_weights,
    check_feasible,
    cooling_weights,
    reheating,
    window_extremes,
)

T_GRID = np.linspace(0.0, 100.0, 64)
ENVELOPE = PriorEnvelope(T_GRID, np.zeros(64), np.r_[25.0, np.full(63, 200.0)], rate_max=50.0)


def _brute_extremes(T: np.ndarray, window: tuple[float, float], n: int = 20_001) -> tuple[np.ndarray, np.ndarray]:
    tt = np.linspace(*window, n)
    tt = np.r_[tt, T_GRID[(T_GRID > window[0]) & (T_GRID < window[1])]]    # the kinks, so the check is exact
    vals = np.stack([np.interp(tt, T_GRID, row) for row in T])
    return vals.min(1), vals.max(1)


def test_window_extremes_are_the_path_s_extremes_between_nodes_too():
    rng = np.random.default_rng(0)
    T = np.cumsum(rng.normal(0, 8, (200, 64)), axis=1) + 100
    for window in [(10.0, 30.0), (3.1, 3.9), (0.0, 100.0), (47.3, 47.3), (1.5873, 3.1746)]:
        lo, hi = window_extremes(T_GRID, torch.as_tensor(T), window)
        blo, bhi = _brute_extremes(T, window)
        np.testing.assert_allclose(lo.numpy(), blo, atol=1e-6)
        np.testing.assert_allclose(hi.numpy(), bhi, atol=1e-6)


def test_a_box_narrower_than_the_grid_spacing_sees_the_interpolated_path():
    # A path that ramps 0 -> 160 over one 1.59 Ma segment crosses 70-90 °C only
    # between the nodes; a node-only test would say it never passes through.
    T = np.zeros((1, 64))
    T[0, 11:] = 160.0
    t_mid = 0.5 * (T_GRID[10] + T_GRID[11])
    box = Box((t_mid - 0.1, t_mid + 0.1), (70.0, 90.0), PASSES)
    assert box_weights([box], T_GRID, torch.as_tensor(T)).item() == 1.0
    assert not ((T[0] >= 70) & (T[0] <= 90)).any()
    # ...and it does not stay within it, since the window reaches outside 70-90.
    within = Box((t_mid - 0.1, t_mid + 0.1), (70.0, 90.0), WITHIN)
    assert box_weights([within], T_GRID, torch.as_tensor(T)).item() == 0.0


def test_passes_and_within_differ_and_agree_at_a_single_time():
    T = torch.as_tensor(np.interp(T_GRID, [0, 50, 100], [10, 120, 60]))[None]
    assert box_weights([Box((40, 60), (100, 200), PASSES)], T_GRID, T).item() == 1.0
    assert box_weights([Box((40, 60), (100, 200), WITHIN)], T_GRID, T).item() == 0.0
    assert box_weights([Box((45, 55), (95, 200), WITHIN)], T_GRID, T).item() == 1.0
    for T_range in [(100, 105), (118, 125), (0, 50)]:
        a = box_weights([Box((50, 50), T_range, PASSES)], T_GRID, T).item()
        b = box_weights([Box((50, 50), T_range, WITHIN)], T_GRID, T).item()
        assert a == b


def test_a_brute_force_test_agrees_on_random_paths_and_boxes():
    rng = np.random.default_rng(1)
    T = np.clip(np.cumsum(rng.normal(0, 6, (500, 64)), axis=1) + 80, 0, 200)
    for _ in range(40):
        t0 = rng.uniform(0, 95)
        window = (t0, t0 + rng.choice([0.0, 0.7, 5.0, 20.0]))
        lo = rng.uniform(0, 180)
        rng_T = (lo, lo + rng.uniform(1, 40))
        blo, bhi = _brute_extremes(T, (window[0], min(window[1], 100.0)))
        for mode in (PASSES, WITHIN):
            box = Box((window[0], min(window[1], 100.0)), rng_T, mode)
            w = box_weights([box], T_GRID, torch.as_tensor(T)).numpy()
            want = ((bhi >= rng_T[0]) & (blo <= rng_T[1])) if mode == PASSES else ((blo >= rng_T[0]) & (bhi <= rng_T[1]))
            np.testing.assert_array_equal(w.astype(bool), want)


def test_several_boxes_multiply():
    T = torch.as_tensor(np.interp(T_GRID, [0, 50, 100], [10, 120, 60]))[None]
    a, b = Box((50, 50), (110, 130)), Box((0, 0), (50, 60))
    assert box_weights([a], T_GRID, T).item() == 1.0
    assert box_weights([a, b], T_GRID, T).item() == 0.0


def test_a_soft_box_is_gaussian_in_the_miss_and_one_inside():
    T = torch.full((1, 64), 100.0, dtype=torch.float64)
    inside = Box((10, 20), (90, 110), soft_sd=5.0)
    assert box_weights([inside], T_GRID, T).item() == 1.0
    for miss in (2.0, 5.0, 12.0):
        box = Box((10, 20), (100 + miss, 150), soft_sd=5.0)
        np.testing.assert_allclose(box_weights([box], T_GRID, T).item(), np.exp(-0.5 * (miss / 5.0) ** 2))
    # within: the worst excursion counts
    T2 = torch.as_tensor(np.interp(T_GRID, [0, T_GRID[9], 100], [100, 108, 100]))[None]    # peak on a node
    box = Box((10, 20), (90, 104), WITHIN, soft_sd=2.0)
    np.testing.assert_allclose(box_weights([box], T_GRID, T2).item(), np.exp(-0.5 * (4.0 / 2.0) ** 2), rtol=1e-6)


def test_box_arguments_are_validated():
    with pytest.raises(ValueError):
        Box((30, 20), (0, 10))
    with pytest.raises(ValueError):
        Box((10, 20), (50, 40))
    with pytest.raises(ValueError):
        Box((10, 20), (0, 10), mode="through")
    with pytest.raises(ValueError):
        Box((10, 20), (0, 10), soft_sd=-1.0)


@pytest.mark.parametrize("boxes, what", [
    ([Box((90, 120), (50, 60))], "window outside"),
    ([Box((10, 20), (210, 250))], "walls"),
    ([Box((0, 0), (40, 60))], "rate limit"),            # the present day is 0-25 °C
    ([Box((0, 5), (30, 40), WITHIN)], "walls"),           # within reaches back to t = 0
    ([Box((1, 2), (150, 160))], "rate limit"),            # 125 K in 2 Ma from the present
    ([Box((30, 30), (0, 10)), Box((31, 31), (150, 200))], "rate limit"),
    ([Box((20, 40), (0, 10), WITHIN), Box((30, 50), (100, 120), WITHIN)], "rate limit"),
])
def test_boxes_the_prior_cannot_satisfy_are_refused(boxes, what):
    with pytest.raises(InfeasibleBoxes, match=what):
        check_feasible(boxes, ENVELOPE)


@pytest.mark.parametrize("boxes", [
    [Box((10, 20), (60, 100))],
    [Box((0, 5), (0, 25), WITHIN)],
    [Box((30, 30), (0, 10)), Box((33, 33), (140, 150))],             # 130 K in 3 Ma: 43 K/Ma, allowed
    [Box((20, 40), (0, 10), PASSES), Box((30, 50), (100, 120), PASSES)],   # 20 Ma apart at the far ends
    [Box((1, 2), (150, 160), soft_sd=10.0)],                          # soft boxes are never refused
    [Box((95, 100), (0, 200))],
])
def test_boxes_the_prior_can_satisfy_pass(boxes):
    check_feasible(boxes, ENVELOPE)


def test_reheating_is_the_largest_rise_towards_the_present():
    # node 0 is the present; a path that only cools going forward in time rises with age
    T = torch.tensor([[10.0, 40.0, 80.0, 120.0],        # only cools
                      [10.0, 40.0, 30.0, 120.0],        # reheated by 10 K between nodes 2 and 1
                      [60.0, 20.0, 30.0, 50.0],         # hottest-at-present: 60 - 20 = 40 K
                      [20.0, 20.0, 20.0, 20.0]])        # flat
    np.testing.assert_allclose(reheating(T).numpy(), [0.0, 10.0, 40.0, 0.0])
    np.testing.assert_array_equal(cooling_weights(T, 10.0).numpy(), [1, 1, 0, 1])
    np.testing.assert_array_equal(cooling_weights(T, 9.9).numpy(), [1, 0, 0, 1])
    rng = np.random.default_rng(3)
    X = rng.normal(0, 5, (500, 30))
    brute = np.array([max(max(x[s] - x[t] for s in range(t)) if t else 0.0 for t in range(30)) for x in X])
    np.testing.assert_allclose(reheating(torch.as_tensor(X)).numpy(), np.maximum(brute, 0.0), atol=1e-12)
    with pytest.raises(ValueError, match="reheat_tol"):
        cooling_weights(T, -1.0)
