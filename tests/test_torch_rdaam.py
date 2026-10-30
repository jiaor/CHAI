"""The batched PyTorch RDAAM model must reproduce the C++ oracle.

As with Ketcham, the two are compared after establishing which one is
converging.  Here the oracle has two knobs: `precision`, and the density of the
input path (its interpolator forces at least five steps per input segment, so
feeding it a pre-subdivided version of the *same* piecewise-linear path refines
its internal grid without changing the physics).
"""
import numpy as np
import pytest
import torch

from chai.forward.native import PREC_BETTER, PREC_GOOD, calculate_rdaam, calculate_rdaam_debug
from chai.forward.torchimpl.rdaam import grain_setup, rdaam_age
from chai.paths import refine

N_CTRL = 6

# The path and grain pinned in tests/test_native_forward.py.
PINNED_T = [0.0, 1.3284607, 3.9213091, 16.86698, 72.74038]
PINNED_TEMP = [30.0, 25.791878, 15.81271, 96.09523, 120.393248]
PINNED_GRAIN = {"radius": 50.0, "U": 50.0, "Th": 100.0}


def _paths(n, seed=0):
    """Paths and grains spanning the AHe partial-retention zone."""
    rng = np.random.default_rng(seed)
    t = np.zeros((n, N_CTRL))
    t[:, 1:] = np.sort(rng.uniform(1.0, 100.0, size=(n, N_CTRL - 1)), axis=1)
    t[:, 1:] = np.maximum.accumulate(t[:, 1:] + 1e-3 * np.arange(N_CTRL - 1), axis=1)
    T = np.empty((n, N_CTRL))
    T[:, 0] = rng.uniform(0.0, 30.0, size=n)
    mono = np.sort(rng.uniform(0.0, 120.0, size=(n, N_CTRL - 1)), axis=1)
    free = rng.uniform(0.0, 120.0, size=(n, N_CTRL - 1))
    T[:, 1:] = np.where((rng.random(n) < 0.5)[:, None], mono, free)
    T[:, 1:] = np.maximum(T[:, 1:], T[:, 0:1])
    return (t, T, rng.uniform(30.0, 100.0, size=n),
            rng.uniform(5.0, 150.0, size=n), rng.uniform(1.0, 200.0, size=n))


def _tensors(*arrays, dtype=torch.float64):
    return [torch.as_tensor(a, dtype=dtype) for a in arrays]


def _oracle(t, T, rad, U, Th):
    return np.array([calculate_rdaam(t[i], T[i], radius=float(rad[i]), U=float(U[i]),
                                     Th=float(Th[i])).age for i in range(len(t))])


class TestGrainSetup:
    """The path-independent half: geometry, composition and Ft."""

    def test_ft_matches_the_oracle(self):
        for radius, U, Th, Sm in [(50.0, 50.0, 100.0, 0.0), (30.0, 10.0, 5.0, 20.0),
                                  (100.0, 150.0, 200.0, 50.0)]:
            ref = calculate_rdaam_debug(np.array(PINNED_T), np.array(PINNED_TEMP),
                                        radius, U, Th, Sm)
            got = grain_setup(*_tensors([radius], [U], [Th], [Sm]), rdim=512).ft[0].numpy()
            # The C stores these as float, hence the 1e-7 floor.
            assert got == pytest.approx(ref.ft, abs=2e-7)

    def test_ft_is_a_retained_fraction(self):
        setup = grain_setup(*_tensors([30.0, 60.0, 120.0], [50.0] * 3, [50.0] * 3, [50.0] * 3))
        assert ((setup.ft > 0) & (setup.ft <= 1)).all()
        # Bigger grains lose a smaller fraction of their alphas.
        assert (setup.ft[:, 0].diff() > 0).all()
        # 147Sm has by far the shortest stopping distance, so it is least ejected.
        assert (setup.ft[:, 3] > setup.ft[:, 0]).all()

    def test_totals_scale_with_parent_and_volume(self):
        one = grain_setup(*_tensors([50.0], [50.0], [100.0], [0.0]))
        twice_u = grain_setup(*_tensors([50.0], [100.0], [100.0], [0.0]))
        assert twice_u.totals[0, 0] == pytest.approx(2 * one.totals[0, 0].item(), rel=1e-12)
        bigger = grain_setup(*_tensors([100.0], [50.0], [100.0], [0.0]))
        assert bigger.totals[0, 0] == pytest.approx(8 * one.totals[0, 0].item(), rel=1e-12)

    def test_rejects_tiny_grids(self):
        with pytest.raises(ValueError, match="rdim"):
            grain_setup(*_tensors([50.0], [50.0], [100.0], [0.0]), rdim=4)


class TestAgainstOracle:
    def test_pinned_path_converges_onto_the_oracle(self):
        """Refine both and they meet; at default settings they differ by ~2e-3.

        The oracle's default answer for this path uses 269 internal nodes and
        carries its own discretisation error, so the test asserts that the gap
        *closes* when each side is refined, not that they agree as they stand.
        """
        t, T = np.array([PINNED_T]), np.array([PINNED_TEMP])
        args = _tensors(t, T, [PINNED_GRAIN["radius"]], [PINNED_GRAIN["U"]],
                        [PINNED_GRAIN["Th"]])
        # Refine the oracle by handing it the same path, pre-subdivided.
        tf, Tf = refine(*_tensors(t, T), 64)
        fine = calculate_rdaam_debug(tf[0].numpy(), Tf[0].numpy(), **PINNED_GRAIN)
        assert fine.n_nodes > 1000, "expected the oracle's grid to actually refine"

        coarse = calculate_rdaam(t[0], T[0], **PINNED_GRAIN).age
        with torch.no_grad():
            got = rdaam_age(*args, n_sub=256, rdim=512, use_checkpoint=False)["age"].item()
        assert abs(got - fine.age) / fine.age < abs(coarse - fine.age) / fine.age
        assert got == pytest.approx(fine.age, rel=3e-3)

    def test_batch_median_error_meets_the_stage_3_target(self):
        t, T, rad, U, Th = _paths(60, seed=17)
        ref = _oracle(t, T, rad, U, Th)
        with torch.no_grad():
            out = rdaam_age(*_tensors(t, T, rad, U, Th), n_sub=64, rdim=512,
                            use_checkpoint=False)
        for key, want in (("age", ref), ("corrected_age", None)):
            got = out[key].numpy()
            assert not np.isnan(got).any()
            assert (got > 0).all()
        rel = np.abs(out["age"].numpy() - ref) / np.maximum(ref, 1e-9)
        assert np.median(rel) < 1e-3
        assert np.percentile(rel, 90) < 5e-3

    def test_corrected_age_exceeds_raw_age(self):
        """Alpha ejection removes He, so correcting for it can only raise the age."""
        t, T, rad, U, Th = _paths(20, seed=19)
        with torch.no_grad():
            out = rdaam_age(*_tensors(t, T, rad, U, Th), n_sub=32, use_checkpoint=False)
        assert (out["corrected_age"] > out["age"]).all()


class TestConvergence:
    def test_converges_in_n_sub(self):
        t, T, rad, U, Th = _paths(20, seed=23)
        args = _tensors(t, T, rad, U, Th)
        with torch.no_grad():
            fine = rdaam_age(*args, n_sub=128, rdim=256, use_checkpoint=False)["age"]
            prev = None
            for n_sub in (16, 32, 64):
                a = rdaam_age(*args, n_sub=n_sub, rdim=256, use_checkpoint=False)["age"]
                err = ((a - fine).abs() / fine.clamp_min(0.01)).median().item()
                if prev is not None:
                    assert err < prev, f"n_sub={n_sub} did not improve on the previous step"
                prev = err
            assert prev < 1e-3

    def test_converges_in_rdim(self):
        t, T, rad, U, Th = _paths(20, seed=29)
        args = _tensors(t, T, rad, U, Th)
        with torch.no_grad():
            fine = rdaam_age(*args, n_sub=32, rdim=1024, use_checkpoint=False)["age"]
            prev = None
            for rdim in (64, 128, 256, 512):
                a = rdaam_age(*args, n_sub=32, rdim=rdim, use_checkpoint=False)["age"]
                err = ((a - fine).abs() / fine.clamp_min(0.01)).median().item()
                if prev is not None:
                    assert err < prev, f"rdim={rdim} did not improve on the previous step"
                prev = err
            assert prev < 2e-4


class TestNumerics:
    def test_gradients_match_finite_differences_through_the_model(self):
        """Checks autograd, including through the checkpointed diffusion loop."""
        t, T, rad, U, Th = _paths(1, seed=3)
        TT = torch.as_tensor(T).requires_grad_(True)
        rest = _tensors(rad, U, Th)
        rdaam_age(torch.as_tensor(t), TT, *rest, n_sub=16, rdim=128,
                  use_checkpoint=True)["age"].backward()
        assert torch.isfinite(TT.grad).all()
        for node in (2, 4):
            h = 1e-4
            vals = []
            for sign in (+1, -1):
                Tp = T.copy()
                Tp[0, node] += sign * h
                with torch.no_grad():
                    vals.append(rdaam_age(*_tensors(t, Tp), *rest, n_sub=16, rdim=128,
                                          use_checkpoint=False)["age"].item())
            fd = (vals[0] - vals[1]) / (2 * h)
            assert TT.grad[0, node].item() == pytest.approx(fd, rel=1e-4, abs=1e-9)

    def test_heating_never_raises_the_age(self):
        t, T, rad, U, Th = _paths(4, seed=31)
        TT = torch.as_tensor(T).requires_grad_(True)
        rdaam_age(torch.as_tensor(t), TT, *_tensors(rad, U, Th), n_sub=16,
                  rdim=128)["age"].sum().backward()
        assert (TT.grad <= 1e-9).all()

    def test_gradients_flow_to_the_grain_parameters(self):
        t, T, rad, U, Th = _paths(3, seed=37)
        args = [x.requires_grad_(True) for x in _tensors(rad, U, Th)]
        rdaam_age(*_tensors(t, T), *args, n_sub=16, rdim=128)["age"].sum().backward()
        for g in args:
            assert torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0

    def test_float32_is_adequate(self):
        """Contrary to expectation, the diffusion solve survives single precision."""
        t, T, rad, U, Th = _paths(20, seed=41)
        with torch.no_grad():
            a64 = rdaam_age(*_tensors(t, T, rad, U, Th), n_sub=32, rdim=256,
                            use_checkpoint=False)["age"]
            a32 = rdaam_age(*_tensors(t, T, rad, U, Th, dtype=torch.float32), n_sub=32,
                            rdim=256, use_checkpoint=False)["age"]
        rel = (a32.double() - a64).abs() / a64.clamp_min(0.01)
        assert not torch.isnan(a32).any()
        assert rel.median() < 1e-4 and rel.max() < 1e-3

    def test_float32_gradients_are_finite_and_match_float64(self):
        """A float32 overflow in a masked branch used to NaN every d/dT.

        The annealing chain for the oldest cohort has no next interval, and its
        update read ``_lag``'s zero fill as a temperature: exp(~310), fine in
        float64, inf in float32, and the masked inf still sent NaN back through
        ``torch.where`` -- on every prior path (docs/speed-accuracy.md).
        """
        t = [[0.0, 10.0, 40.0, 100.0]] * 3
        T = [[15.0, 45.0, 90.0, 120.0], [15.0, 45.0, 150.0, 180.0], [5.0, 60.0, 200.0, 200.0]]
        grads = {}
        for dt in (torch.float32, torch.float64):
            tt, TT, rr, uu, hh = _tensors(t, T, [60.0] * 3, [40.0] * 3, [80.0] * 3, dtype=dt)
            TT.requires_grad_(True)
            rdaam_age(tt, TT, rr, uu, hh, n_sub=8, rdim=64)["corrected_age"].sum().backward()
            assert torch.isfinite(TT.grad).all(), dt
            grads[dt] = TT.grad.double()
        scale = grads[torch.float64].abs().max()
        assert (grads[torch.float32] - grads[torch.float64]).abs().max() < 1e-3 * scale

    def test_batching_is_independent(self):
        t, T, rad, U, Th = _paths(8, seed=43)
        args = _tensors(t, T, rad, U, Th)
        with torch.no_grad():
            full = rdaam_age(*args, n_sub=16, rdim=128, use_checkpoint=False)["age"]
            one = rdaam_age(*[a[3:4] for a in args], n_sub=16, rdim=128,
                            use_checkpoint=False)["age"]
        assert torch.allclose(full[3:4], one, atol=1e-12)

    def test_checkpointing_does_not_change_the_answer(self):
        t, T, rad, U, Th = _paths(4, seed=47)
        args = _tensors(t, T, rad, U, Th)
        a = rdaam_age(*args, n_sub=16, rdim=128, use_checkpoint=True)["age"]
        b = rdaam_age(*args, n_sub=16, rdim=128, use_checkpoint=False)["age"]
        assert torch.allclose(a, b, atol=1e-12)


class TestEdgeCases:
    def test_no_parent_means_no_age(self):
        t, T, rad, _, _ = _paths(3, seed=53)
        with torch.no_grad():
            out = rdaam_age(*_tensors(t, T, rad, [0.0] * 3, [0.0] * 3), n_sub=16, rdim=128)
        assert out["age"].abs().max().item() == 0.0
        assert out["corrected_age"].abs().max().item() == 0.0

    def test_survives_extreme_paths(self):
        t = torch.tensor([[0.0, 1e-3, 1.0, 500.0]], dtype=torch.float64).repeat(4, 1)
        T = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],           # never heated: age ~ path duration
            [300.0, 350.0, 400.0, 450.0],   # everything degassed
            [10.0, 250.0, 15.0, 300.0],     # violent reheating
            [20.0, 20.0, 20.0, 20.0],
        ], dtype=torch.float64)
        with torch.no_grad():
            out = rdaam_age(t, T, *_tensors([60.0] * 4, [40.0] * 4, [80.0] * 4),
                            n_sub=16, rdim=128)
        for key in ("age", "corrected_age", "total_he"):
            assert torch.isfinite(out[key]).all(), key
        assert (out["age"] >= 0).all()
        assert out["age"][1] < out["age"][0]   # the degassed grain is the youngest

    def test_oracle_is_deterministic(self):
        """RDAAM.cpp read `annealingTraps[endNode]` before writing it.

        `dmatrix` uses malloc, so that row held whatever the allocator last had.
        When it happened to be large, the first Crank-Nicolson step blew up, the
        He profile went negative, and `RDAAM_ExtractHeProfile`'s guard threw the
        whole result away -- returning ~3e-5 Ma, the bisection floor, instead of
        the true age.  Silent, intermittent, and catastrophic when it fired.
        Fixed by zeroing the row; this pins it.
        """
        t, T = np.array(PINNED_T), np.array(PINNED_TEMP)
        ages = {calculate_rdaam(t, T, **PINNED_GRAIN).age for _ in range(20)}
        assert len(ages) == 1
        assert ages.pop() == pytest.approx(7.7456819911, rel=1e-9)

    def test_oracle_precision_knob_bounds_its_own_error(self):
        """BEST is not exact: the gap to BETTER is the oracle's own uncertainty."""
        t, T = np.array(PINNED_T), np.array(PINNED_TEMP)
        best = calculate_rdaam(t, T, **PINNED_GRAIN).age
        better = calculate_rdaam(t, T, **PINNED_GRAIN, precision=PREC_BETTER).age
        good = calculate_rdaam(t, T, **PINNED_GRAIN, precision=PREC_GOOD).age
        assert good < better < best            # refining moves it monotonically
        assert abs(best - better) / best > 1e-4
