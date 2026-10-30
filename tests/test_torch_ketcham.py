"""The batched PyTorch Ketcham AFT model must reproduce the C oracle.

Where the two differ at the oracle's default settings, refining the *oracle's*
own quadrature grid (`set_time_step_pct`) moves it onto the torch value -- so
these tests compare against a refined oracle, not the default one.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from chai.forward.native import calculate_aft, set_time_step_pct
from chai.forward.native.ketcham import set_ket07_published
from chai.forward.torchimpl.ketcham import (
    KETCHAM_1999,
    KETCHAM_2007,
    MIN_LENGTH,
    aft_age,
    aft_forward,
    age_correction,
    initial_track_length,
    length_bin_centres,
    rmr0_from_dpar,
    rmr0_from_dpar_ket07,
    track_length_pdf,
)
from chai.paths import refine, validate

N_CTRL = 10


@pytest.fixture(autouse=True)
def _restore_oracle_globals():
    """KETCH.c keeps both of these in file-scope state."""
    yield
    from chai.forward.native import ketcham as _native_ketcham
    if _native_ketcham._lib is not None:          # never opened here (no native/): nothing to restore
        set_time_step_pct(1.0)
        set_ket07_published(True)


def _paths(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.zeros((n, N_CTRL))
    t[:, 1:] = np.sort(rng.uniform(1.0, 100.0, size=(n, N_CTRL - 1)), axis=1)
    t[:, 1:] = np.maximum.accumulate(t[:, 1:] + 1e-3 * np.arange(N_CTRL - 1), axis=1)
    T = np.empty((n, N_CTRL))
    T[:, 0] = rng.uniform(0.0, 30.0, size=n)
    T[:, 1:] = np.sort(rng.uniform(0.0, 125.0, size=(n, N_CTRL - 1)), axis=1)
    T[:, 1:] = np.maximum(T[:, 1:], T[:, 0:1])
    return t, T, rng.uniform(1.5, 3.5, size=n)


def _oracle(t, T, dpar, pct=0.05, annealing_model=2):
    set_time_step_pct(pct)
    return np.array([calculate_aft(t[i], T[i], kinetic_par=float(dpar[i]),
                                   annealing_model=annealing_model).age
                     for i in range(len(t))])


def _oracle_lengths(t, T, dpar, pct=0.05, annealing_model=2):
    """``(pdf, mean_length, std_length)`` from the oracle, on a refined grid."""
    set_time_step_pct(pct)
    out = [calculate_aft(t[i], T[i], kinetic_par=float(dpar[i]),
                         annealing_model=annealing_model) for i in range(len(t))]
    return (np.array([r.length_pdf for r in out]),
            np.array([r.mean_length for r in out]),
            np.array([r.dfmeanp for r in out]))


@pytest.mark.parametrize("model,oracle_id", [(KETCHAM_2007, 2), (KETCHAM_1999, 1)])
def test_matches_refined_oracle(model, oracle_id):
    t, T, dpar = _paths(300)
    ref = _oracle(t, T, dpar, pct=0.05, annealing_model=oracle_id)
    got = aft_age(*[torch.as_tensor(x) for x in (t, T, dpar)], n_sub=256,
                  model=model).numpy()
    ok = ref > 0.1
    rel = np.abs(got - ref) / np.maximum(ref, 1e-9)
    assert not np.isnan(got).any()
    assert np.median(rel[ok]) < 1e-4
    assert np.percentile(rel[ok], 99) < 3e-3


def test_prototype_reference_path():
    """The path pinned in tests/test_native_forward.py, via the torch model.

    That value came from the 1999 model, so ask for it by name.
    """
    t = torch.tensor([[0.0, 1.3284607, 23.9213091, 26.86698, 72.74038]], dtype=torch.float64)
    T = torch.tensor([[30.0, 45.791878, 24.81271, 100.09523, 120.393248]], dtype=torch.float64)
    age = aft_age(t, T, torch.tensor([2.5], dtype=torch.float64), n_sub=512,
                  model=KETCHAM_1999).item()
    assert age == pytest.approx(55.189, abs=0.01)


def test_converges_in_n_sub():
    """Refining the internal grid must make the answer settle, not wander."""
    t, T, dpar = _paths(200, seed=3)
    args = [torch.as_tensor(x) for x in (t, T, dpar)]
    fine = aft_age(*args, n_sub=1024, model=KETCHAM_1999)
    prev = None
    for n_sub in (64, 128, 256, 512):
        err = ((aft_age(*args, n_sub=n_sub, model=KETCHAM_1999) - fine).abs() / fine.clamp_min(0.1)).median().item()
        if prev is not None:
            assert err < prev, f"n_sub={n_sub} did not improve on the previous step"
        prev = err
    assert prev < 1e-5


def test_gradients_are_finite_and_signed():
    t, T, dpar = _paths(8, seed=5)
    tt = torch.as_tensor(t)
    TT = torch.as_tensor(T).requires_grad_(True)
    dd = torch.as_tensor(dpar).requires_grad_(True)
    aft_age(tt, TT, dd, n_sub=32, model=KETCHAM_1999).sum().backward()
    assert torch.isfinite(TT.grad).all() and torch.isfinite(dd.grad).all()
    # Heating a path can only destroy tracks, so the age never increases with T.
    assert (TT.grad <= 1e-9).all()


@pytest.mark.parametrize("model", [KETCHAM_1999, KETCHAM_2007], ids=["1999", "2007"])
def test_float32_gradients_are_finite_and_match_float64(model):
    """A float32 overflow in a masked branch used to NaN d/dT.

    The equivalent time of a population that has just died was still computed,
    overflowed float32, and the masked inf sent NaN back through ``torch.where``
    -- on 60% of prior paths under the 2007 model (docs/speed-accuracy.md).
    """
    t = [[0.0, 10.0, 40.0, 100.0]] * 3
    T = [[15.0, 45.0, 90.0, 120.0], [15.0, 45.0, 150.0, 180.0], [15.0, 30.0, 60.0, 80.0]]
    grads = {}
    for dt in (torch.float32, torch.float64):
        tt = torch.tensor(t, dtype=dt)
        TT = torch.tensor(T, dtype=dt, requires_grad=True)
        out = aft_forward(tt, TT, torch.full((3,), 2.0, dtype=dt), n_sub=16, model=model)
        (out["age"].sum() + out["mean_length"].sum()).backward()
        assert torch.isfinite(TT.grad).all(), dt
        grads[dt] = TT.grad.double()
    scale = grads[torch.float64].abs().max()
    assert (grads[torch.float32] - grads[torch.float64]).abs().max() < 1e-3 * scale


def test_gradient_matches_finite_difference_through_oracle():
    t, T, dpar = _paths(1, seed=11)
    set_time_step_pct(0.02)
    tt = torch.as_tensor(t)
    TT = torch.as_tensor(T).requires_grad_(True)
    aft_age(tt, TT, torch.as_tensor(dpar), n_sub=512, model=KETCHAM_1999).backward()
    for node in (1, 5, 9):
        h = 1e-3
        Tp, Tm = T[0].copy(), T[0].copy()
        Tp[node] += h
        Tm[node] -= h
        oracle = {"kinetic_par": float(dpar[0]), "annealing_model": 1}
        fd = (calculate_aft(t[0], Tp, **oracle).age
              - calculate_aft(t[0], Tm, **oracle).age) / (2 * h)
        assert TT.grad[0, node].item() == pytest.approx(fd, abs=2e-3)


def test_float32_is_adequate():
    """float32 error must stay far below any analytical uncertainty."""
    t, T, dpar = _paths(200, seed=9)
    a64 = aft_age(*[torch.as_tensor(x) for x in (t, T, dpar)], n_sub=64,
                  model=KETCHAM_1999)
    a32 = aft_age(*[torch.as_tensor(x, dtype=torch.float32) for x in (t, T, dpar)],
                  n_sub=64, model=KETCHAM_1999)
    rel = (a32.double() - a64).abs() / a64.clamp_min(0.1)
    assert not torch.isnan(a32).any()
    assert rel.median() < 1e-5 and rel.max() < 1e-3


def test_batching_is_independent():
    """A path's age must not depend on what else is in the batch."""
    t, T, dpar = _paths(16, seed=13)
    args = [torch.as_tensor(x) for x in (t, T, dpar)]
    full = aft_age(*args, n_sub=64, model=KETCHAM_1999)
    one = aft_age(*[a[5:6] for a in args], n_sub=64, model=KETCHAM_1999)
    assert torch.allclose(full[5:6], one, atol=1e-12)


@pytest.mark.parametrize("dpar,expected", [(1.0, 0.84), (1.75, 0.84), (5.0, 0.0)])
def test_rmr0_endpoints(dpar, expected):
    assert rmr0_from_dpar(torch.tensor([dpar])).item() == pytest.approx(expected)


def test_age_correction_is_zero_below_total_annealing():
    assert age_correction(torch.tensor([0.0, 0.3, 0.54])).abs().max().item() == 0.0


class TestRefine:
    def test_shape_and_endpoints(self):
        t = torch.tensor([[0.0, 10.0, 50.0]])
        T = torch.tensor([[20.0, 60.0, 100.0]])
        tf, Tf = refine(t, T, n_sub=4)
        assert tf.shape == (1, (3 - 1) * 4 + 1)
        assert tf[0, 0] == 0.0 and tf[0, -1] == 50.0
        assert Tf[0, 0] == 20.0 and Tf[0, -1] == 100.0
        assert torch.all(torch.diff(tf, dim=-1) > 0)

    def test_control_points_are_preserved(self):
        t = torch.tensor([[0.0, 10.0, 50.0]])
        T = torch.tensor([[20.0, 60.0, 100.0]])
        tf, Tf = refine(t, T, n_sub=5)
        assert torch.allclose(tf[0, ::5], t[0])
        assert torch.allclose(Tf[0, ::5], T[0])

    def test_validate_rejects_bad_paths(self):
        with pytest.raises(ValueError, match="present day"):
            validate(torch.tensor([[1.0, 2.0]]), torch.tensor([[20.0, 40.0]]))
        with pytest.raises(ValueError, match="increasing"):
            validate(torch.tensor([[0.0, 2.0, 1.0]]), torch.tensor([[20.0, 40.0, 50.0]]))


class TestLengthDistribution:
    """SumPopulationsKet -> CalcModelStats, ported in `track_length_pdf`."""

    def test_matches_refined_oracle(self):
        t, T, dpar = _paths(150, seed=21)
        ref_pdf, ref_mean, ref_std = _oracle_lengths(t, T, dpar, pct=0.05)
        out = aft_forward(*[torch.as_tensor(x) for x in (t, T, dpar)], n_sub=256)
        pdf = out["length_pdf"].numpy()

        # L1 distance between two densities that each integrate to 1, so this is
        # directly the fraction of probability mass in the wrong bin.
        l1 = np.abs(pdf - ref_pdf).sum(axis=1) * (20.0 / 200)
        assert not np.isnan(pdf).any()
        assert np.median(l1) < 2e-3 and l1.max() < 2e-2
        # Measured mean track lengths are quoted to ~0.01 um; stay far inside that.
        assert np.abs(out["mean_length"].numpy() - ref_mean).max() < 5e-3
        assert np.abs(out["std_length"].numpy() - ref_std).max() < 5e-3

    def test_prototype_reference_path(self):
        t = torch.tensor([[0.0, 1.3284607, 23.9213091, 26.86698, 72.74038]], dtype=torch.float64)
        T = torch.tensor([[30.0, 45.791878, 24.81271, 100.09523, 120.393248]], dtype=torch.float64)
        set_time_step_pct(0.02)
        ref = calculate_aft(t[0].numpy(), T[0].numpy(), kinetic_par=2.5)
        out = aft_forward(t, T, torch.tensor([2.5], dtype=torch.float64), n_sub=512)
        assert out["mean_length"].item() == pytest.approx(ref.mean_length, abs=0.01)
        assert np.abs(out["length_pdf"][0].numpy() - ref.length_pdf).max() < 1e-3

    def test_is_a_normalised_density(self):
        t, T, dpar = _paths(50, seed=22)
        pdf, bins = track_length_pdf(*[torch.as_tensor(x) for x in (t, T, dpar)], n_sub=64)
        assert (pdf >= 0).all()
        # normalisation is by the trapezoidal rule, matching SumPopulationsKet
        integral = ((pdf[:, 1:] + pdf[:, :-1]) / 2 * (bins[1:] - bins[:-1])).sum(dim=1)
        assert torch.allclose(integral, torch.ones_like(integral), atol=1e-12)

    def test_zero_below_the_minimum_observable_length(self):
        t, T, dpar = _paths(20, seed=23)
        pdf, bins = track_length_pdf(*[torch.as_tensor(x) for x in (t, T, dpar)], n_sub=64)
        assert pdf[:, bins < MIN_LENGTH].abs().max().item() == 0.0

    def test_totally_annealed_path_gives_zero_density(self):
        """No surviving tracks: the C leaves the pdf at zero rather than normalising."""
        t = torch.tensor([[0.0, 50.0, 100.0]], dtype=torch.float64)
        T = torch.tensor([[250.0, 260.0, 280.0]], dtype=torch.float64)
        dpar = torch.tensor([2.0], dtype=torch.float64)
        ref = calculate_aft(t[0].numpy(), T[0].numpy(), kinetic_par=2.0)
        out = aft_forward(t, T, dpar, n_sub=64)
        assert ref.length_pdf.max() == 0.0 and ref.mean_length == 0.0
        assert out["length_pdf"].abs().max().item() == 0.0
        assert out["mean_length"].item() == 0.0 and out["std_length"].item() == 0.0

    def test_converges_in_n_sub(self):
        t, T, dpar = _paths(100, seed=24)
        args = [torch.as_tensor(x) for x in (t, T, dpar)]
        fine = track_length_pdf(*args, n_sub=1024)[0]
        prev = None
        for n_sub in (64, 128, 256, 512):
            err = (track_length_pdf(*args, n_sub=n_sub)[0] - fine).abs().max().item()
            if prev is not None:
                assert err < prev, f"n_sub={n_sub} did not improve on the previous step"
            prev = err

    def test_float32_is_adequate(self):
        t, T, dpar = _paths(100, seed=25)
        args = [torch.as_tensor(x) for x in (t, T, dpar)]
        p64 = aft_forward(*args, n_sub=64)
        p32 = aft_forward(*[a.float() for a in args], n_sub=64)
        peak = p64["length_pdf"].max(dim=1).values
        rel = (p32["length_pdf"].double() - p64["length_pdf"]).abs().max(dim=1).values / peak
        assert not torch.isnan(p32["length_pdf"]).any()
        assert rel.median() < 1e-4 and rel.max() < 1e-3
        assert (p32["mean_length"].double() - p64["mean_length"]).abs().max() < 1e-3

    def test_mean_length_gradient_matches_finite_difference_through_oracle(self):
        t, T, dpar = _paths(1, seed=26)
        set_time_step_pct(0.02)
        tt = torch.as_tensor(t)
        TT = torch.as_tensor(T).requires_grad_(True)
        aft_forward(tt, TT, torch.as_tensor(dpar), n_sub=512)["mean_length"].backward()
        assert torch.isfinite(TT.grad).all()
        for node in (1, 4, 8):
            h = 1e-2
            Tp, Tm = T[0].copy(), T[0].copy()
            Tp[node] += h
            Tm[node] -= h
            fd = (calculate_aft(t[0], Tp, kinetic_par=float(dpar[0])).mean_length
                  - calculate_aft(t[0], Tm, kinetic_par=float(dpar[0])).mean_length) / (2 * h)
            assert TT.grad[0, node].item() == pytest.approx(fd, abs=1e-4)

    def test_bin_chunking_does_not_change_the_result(self):
        """`bin_chunk` is a memory knob only."""
        t, T, dpar = _paths(8, seed=27)
        args = [torch.as_tensor(x) for x in (t, T, dpar)]
        ref = track_length_pdf(*args, n_sub=64, bin_chunk=1)[0]
        for chunk in (3, 64, 200, 1000):
            got = track_length_pdf(*args, n_sub=64, bin_chunk=chunk)[0]
            assert torch.allclose(got, ref, atol=1e-14)

    def test_forward_agrees_with_the_individual_entry_points(self):
        t, T, dpar = _paths(8, seed=28)
        args = [torch.as_tensor(x) for x in (t, T, dpar)]
        out = aft_forward(*args, n_sub=64)
        assert torch.allclose(out["age"], aft_age(*args, n_sub=64), atol=1e-12)
        assert torch.allclose(out["length_pdf"], track_length_pdf(*args, n_sub=64)[0], atol=1e-14)

    def test_batching_is_independent(self):
        t, T, dpar = _paths(16, seed=29)
        args = [torch.as_tensor(x) for x in (t, T, dpar)]
        full = track_length_pdf(*args, n_sub=64)[0]
        one = track_length_pdf(*[a[5:6] for a in args], n_sub=64)[0]
        assert torch.allclose(full[5:6], one, atol=1e-14)

    def test_bins_match_the_oracle_axis(self):
        from chai.forward.native.ketcham import LENGTH_BINS
        assert np.allclose(length_bin_centres().numpy(), LENGTH_BINS)

    def test_initial_track_length(self):
        """l0vsKparDefault[1]: the Dpar-projected relation."""
        got = initial_track_length(torch.tensor([1.75, 2.5])).numpy()
        assert got == pytest.approx([0.205 * 1.75 + 16.10, 0.205 * 2.5 + 16.10])


def _legacy_ketch07():
    """Import `legacy/Ketch07.py`, the independent Ketcham (2007) prototype."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "legacy" / "Ketch07.py"
    if not path.exists():      # the public tree carries no legacy/ (scripts/public_tree.sh)
        pytest.skip("legacy/Ketch07.py is not in this checkout")
    spec = importlib.util.spec_from_file_location("_legacy_ketch07", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestKetcham2007:
    """Ketcham et al. (2007) as published -- the default model."""

    def test_age_matches_refined_oracle(self):
        t, T, dpar = _paths(200, seed=41)
        ref = _oracle(t, T, dpar, pct=0.05, annealing_model=2)
        got = aft_age(*[torch.as_tensor(x) for x in (t, T, dpar)],
                      n_sub=256, model=KETCHAM_2007).numpy()
        ok = ref > 0.1
        rel = np.abs(got - ref) / np.maximum(ref, 1e-9)
        assert not np.isnan(got).any()
        assert np.median(rel[ok]) < 1e-4
        assert np.percentile(rel[ok], 99) < 3e-3

    def test_lengths_match_refined_oracle(self):
        t, T, dpar = _paths(100, seed=42)
        ref_pdf, ref_mean, _ = _oracle_lengths(t, T, dpar, pct=0.05, annealing_model=2)
        out = aft_forward(*[torch.as_tensor(x) for x in (t, T, dpar)],
                          n_sub=256, model=KETCHAM_2007)
        pdf = out["length_pdf"].numpy()
        l1 = np.abs(pdf - ref_pdf).sum(axis=1) * (20.0 / 200)
        assert not np.isnan(pdf).any()
        assert np.median(l1) < 2e-3 and l1.max() < 3e-2
        assert np.abs(out["mean_length"].numpy() - ref_mean).max() < 3e-2

    def test_matches_the_independent_legacy_implementation(self):
        """The 2007 equations and rmr0, cross-checked outside the KETCH.c line.

        `legacy/Ketch07.py` came from the PINNs work and shares no code with
        KETCH.c, so it is the only independent check on the annealing equations,
        the power-law rmr0 and the kinetic conversion.  It is compared here with
        its own 0.13 cutoff, so that the cutoff -- the one place we deliberately
        differ from it -- is not what is being tested.

        They agree to ~0.3%.  The residual is `legacy/Ketch07.py` breaking its
        node loop at `node == 1` instead of `node == 0`, and skipping the C's
        re-check of the total-annealing cutoff after the kinetic conversion.
        """
        as_legacy = replace(KETCHAM_2007, min_obs_rcmod=0.13)
        legacy = _legacy_ketch07()
        t, T, dpar = _paths(40, seed=51)
        n_sub = 64
        # The legacy code has no interpolator: it takes an already-refined path,
        # oldest-first, in Ma and Kelvin.
        tf, Tf = refine(torch.as_tensor(t), torch.as_tensor(T), n_sub)
        ref = np.array([legacy.model_age(np.flip(tf[i].numpy()).copy(),
                                         np.flip(Tf[i].numpy()).copy() + 273.15,
                                         dpar=float(dpar[i]))
                        for i in range(len(t))])
        got = aft_age(*[torch.as_tensor(x) for x in (t, T, dpar)],
                      n_sub=n_sub, model=as_legacy).numpy()
        rel = np.abs(got - ref) / np.maximum(ref, 1e-9)
        assert np.median(rel) < 3e-3 and rel.max() < 1e-2

        # And the model we actually ship differs from it, by the cutoff alone.
        shipped = aft_age(*[torch.as_tensor(x) for x in (t, T, dpar)],
                          n_sub=n_sub, model=KETCHAM_2007).numpy()
        assert np.abs(shipped - got).max() > 1e-3 * got.max()

    def test_the_old_hybrid_differed_materially(self):
        """Pin why the port carries only the published calibration.

        Until 2026-09-19 `CalcModelLengthsKet07` paired the 2007
        `k = 1.04 - rmr0` with the *1999* exponential Dpar -> rmr0 conversion --
        a combination in no publication.  It is reachable only from the C now,
        via `set_ket07_published(False)`, and this pins the size of the
        difference so docs/inherited.md cannot go stale.
        """
        t, T, dpar = _paths(120, seed=61)
        try:
            set_ket07_published(False)
            hybrid = _oracle(t, T, dpar, pct=0.2, annealing_model=2)
        finally:
            set_ket07_published(True)
        published = _oracle(t, T, dpar, pct=0.2, annealing_model=2)
        ok = hybrid > 0.1
        rel = np.abs(published - hybrid)[ok] / hybrid[ok]
        assert np.median(rel) > 0.01, "expected a percent-level median difference"
        assert rel.max() > 0.1, "expected tens of percent in the annealed tail"
        # The published rmr0 is the larger, so it anneals more and ages younger.
        assert (published <= hybrid)[ok].mean() > 0.7

    def test_survives_extreme_paths(self):
        """The C takes a fractional power of a possibly-negative base; we clamp."""
        t = torch.tensor([[0.0, 1e-3, 1.0, 500.0]], dtype=torch.float64).repeat(4, 1)
        T = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],          # never heated
            [20.0, 20.0, 20.0, 20.0],
            [300.0, 400.0, 500.0, 600.0],  # totally annealed
            [10.0, 200.0, 15.0, 400.0],    # violent reheating
        ], dtype=torch.float64)
        out = aft_forward(t, T, torch.full((4,), 2.0, dtype=torch.float64), n_sub=32)
        for key in ("age", "length_pdf", "mean_length", "std_length"):
            assert torch.isfinite(out[key]).all(), key
        assert (out["age"] >= 0).all()

    def test_gradients_are_finite(self):
        t, T, dpar = _paths(8, seed=43)
        tt = torch.as_tensor(t)
        TT = torch.as_tensor(T).requires_grad_(True)
        dd = torch.as_tensor(dpar).requires_grad_(True)
        aft_age(tt, TT, dd, n_sub=32, model=KETCHAM_2007).sum().backward()
        assert torch.isfinite(TT.grad).all() and torch.isfinite(dd.grad).all()
        assert (TT.grad <= 1e-9).all()

    def test_the_density_correction_is_monotonic_over_its_whole_range(self):
        """Why `min_obs_rcmod` stays 0.55 for 2007, against `legacy/Ketch07.py`.

        The quadratic branch of `AgeCorrectionKet` has roots at 0.4426/0.5515,
        so extending it down to 0.13 makes it negative near c=0.5 and then climb
        to 1.234 -- a totally annealed track outweighing an unannealed one, and
        d(age)/dT > 0.  `RDAAM.cpp` pairs the same 2007 constants with 0.55.
        """
        assert KETCHAM_2007.min_obs_rcmod == 0.55
        c = torch.linspace(0.0, 1.0, 501, dtype=torch.float64)
        dens = age_correction(c, KETCHAM_2007.min_obs_rcmod)
        assert (dens >= 0).all()
        assert dens.max() <= 1.01
        # Non-decreasing apart from one known wart in the C: the quadratic and
        # linear branches do not meet at 0.765, where density steps down by
        # 0.026 (0.6509 -> 0.625).  Reproduced, not fixed.
        assert dens.diff().min() > -0.03
        assert (age_correction(torch.tensor([0.765], dtype=torch.float64), 0.55).item()
                - (1.600 * 0.765 - 0.599)) == pytest.approx(0.0259, abs=1e-3)
        # The rejected 0.13 setting breaks both bounds outright.
        bad = age_correction(c, 0.13)
        assert bad.min() < 0 and bad.max() > 1.2

    def test_the_two_rmr0_conversions_share_their_endpoints_only(self):
        dpar = torch.tensor([1.0, 1.75, 2.5, 3.5, 4.58, 5.0], dtype=torch.float64)
        exp99, pow07 = rmr0_from_dpar(dpar), rmr0_from_dpar_ket07(dpar)
        assert torch.allclose(exp99[[0, 1, 4, 5]], pow07[[0, 1, 4, 5]])
        # A higher rmr0 means a *less* resistant apatite, so over the middle of
        # the Dpar range the paper's conversion anneals more readily than the
        # 1999 one, and gives a younger age (-2.8% median over the prior).
        assert (pow07[2:4] > exp99[2:4]).all()
        assert (pow07 >= 0).all() and (pow07 <= 0.84).all()

    def test_2007_is_the_default_everywhere(self):
        t, T, dpar = _paths(8, seed=44)
        args = [torch.as_tensor(x) for x in (t, T, dpar)]
        assert torch.equal(aft_age(*args, n_sub=32),
                           aft_age(*args, n_sub=32, model=KETCHAM_2007))
        assert not torch.allclose(aft_age(*args, n_sub=32),
                                  aft_age(*args, n_sub=32, model=KETCHAM_1999))
        # ...and the oracle agrees, so the two never silently diverge.
        assert calculate_aft(t[0], T[0], kinetic_par=float(dpar[0])).age == pytest.approx(
            calculate_aft(t[0], T[0], kinetic_par=float(dpar[0]), annealing_model=2).age)
