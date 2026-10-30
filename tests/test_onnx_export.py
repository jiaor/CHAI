"""The browser app's export: ONNX against PyTorch, every member, at fixed base draws."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from chai import api
from chai.export import (
    FIXTURE_PRESENT,
    INPUTS,
    example,
    example_inputs,
    export_family,
    fixture_samples,
    sample_from_dict,
)


@pytest.fixture(scope="module")
def family(tiny_corpus, tiny_corpus_long) -> api.Family:
    """A tiny family whose members draw their own present day (a GP prior on the conftest corpora), randomly
    initialised and jittered so every part of the graph is exercised: the export's path for such members,
    which the shipped family does not take, kept working for families trained later."""
    import json
    from dataclasses import asdict

    import h5py

    from chai.data.observe import ObservationModel
    from chai.data.priors import GrainPrior
    from chai.models.chooser import MemberChooser
    from chai.models.encoder import EncoderConfig
    from chai.models.flow import FlowConfig
    from chai.models.gaussian import AmortizedPosterior
    from chai.models.latent import Latent
    from chai.train.data import AFT_FLAGS, AFT_INPUTS, GRAIN_INPUTS, EffortPrior

    g = torch.Generator().manual_seed(3)
    enc = EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=8)
    shape = {"n_aft": len(AFT_INPUTS), "n_bins": api.N_BINS, "n_grain": len(GRAIN_INPUTS), "aft_flags": AFT_FLAGS}
    gp = GrainPrior()
    grain = {"radius_range": gp.radius_range, "U_range": gp.U_range, "th_u_ratio_range": gp.th_u_ratio_range,
             "dpar_range": gp.dpar_range}

    def jitter(net):
        with torch.no_grad():
            for prm in net.parameters():
                prm.add_(0.05 * torch.randn(prm.shape, generator=g))
        return net.eval()

    members = {}
    for h, corpus in ((100, tiny_corpus), (250, tiny_corpus_long)):
        with h5py.File(corpus, "r") as f:
            prior = json.loads(f.attrs["prior_config"])
        net = AmortizedPosterior(Latent.fit(corpus), encoder=enc, head=FlowConfig(n_layers=2, n_bins=4, d_hidden=16, base="student_t"),
                                 meta={"config": {"effort": asdict(EffortPrior())}}, **shape)
        members[h] = api.Model(jitter(net), ObservationModel(), prior, grain, g_max=5)
    return api.Family(members, jitter(MemberChooser(**shape, members=(100, 250), encoder=enc, d_hidden=16)))


@pytest.fixture(scope="module")
def exported(family, tmp_path_factory):
    out = tmp_path_factory.mktemp("web-models")
    export_family(family, out)
    return out


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("n_grains", [1, 3, 14])
def test_every_member_s_graphs_match_pytorch_draw_for_draw(family, exported, n_grains):
    for h, m in family.members.items():
        args = example_inputs(m, n_grains)
        with torch.no_grad():
            q = m.net(dict(zip(INPUTS, args)))
            u = m.net.head.base_sample((1000, q.loc.shape[-1]), torch.Generator().manual_seed(h + n_grains), like=q.loc)
            T, hyper = m.net.latent.inverse(q.push(u).double(), clamp=True)
        embed, loc, L = _session(exported / f"condition{h}.onnx").run(None, {k: a.numpy() for k, a in zip(INPUTS, args)})
        np.testing.assert_allclose(embed, q.embed.numpy(), atol=1e-5)
        np.testing.assert_allclose(loc, q.loc.numpy(), atol=1e-5)
        T2, h2 = _session(exported / f"sampler{h}.onnx").run(None, {"u": u.numpy(), "embed": embed, "loc": loc,
                                                                     "scale_tril": L})
        assert T2.shape == (1000, m.t.size)
        # float32 latent inverse against the package's float64 one: hundredths of a kelvin at worst
        assert np.abs(T2 - T.numpy()).max() < 0.03
        assert np.quantile(np.abs(T2 - T.numpy()), 0.99) < 2e-3
        ref = np.stack([hyper[k].numpy() for k in m.net.latent.hyper_names], -1)
        # 1% relative, with a floor for values near zero: a drift of 5e-7 is off by 9e-9 in float32
        np.testing.assert_allclose(h2, ref, rtol=1e-2, atol=1e-6)
        # the clamp is the prior's walls
        assert (T2 >= m.net.latent.box_lo.numpy() - 1e-4).all() and (T2 <= m.net.latent.box_hi.numpy() + 1e-4).all()


def test_the_chooser_matches_pytorch(family, exported):
    s = _session(exported / "chooser.onnx")
    for n_grains in (1, 5):
        args = example_inputs(family.long, n_grains)
        with torch.no_grad():
            ref = family.chooser(dict(zip(INPUTS, args))).numpy()
        np.testing.assert_allclose(s.run(None, {k: a.numpy() for k, a in zip(INPUTS, args)})[0], ref, atol=1e-4)


def test_meta_carries_what_the_port_reads(family, exported):
    meta = json.loads((exported / "meta.json").read_text())
    assert meta["package"] == "chai-thermo" and meta["weights"] == "float32"
    for h, m in family.members.items():
        mm = meta["members"][str(h)]
        assert len(mm["df"]) == mm["dim"] == m.net.latent.dim
        np.testing.assert_allclose(mm["t"], m.t)
        assert mm["n_lengths"] == list(m.effort["n_lengths"]) and mm["g_max"] == m.g_max
        assert mm["present_input"] is False and mm["T_present_range"] is None
        assert (exported / mm["condition"]).exists() and (exported / mm["sampler"]).exists()
    assert meta["observation"]["he_rel_sd"] == pytest.approx(family[100].observation.he.rel_sd)
    assert set(meta["projections"]) == {"ketcham2007_b03", "ketcham2007_b03_analyst3", "ketcham2007_c99",
                                        "donelick1999"}
    assert set(meta["bytes"]) == {p.name for p in exported.glob("*.onnx")}
    ex = json.loads((exported / "example.json").read_text())
    assert {e["prior"] for e in ex["examples"]} == set(family.members)
    for e in ex["examples"]:
        assert sample_from_dict(e["sample"]).aft is not None and len(e["truth"]["t"]) == family[e["prior"]].t.size


def test_the_spline_s_knot_count_is_searchsorted():
    # the export's one change to the model code: the same bin index, without searchsorted
    g = torch.Generator().manual_seed(0)
    knots = torch.sort(torch.randn(500, 9, generator=g), dim=-1).values
    x = torch.randn(500, generator=g) * 2
    x[:9] = knots[:9, 3]                                   # exactly on a knot
    count = (knots <= x.unsqueeze(-1)).sum(-1, keepdim=True) - 1
    ss = torch.searchsorted(knots, x.unsqueeze(-1), right=True) - 1
    assert torch.equal(count, ss)


def test_fixture_refusals_are_refusals():
    shipped = api.load()
    for case in fixture_samples(shipped):
        if not case["name"].startswith("err_"):
            continue
        with pytest.raises(ValueError):
            if "raw" in case:                             # a sample the API refuses to build
                shipped.long.to_aliquot(sample_from_dict(case["raw"]))
            else:                                         # a present-day range the API refuses
                shipped.infer(sample_from_dict(case["sample"]), 200, T_present=case["T_present"])


# ---------------------------------------------------------------------------
# A small family given the present day, randomly initialised on the shipped corpora's latents
# ---------------------------------------------------------------------------

def _present_family() -> api.Family:
    """Two present_input members on data/latent_v22_{100,250}.pt and a present_input chooser, weights jittered.

    Small widths, so the export takes seconds; every weight moved off its initialisation (the couplings start at
    the identity) and T0's standardisation set, so each part of the graph is exercised.
    """
    import yaml

    from chai.data.observe import ObservationModel
    from chai.models.chooser import MemberChooser
    from chai.models.encoder import EncoderConfig
    from chai.models.flow import FlowConfig
    from chai.models.gaussian import AmortizedPosterior
    from chai.models.latent import Latent
    from chai.train.data import AFT_FLAGS, AFT_INPUTS, GRAIN_INPUTS

    g = torch.Generator().manual_seed(5)
    enc = EncoderConfig(d_embed=32, d_hidden=32, d_grain=16, d_hist=16, n_trunk=2, hist_pool=2)
    shape = {"n_aft": len(AFT_INPUTS), "n_bins": api.N_BINS, "n_grain": len(GRAIN_INPUTS), "aft_flags": AFT_FLAGS}
    obs = ObservationModel.from_config(yaml.safe_load(api._resolve("configs/observation_v2.yaml").read_text()))

    def jitter(net):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
            net.encoder.norm_T0.mean.fill_(12.5)
            net.encoder.norm_T0.sd.fill_(7.0)
        return net.eval()

    members = {}
    for h, cfg in ((100, "configs/prior_v22.yaml"), (250, "configs/prior_v22_250.yaml")):
        prior = yaml.safe_load(api._resolve(cfg).read_text())
        latent = Latent.load(api._resolve(f"data/latent_v22_{h}.pt"))
        net = AmortizedPosterior(latent, encoder=enc, meta={"config": {"effort": {"n_lengths": (40, 150)}}},
                                 head=FlowConfig(n_layers=4, d_hidden=32, affine_hidden=32, base="student_t"), **shape)
        members[h] = api.Model(jitter(net), obs, prior["history"], prior["grain"], 12)
    chooser = MemberChooser(**shape, members=(100, 250), encoder=enc, present_input=True, d_hidden=32)
    return api.Family(members, jitter(chooser))


@pytest.fixture(scope="module")
def present_family():
    try:
        return _present_family()
    except FileNotFoundError as err:
        pytest.skip(f"no corpus latent in this checkout: {err}")


@pytest.fixture(scope="module")
def present_exported(present_family, tmp_path_factory):
    out = tmp_path_factory.mktemp("web-models-present")
    export_family(present_family, out)
    return out


def _clamp_present(T, T0, latent):
    """The page's clamp of a present sampler's paths: to the walls, then to ``T >= T0``."""
    return np.maximum(np.maximum(np.minimum(T, latent.box_hi.numpy()), latent.box_lo.numpy()), T0[:, None])


def _rows(batch, T0):
    return {**{k: v.expand(len(T0), *v.shape[1:]) for k, v in batch.items()}, "T0": T0}


@pytest.mark.parametrize("n_grains", [1, 3, 14])
def test_a_present_member_s_graphs_match_pytorch_draw_for_draw(present_family, present_exported, n_grains):
    for h, m in present_family.members.items():
        args = example_inputs(m, n_grains)
        b = dict(zip(INPUTS, args))
        n = 600
        # T0 per draw, as Model.infer draws it, and the trained range's two ends
        T0 = torch.empty(n).uniform_(8.0, 12.0, generator=torch.Generator().manual_seed(h))
        T0[:2] = torch.tensor(m.T_present_range)
        with torch.no_grad():
            q = m.net(_rows(b, T0))
            u = m.net.head.base_sample((n, m.net.latent.dim), torch.Generator().manual_seed(h + n_grains), like=q.loc)
            T, _ = m.net.latent.inverse(q.push(u).double(), clamp=True, T0=T0.double())
        (feat,) = _session(present_exported / f"condition{h}.onnx").run(None, {k: a.numpy() for k, a in zip(INPUTS, args)})
        assert feat.shape[0] == 1
        s = _session(present_exported / f"sampler{h}.onnx")
        assert [o.name for o in s.get_outputs()] == ["T"]              # a pulse prior has no hyperparameters
        (T2,) = s.run(None, {"u": u.numpy(), "feat": feat, "T0": T0.numpy()})
        assert T2.shape == (n, m.t.size)
        # unclamped (the page takes a floor out of the batch first), then clamped as web/js/floor.js does
        L = m.net.latent
        assert (T2 > L.box_hi.numpy() + 1).any() or (T2 < np.maximum(L.box_lo.numpy(), T0.numpy()[:, None]) - 1).any()
        T2 = _clamp_present(T2, T0.numpy(), L)
        err = np.abs(T2 - T.numpy())
        assert err.max() < 0.03 and np.quantile(err, 0.99) < 2e-3, (err.max(), np.quantile(err, 0.99))
        # the present day is the input, and every node is at or above it and inside the walls
        np.testing.assert_allclose(T2[:, 0], T0.numpy(), atol=1e-5)
        assert (T2 >= T0.numpy()[:, None] - 1e-5).all()
        assert (T2 <= m.net.latent.box_hi.numpy() + 1e-4).all()


def test_a_present_chooser_matches_pytorch(present_family, present_exported):
    s = _session(present_exported / "chooser.onnx")
    fam = present_family
    for n_grains in (1, 5):
        args = example_inputs(fam.long, n_grains)
        T0 = np.linspace(*FIXTURE_PRESENT, api.N_PRESENT_BF).astype(np.float32)
        with torch.no_grad():
            ref = fam.chooser(_rows(dict(zip(INPUTS, args)), torch.as_tensor(T0))).numpy()
        got = s.run(None, {**{k: a.numpy() for k, a in zip(INPUTS, args)}, "T0": T0})[0]
        assert got.shape == (api.N_PRESENT_BF,)
        np.testing.assert_allclose(got, ref, atol=1e-5)
    # the page's mean over the range is Family.log_bayes_factor's
    sample = sample_from_dict(example(fam, n=1)["examples"][0]["sample"])
    obs, _ = fam.long.to_aliquot(sample)
    b = fam.long.batch(obs)
    got = s.run(None, {**{k: v.numpy() for k, v in b.items()}, "T0": T0})[0].mean()
    assert got == pytest.approx(fam.log_bayes_factor(sample, FIXTURE_PRESENT), abs=1e-5)


def test_present_meta_and_examples(present_family, present_exported):
    meta = json.loads((present_exported / "meta.json").read_text())
    assert meta["n_present_bf"] == api.N_PRESENT_BF and meta["min_present_range"] == api.MIN_PRESENT_RANGE
    assert meta["chooser"]["present_input"] is True
    assert meta["observation"]["he_rel_sd"] == pytest.approx(0.123, abs=1e-3)    # the shipped observation model
    for h, m in present_family.members.items():
        mm = meta["members"][str(h)]
        assert mm["present_input"] is True and mm["T_present_range"] == [0.0, 25.0]
        assert mm["n_hyper"] == 0 and mm["dim"] == m.t.size - 1 == len(mm["df"])
        assert mm["T_max"] == 200.0 and mm["rate_max"] == 100.0 and mm["n_lengths"] == [40, 150]
        assert mm["floor_sd"] == 0.0 and mm["min_draws"] == 1               # no floor taken out unless flagged
    ex = json.loads((present_exported / "example.json").read_text())
    for e in ex["examples"]:
        lo, hi = e["T_present"]
        assert 0.0 <= lo and hi <= 25.0 and hi - lo >= api.MIN_PRESENT_RANGE - 1e-9
        assert lo <= e["truth"]["T"][0] <= hi
        present_family[e["prior"]].present_range(e["T_present"])



def test_float16_weights_halve_the_graphs_and_barely_move_the_draws(present_family, present_exported, tmp_path):
    # the shipped export (scripts/export_onnx.py's default): the large initializers stored as float16, cast on load
    import onnx

    export_family(present_family, tmp_path, float16=True)
    assert json.loads((tmp_path / "meta.json").read_text())["weights"] == "float16"
    for h, m in present_family.members.items():
        g = onnx.load(str(tmp_path / f"sampler{h}.onnx")).graph
        assert any(i.data_type == onnx.TensorProto.FLOAT16 for i in g.initializer)
        assert (tmp_path / f"sampler{h}.onnx").stat().st_size < 0.7 * (present_exported / f"sampler{h}.onnx").stat().st_size
        args = {k: a.numpy() for k, a in zip(INPUTS, example_inputs(m, 3))}
        u = m.net.head.base_sample((4000, m.net.latent.dim), torch.Generator().manual_seed(h), like=torch.zeros(1)).numpy()
        T0 = np.random.default_rng(h).uniform(8.0, 12.0, 4000).astype(np.float32)
        T = {}
        for w, d in (("f32", present_exported), ("f16", tmp_path)):
            (feat,) = _session(d / f"condition{h}.onnx").run(None, args)
            (T[w],) = _session(d / f"sampler{h}.onnx").run(None, {"u": u, "feat": feat, "T0": T0})
            T[w] = _clamp_present(T[w], T0, m.net.latent)
        err = np.abs(T["f16"] - T["f32"])
        # a few hundredths of a kelvin per draw (0.01 median, 0.08 p99 measured here)
        assert np.median(err) < 0.05 and np.quantile(err, 0.99) < 0.5, (np.median(err), np.quantile(err, 0.99))
        # what the page reports: the quantiles per node
        qs = [0.025, 0.25, 0.5, 0.75, 0.975]
        dq = np.abs(np.quantile(T["f16"], qs, axis=0) - np.quantile(T["f32"], qs, axis=0)).max()
        assert dq < 0.5, dq                  # 0.1 K here; at most 0.39 K on the shipped family, 20 000 draws


def test_the_v1_family_takes_no_present_day(family):
    assert not family.long.present_input
    assert all(c["T_present"] is None for c in fixture_samples(family))


def test_present_fixture_refusals_are_refusals(present_family):
    cases = {c["name"]: c for c in fixture_samples(present_family)}
    assert cases["slow_cooling"]["T_present"] == list(FIXTURE_PRESENT)
    for name in ("err_present_missing", "err_present_narrow", "err_present_outside", "err_present_reversed"):
        with pytest.raises(ValueError, match="T_present"):
            present_family.long.present_range(cases[name]["T_present"])
