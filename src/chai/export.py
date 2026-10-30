"""The family as ONNX graphs, for CHAI in the browser.

The browser app (``web/``) runs the three networks with ONNX Runtime Web and
does everything else in JavaScript.  This module cuts each member at the one
place where the work splits by how often it is done:

* a **condition** graph, run once per sample: the encoder's four inputs (any
  number of helium grains) to the embedding and the flow's affine ``(loc, L)``;
* a **sampler** graph, run per batch of draws: base draws ``u`` to clamped
  paths and hyperparameters, through the coupling layers, the affine layer and
  the latent's inverse;

and exports the chooser whole (the four inputs to the log Bayes factor).  The
base draws themselves are made in JavaScript (Student-t, one learned df per
coordinate, carried in ``meta.json``), so the sampler is deterministic and can
be checked against PyTorch draw for draw.

The latent's inverse is written out again here in float32: the package's own
runs in float64, which ONNX Runtime Web does not cover op for op, and the
paths agree to a few thousandths of a kelvin (``tests/test_onnx_export.py``).
It assumes what both shipped latents are -- probit hyperparameter map, no box
logit -- and refuses anything else.

**A member given the present day** (``Model.present_input``, as the shipped members are)
reads the user's surface temperature ``T0`` beside the data, and
:meth:`chai.api.Model.infer` draws one ``T0 ~ U(lo, hi)`` per path, so the
embedding, the affine layer and the couplings' context differ from draw to
draw.  The cut moves to where ``T0`` first enters, the encoder's trunk:

* :class:`PresentCondition`, once per sample: the four inputs to ``feat``,
  everything the encoder computes before its trunk (the AFT scalars, the
  pooled histogram, the grain pool; ``(1, F)``), which ``T0`` does not touch;
* :class:`PresentSampler`, per batch: ``(u, feat, T0)`` to paths.  Each row
  appends its standardised ``T0`` to ``feat`` and runs the trunk, the affine
  layer (its own ``loc`` and Cholesky factor), the couplings with its own
  embedding as context, and the latent's inverse with its ``T0``,
  **unclamped**: a member that takes its training floor back out
  (``floor_sd`` in ``meta.json``, :func:`chai.models.gaussian.remove_floor`)
  does so across a batch of draws before any clamp, which a graph of one row
  per draw cannot, so the page corrects the batch if asked and then clamps to
  the walls and to ``T >= T0`` (``web/js/floor.js``), for every such member.

So the grain encoder and the histogram MLP run once, not per draw; what runs
per draw is the part the Python loop also runs per draw.  The Cholesky factor
is built per row as ``(N, dim, dim)`` -- 26 MB at 256 rows of the 250 Ma
member's 159 coordinates, which bounds the page's batch -- by a gather from
the packed lower triangle rather than a scatter.  :class:`PresentChooser`
takes ``T0`` as ``(K,)`` and returns ``K`` logits; the page reads the range at
:data:`chai.api.N_PRESENT_BF` evenly spaced values and averages them, as
:meth:`chai.api.Family.log_bayes_factor` does.

``feat`` comes from the encoder's own ``forward`` with the trunk swapped for
the identity (:func:`_pre_trunk`), so the export cannot drift from it; only
the trunk's two-line tail is repeated here, and the tests hold both to
PyTorch draw for draw.

:func:`meta` is everything else the port needs, read off the loaded family
rather than retyped: the grids, the walls, the observation model's constants,
the grain prior and effort ranges, the length bins, the projections, the
family's thresholds.  :func:`fixture_samples` is the set of samples the port
is checked against (``scripts/export_onnx.py --fixtures``).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from chai import api
from chai.data.projection import PROJECTIONS
from chai.inputs import sample_from_dict, sample_to_dict

__all__ = ["Chooser", "Condition", "PresentChooser", "PresentCondition", "PresentSampler", "Sampler", "example",
           "export_family", "fixture_samples", "meta", "sample_from_dict", "sample_to_dict"]

_EDGE = 1e-7          # chai.models.latent._EDGE
INPUTS = ("aft", "length_pdf", "grains", "grain_mask")


class Condition(nn.Module):
    """``(aft, length_pdf, grains, grain_mask) -> (embed, loc, scale_tril)``: once per sample."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.encoder, self.head = net.encoder, net.head

    def forward(self, aft: Tensor, length_pdf: Tensor, grains: Tensor, grain_mask: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        e = self.encoder({"aft": aft, "length_pdf": length_pdf, "grains": grains, "grain_mask": grain_mask})
        q = self.head(e)
        return q.embed, q.loc, q.scale_tril


class Sampler(nn.Module):
    """``(u, embed, loc, scale_tril) -> (T, hyper)``: base draws to clamped paths, °C, and ``(tau, rate_sd, drift)``."""

    def __init__(self, net: nn.Module):
        super().__init__()
        L = net.latent
        if L.config.hyper_map != "probit" or L.config.box_logit:
            raise ValueError(f"the export's latent inverse is probit without a box logit; this latent is {L.config}")
        self.head = net.head
        self.n_hyper = L.n_hyper
        f32 = {"dtype": torch.float32}
        g_lo = torch.where(L.h_log, torch.log(L.h_lo), L.h_lo)
        g_hi = torch.where(L.h_log, torch.log(L.h_hi), L.h_hi)
        self.register_buffer("mean", L.mean.to(**f32))
        self.register_buffer("W_inv_T", L.W_inv.T.contiguous().to(**f32))
        self.register_buffer("g_lo", g_lo.to(**f32))
        self.register_buffer("g_span", (g_hi - g_lo).to(**f32))
        self.register_buffer("h_log", L.h_log.clone())
        self.register_buffer("box_lo", L.box_lo.to(**f32))
        self.register_buffer("box_hi", L.box_hi.to(**f32))

    def forward(self, u: Tensor, embed: Tensor, loc: Tensor, scale_tril: Tensor) -> tuple[Tensor, Tensor]:
        # ConditionalFlow.push for one sample's conditioning, with a plain matmul for its einsum:
        # the einsum's broadcast leaves ONNX shape inference sure there is one draw.
        ctx = embed.expand(u.shape[0], embed.shape[-1])
        v = u
        for layer in self.head.layers:
            v, _ = layer.transform(v, ctx, inverse=False)
        z = loc + v @ scale_tril[0].T
        x = self.mean + z @ self.W_inv_T
        y_h, T = x[..., : self.n_hyper], x[..., self.n_hyper:]
        p = (0.5 * (1.0 + torch.erf(y_h / math.sqrt(2.0)))).clamp(_EDGE, 1.0 - _EDGE)
        g = self.g_lo + p * self.g_span
        h = torch.where(self.h_log, torch.exp(g), g)
        T = torch.maximum(torch.minimum(T, self.box_hi), self.box_lo)
        return T, h


class Chooser(nn.Module):
    """``(aft, length_pdf, grains, grain_mask) -> (1,)`` the log Bayes factor of the long member."""

    def __init__(self, chooser: nn.Module):
        super().__init__()
        self.chooser = chooser

    def forward(self, aft: Tensor, length_pdf: Tensor, grains: Tensor, grain_mask: Tensor) -> Tensor:
        return self.chooser({"aft": aft, "length_pdf": length_pdf, "grains": grains, "grain_mask": grain_mask})


# ---------------------------------------------------------------------------
# Members and chooser given the present day
# ---------------------------------------------------------------------------

def _pre_trunk(encoder: nn.Module) -> nn.Module:
    """The encoder up to its trunk: ``forward`` returns ``cat([aft, hist, pooled])``, ``(n, F)``.

    A shallow copy sharing the weights, with the trunk the identity and ``T0`` not read.
    """
    pre = copy.copy(encoder)
    pre._modules = dict(encoder._modules)
    pre._modules["trunk"] = nn.Identity()
    pre.present_input = False
    return pre


def _trunk(encoder: nn.Module, feat: Tensor, T0: Tensor) -> Tensor:
    """The rest of ``AliquotEncoder.forward``: ``(1, F)`` features and ``(N,)`` T0 to ``(N, d_embed)``."""
    x = torch.cat([feat.expand(T0.shape[0], feat.shape[-1]), encoder.norm_T0(T0.reshape(-1, 1))], dim=-1)
    return encoder.trunk(x)


class PresentCondition(nn.Module):
    """``(aft, length_pdf, grains, grain_mask) -> feat`` ``(1, F)``: once per sample, for a member given ``T0``."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.pre = _pre_trunk(net.encoder)

    def forward(self, aft: Tensor, length_pdf: Tensor, grains: Tensor, grain_mask: Tensor) -> Tensor:
        return self.pre({"aft": aft, "length_pdf": length_pdf, "grains": grains, "grain_mask": grain_mask})


class PresentSampler(nn.Module):
    """``(u, feat, T0) -> (T, [hyper])``: one path per row, each with its own present day, °C, unclamped.

    ``hyper`` is an output only for a latent with hyperparameters (a pulse prior's has none).
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        L = net.latent
        if not L.present_input or L.config.hyper_map != "probit" or L.config.box_logit:
            raise ValueError(f"PresentSampler is for a probit latent given T0, without a box logit; this one is {L.config}")
        self.encoder, self.head = net.encoder, net.head
        aff = net.head.affine
        self.dim, self.n_hyper, self.min_scale = aff.dim, L.n_hyper, float(aff.config.min_scale)
        # Each entry of the (dim, dim) factor, row-major, as an index into [packed lower triangle, 0].
        full = torch.full((aff.dim, aff.dim), aff.n_tril, dtype=torch.long)
        full[aff.tril_rows, aff.tril_cols] = torch.arange(aff.n_tril)
        self.register_buffer("tril_gather", full.reshape(-1))
        self.register_buffer("is_diag", aff.is_diag.clone())
        f32 = {"dtype": torch.float32}
        g_lo = torch.where(L.h_log, torch.log(L.h_lo), L.h_lo)
        g_hi = torch.where(L.h_log, torch.log(L.h_hi), L.h_hi)
        self.register_buffer("mean", L.mean.to(**f32))
        self.register_buffer("W_inv_T", L.W_inv.T.contiguous().to(**f32))
        self.register_buffer("g_lo", g_lo.to(**f32))
        self.register_buffer("g_span", (g_hi - g_lo).to(**f32))
        self.register_buffer("h_log", L.h_log.clone())

    def forward(self, u: Tensor, feat: Tensor, T0: Tensor) -> tuple[Tensor, ...]:
        n = u.shape[0]
        embed = _trunk(self.encoder, feat, T0)
        # FlowHead.forward, per row: loc and the Cholesky factor (GaussianHead.scale_tril, gathered, not scattered).
        out = self.head.affine.net(embed)
        loc, raw = out[:, : self.dim], out[:, self.dim:]
        vals = torch.where(self.is_diag, F.softplus(raw) + self.min_scale, raw)
        vals = torch.cat([vals, torch.zeros_like(vals[:, :1])], dim=-1)
        scale_tril = vals.index_select(1, self.tril_gather).reshape(n, self.dim, self.dim)
        # ConditionalFlow.push, each row its own context.
        v = u
        for layer in self.head.layers:
            v, _ = layer.transform(v, embed, inverse=False)
        z = loc + (scale_tril @ v.unsqueeze(-1)).squeeze(-1)
        # Latent.inverse with T0: the nodes after the first are excesses over it.  Not clamped: the page takes
        # the floor out of a batch first if the member asks, then clamps to the walls and to T0 (web/js/floor.js).
        x = self.mean + z @ self.W_inv_T
        y_h, y_T = x[:, : self.n_hyper], x[:, self.n_hyper:]
        t0 = T0.reshape(-1, 1)
        T = torch.cat([t0, t0 + y_T], dim=-1)
        if not self.n_hyper:
            return (T,)
        p = (0.5 * (1.0 + torch.erf(y_h / math.sqrt(2.0)))).clamp(_EDGE, 1.0 - _EDGE)
        g = self.g_lo + p * self.g_span
        return T, torch.where(self.h_log, torch.exp(g), g)


class PresentChooser(nn.Module):
    """``(aft, length_pdf, grains, grain_mask, T0) -> (K,)``: the chooser's logit at each of ``K`` present days."""

    def __init__(self, chooser: nn.Module):
        super().__init__()
        if not chooser.present_input:
            raise ValueError("PresentChooser is for a chooser that reads the present day; export this one with Chooser")
        self.pre = _pre_trunk(chooser.encoder)
        self.encoder, self.head = chooser.encoder, chooser.head

    def forward(self, aft: Tensor, length_pdf: Tensor, grains: Tensor, grain_mask: Tensor, T0: Tensor) -> Tensor:
        feat = self.pre({"aft": aft, "length_pdf": length_pdf, "grains": grains, "grain_mask": grain_mask})
        return self.head(_trunk(self.encoder, feat, T0)).squeeze(-1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def example_inputs(model: api.Model, n_grains: int = 2) -> tuple[Tensor, ...]:
    """Encoder inputs for a plausible sample with ``n_grains`` grains: the shapes the export traces."""
    rng = np.random.default_rng(0)
    s = api.Sample(aft=api.AFT(age=38.0, n_spontaneous=240, lengths=rng.normal(13, 1.2, 100).clip(5, 17), dpar=2.1),
                   grains=[api.Grain(age=24.0 + 3 * k, radius=60 + 5 * k, eU=20 + 5 * k) for k in range(n_grains)])
    b = model.batch(model.to_aliquot(s)[0])
    return tuple(b[k] for k in INPUTS)


def _to_float16_weights(path: Path, min_size: int = 1024) -> None:
    """Store the large float32 initializers of an ONNX file as float16, cast back to float32 on load.

    Half the download; the arithmetic stays float32.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    casts, keep = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_size:
            a = numpy_helper.to_array(init).astype(np.float16)
            name16 = init.name + "__f16"
            keep.append(numpy_helper.from_array(a, name16))
            casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__cast"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.save(m, str(path))


def _export(module: nn.Module, args: tuple[Tensor, ...], path: Path, dynamic: tuple[dict[int, Any], ...],
            names: tuple[str, ...], outputs: tuple[str, ...], float16: bool) -> None:
    import onnx

    torch.onnx.export(module.eval(), args, str(path), dynamo=True, dynamic_shapes=dynamic, input_names=list(names),
                      output_names=list(outputs), external_data=False, optimize=True, verbose=False)
    onnx.checker.check_model(str(path))
    if float16:
        _to_float16_weights(path)


def export_family(family: api.Family, out: Path, *, float16: bool = False) -> dict[str, Any]:
    """Write every member's condition and sampler graphs, the chooser, ``meta.json`` and ``example.json`` into ``out``."""
    import json

    out.mkdir(parents=True, exist_ok=True)
    G = torch.export.Dim("G", min=1, max=256)
    N = torch.export.Dim("N", min=1, max=1_000_000)
    enc_dyn = ({}, {}, {1: G}, {1: G})
    info = meta(family)
    with torch.no_grad():
        for h, m in family.members.items():
            args = example_inputs(m)
            if m.present_input:
                cond = PresentCondition(m.net)
                feat = cond(*args)
                sampler = PresentSampler(m.net)
                u = m.net.head.base_sample((64, m.net.latent.dim), torch.Generator().manual_seed(0), like=feat)
                T0 = torch.linspace(*m.T_present_range, 64)
                _export(cond, args, out / f"condition{h}.onnx", enc_dyn, INPUTS, ("feat",), float16)
                _export(sampler, (u, feat, T0), out / f"sampler{h}.onnx", ({0: N}, {}, {0: N}), ("u", "feat", "T0"),
                        ("T", "hyper") if sampler.n_hyper else ("T",), float16)
                continue
            cond = Condition(m.net)
            embed, loc, L = cond(*args)
            u = m.net.head.base_sample((64, loc.shape[-1]), torch.Generator().manual_seed(0), like=loc)
            _export(cond, args, out / f"condition{h}.onnx", enc_dyn, INPUTS, ("embed", "loc", "scale_tril"), float16)
            _export(Sampler(m.net), (u, embed, loc, L), out / f"sampler{h}.onnx", ({0: N}, {}, {}, {}),
                    ("u", "embed", "loc", "scale_tril"), ("T", "hyper"), float16)
        if family.chooser.present_input:
            K = torch.export.Dim("K", min=1, max=4096)
            T0 = torch.linspace(*family.long.T_present_range, api.N_PRESENT_BF)
            _export(PresentChooser(family.chooser), (*example_inputs(family.long), T0), out / "chooser.onnx",
                    (*enc_dyn, {0: K}), (*INPUTS, "T0"), ("log_bayes_factor",), float16)
        else:
            _export(Chooser(family.chooser), example_inputs(family.long), out / "chooser.onnx", enc_dyn, INPUTS,
                    ("log_bayes_factor",), float16)
    (out / "example.json").write_text(json.dumps(example(family)))
    info["weights"] = "float16" if float16 else "float32"
    info["bytes"] = {p.name: p.stat().st_size for p in sorted(out.glob("*.onnx"))}
    (out / "meta.json").write_text(json.dumps(info, indent=1))
    return info


def meta(family: api.Family) -> dict[str, Any]:
    """Everything the port needs besides the networks, read off the loaded family."""
    from importlib.metadata import PackageNotFoundError, version

    from chai.constraints import PASSES, REHEAT_TOL, WITHIN

    try:
        ver = version("chai-thermo")
    except PackageNotFoundError:
        ver = "unknown"

    def member(h: int, m: api.Model) -> dict[str, Any]:
        L = m.net.latent
        lo_len, hi_len = m.effort.get("n_lengths", (20, 150))
        return {
            "horizon": h, "t": m.t.tolist(), "box_lo": L.box_lo.tolist(), "box_hi": L.box_hi.tolist(),
            "dim": L.dim, "n_hyper": L.n_hyper, "hyper_names": list(L.hyper_names),
            "df": m.net.head.df.detach().double().tolist(),
            "rate_max": float(m.prior["rate_max"]), "T_max": float(m.prior.get("T_max", m.prior.get("T_top"))),
            "present_input": m.present_input,
            "T_present_range": list(m.T_present_range) if m.present_input else None,
            # The training floor, K, the page takes back out of each batch of draws (0: none), and the
            # fewest draws it may correct at once (chai.api.Model.min_draws).
            "floor_sd": float(m.net.floor_sd), "min_draws": m.min_draws,
            "g_max": m.g_max, "n_lengths": [lo_len, hi_len], "grain_prior": m.grain_prior,
            "n_layers": m.net.head.config.n_layers,
            "condition": f"condition{h}.onnx", "sampler": f"sampler{h}.onnx",
        }

    obs = family.long.observation
    return {
        "package": "chai-thermo", "version": ver,
        "members": {str(h): member(h, m) for h, m in sorted(family.members.items())},
        "chooser": {"file": "chooser.onnx", "members": [int(x) for x in family.chooser.members],
                    "present_input": bool(family.chooser.present_input)},
        "n_present_bf": api.N_PRESENT_BF, "min_present_range": api.MIN_PRESENT_RANGE,
        "observation": {"aft_systematic_rel_var": float(obs.aft.systematic_rel_var), "he_rel_sd": float(obs.he.rel_sd),
                        "dpar_sd": float(obs.aft.dpar_sd)},
        "eu_th_weight": api._EU_TH_WEIGHT, "n_bins": api.N_BINS, "bin_width": api.BIN_WIDTH,
        "projections": {k: dataclasses.asdict(p) for k, p in PROJECTIONS.items()},
        "default_projection": "ketcham2007_b03",
        "family": {"undecided": list(api.UNDECIDED), "old_sample_ma": api.OLD_SAMPLE_MA,
                   "long_coverage_68": list(api.LONG_COVERAGE_68)},
        "quantiles": list(api.QUANTILES), "box_modes": [PASSES, WITHIN], "reheat_tol": REHEAT_TOL,
        "few_lengths": api.FEW_LENGTHS,
        "alpha": _alpha_constants(),
        "fit": _fit_constants(obs),
    }


def _fit_constants(obs: Any) -> dict[str, Any]:
    """What :func:`chai.fit.check` uses besides the observation model's helium and AFT noise."""
    from chai import fit

    x, w = np.polynomial.hermite_e.hermegauss(fit.GH_NODES)
    return {"c": fit.FIT_C, "beta": fit.FIT_BETA, "min_lengths": fit.FIT_MIN_LENGTHS, "gh_x": x.tolist(), "gh_w": w.tolist(),
            "lambda_d_ma": fit.LAMBDA_D_MA, "length_meas_sd": float(obs.aft.length_meas_sd_um),
            "bisect": fit._BISECT, "caution": fit.CAUTION}


def _alpha_constants() -> dict[str, Any]:
    """What :func:`chai.api.alpha_correct` uses, per parent (238U, 235U, 232Th)."""
    from chai.forward.torchimpl import rdaam as r

    return {"stopping_um": list(r.AS_DIST[:3]), "decay_per_yr": [r.U238YR, r.U235YR, r.TH232YR],
            "u238_u235": r.U238U235RATIO, "ppm_nmpg": [r.PPM_NMPG_U238, r.PPM_NMPG_U235, r.PPM_NMPG_TH232],
            "yield": list(r.YIELD[:3]), "th_u": api.RAW_AGE_TH_U}


def example(family: api.Family, n: int = 20, seed: int = 7) -> dict[str, Any]:
    """The page's examples: ``n`` random samples per member, each with the true path to draw beside the answer.

    A path from that member's history prior, grains and Dpar from the grain prior (2-5 grains), 60-120
    lengths, and the forward models and noise the heads were trained on (:func:`chai.api.simulate_sample`).
    Samples too hot to have kept 20 lengths are drawn again.  The page picks one at random.
    For a family given the present day each carries ``T_present``: the true path's ``T0`` +- 2.5 K, inside
    the trained range and at least :data:`chai.api.MIN_PRESENT_RANGE` wide.
    """
    from chai.data.priors import GrainPrior, prior_from_config

    rng = np.random.default_rng(seed)
    out = []
    for h, m in sorted(family.members.items()):
        prior = prior_from_config(m.prior, _prior_kind(m.prior))
        gp = GrainPrior.from_config(m.grain_prior)
        kept = 0
        while kept < n:
            T = prior.sample_on(1, rng, m.t)[0][0]
            k = int(rng.integers(2, 6))
            g = gp.sample(k, rng)
            s, (tg, Tg) = api.simulate_sample(m, m.t, T, dpar=float(g["dpar"][0]), n_lengths=int(rng.integers(60, 121)),
                                              grains=list(zip(g["radius"], g["U"], g["Th"])),
                                              seed=int(rng.integers(1 << 31)))
            if s.aft is None or len(s.aft.lengths) < 20 or s.aft.age < 1.0 or min(x.age for x in s.grains) < 1.0:
                continue
            d = sample_to_dict(s)
            d["name"] = "random example"
            d["aft"]["lengths"] = _r(d["aft"]["lengths"])
            for gr in d["grains"]:
                gr.update({key: round(float(gr[key]), 3) for key in ("age", "radius", "eU", "age_sd")})
            d["aft"].update({key: round(float(d["aft"][key]), 3) for key in ("age", "age_sd", "dpar")})
            ex = {"sample": d, "prior": h, "truth": {"t": [float(v) for v in tg], "T": [round(float(v), 3) for v in Tg]}}
            if m.present_input:
                ex["T_present"] = example_present_range(m, float(Tg[0]))
            out.append(ex)
            kept += 1
    return {"examples": out, "description": "A randomly generated sample."}


def example_present_range(m: api.Model, T0: float, half: float = 2.5) -> list[float]:
    """``T0 +- half`` K, clipped to the trained range and widened to :data:`chai.api.MIN_PRESENT_RANGE` inside it."""
    p_lo, p_hi = m.T_present_range
    lo, hi = max(T0 - half, p_lo), min(T0 + half, p_hi)
    if hi - lo < api.MIN_PRESENT_RANGE:
        lo, hi = (p_lo, p_lo + api.MIN_PRESENT_RANGE) if lo <= p_lo else (p_hi - api.MIN_PRESENT_RANGE, p_hi)
    return [round(lo, 2), round(hi, 2)]


def _prior_kind(cfg: dict[str, Any]) -> str:
    """The history prior class a stored config is for: the one whose fields hold all its keys (models do not record it)."""
    from chai.data.priors import PRIOR_TYPES

    keys = {"T_max" if k == "T_top" else k for k in cfg}  # T_top: the bound's name in checkpoints before 2026-10-23
    fits = [k for k, cls in PRIOR_TYPES.items() if keys <= {f.name for f in dataclasses.fields(cls)}]
    if not fits:
        raise ValueError(f"no history prior takes the keys {sorted(cfg)}")
    return fits[0]


# ---------------------------------------------------------------------------
# Samples as JSON, and the fixture set the port is checked against
# ---------------------------------------------------------------------------

def _r(x: np.ndarray, nd: int = 2) -> list[float]:
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def fixture_samples(family: api.Family) -> list[dict[str, Any]]:
    """About thirty cases: simulated samples, every warning the API gives, boxes, prior odds, and refusals.

    Each is ``{"name", "sample": <dict or None>, "boxes": [...], "cooling_only", "prior_odds", "n", "member",
    "T_present"}``; a case whose sample the API refuses carries ``"raw"`` (the dict) and no ``"sample"``.
    ``T_present`` is ``None`` for a family that draws the present day, and :data:`FIXTURE_PRESENT` for one given
    it, except in the cases that try other ranges and the ones the API refuses.
    """
    rng = np.random.default_rng(17)
    cases: list[dict[str, Any]] = []
    present = family.long.present_input
    T_present = list(FIXTURE_PRESENT) if present else None

    def add(name: str, sample: dict[str, Any] | api.Sample, boxes: Sequence[dict[str, Any]] = (),
            prior_odds: float = 1.0, n: int = 2000, member: str | int = "auto", cooling_only: bool = False,
            T_present: Sequence[float] | None = T_present) -> None:
        d = sample_to_dict(sample) if isinstance(sample, api.Sample) else sample
        d["name"] = name
        cases.append({"name": name, "sample": d, "boxes": list(boxes), "cooling_only": cooling_only,
                      "prior_odds": prior_odds, "n": n, "member": member,
                      "T_present": None if T_present is None else list(T_present)})

    histories = {
        "slow_cooling": ([0, 60, 100], [10, 110, 160]),
        "rapid_20ma": ([0, 18, 22, 100], [12, 30, 115, 130]),
        "reheated": ([0, 10, 30, 60, 100], [15, 80, 40, 20, 140]),
        "old_cratonic": ([0, 120, 200, 250], [18, 45, 95, 150]),
        "exhumed_150ma": ([0, 140, 160, 250], [20, 35, 120, 150]),
    }
    sims: dict[str, api.Sample] = {}
    for k, (name, (t, T)) in enumerate(histories.items()):
        s, _ = api.simulate_sample(family, t, T, dpar=1.6 + 0.3 * k, seed=k,
                                   grains=[(45 + 12 * j, 8 + 9 * j, 20 + 5 * j) for j in range(4)])
        sims[name] = s
        add(name, s)
    base = sims["slow_cooling"]
    a = base.aft
    L = np.asarray(a.lengths)

    # Built to misfit: helium ages falling as eU rises, and an AFT age far older than every helium age.
    ages = sorted(g.age for g in base.grains)
    mid = float(np.mean(ages))
    inv = [mid + 1.8 * (x - mid) for x in reversed(ages)]
    add("misfit_inverted_he", api.Sample(aft=a, grains=[dataclasses.replace(g, age=float(x), age_sd=None)
                                                          for g, x in zip(base.grains, inv)]))
    add("misfit_old_aft", api.Sample(aft=dataclasses.replace(a, age=3.0 * a.age, n_spontaneous=3 * a.n_spontaneous,
                                                             age_sd=None), grains=base.grains))
    add("aft_only", api.Sample(aft=a))
    add("he_only", api.Sample(grains=base.grains))
    add("no_lengths", api.Sample(aft=api.AFT(age=a.age, n_spontaneous=a.n_spontaneous, dpar=a.dpar),
                                 grains=base.grains))
    add("age_sd_only", {"aft": {"age": 45.0, "age_sd": 3.1, "lengths": _r(L), "dpar": 2.2}, "grains": []})
    add("age_sd_disagrees", {"aft": {"age": 45.0, "n_spontaneous": 300, "age_sd": 9.0, "lengths": _r(L), "dpar": 2.2},
                             "grains": [{"age": 30.0, "radius": 70.0, "eU": 25.0, "age_sd": 0.3}]})
    # Measured lengths and angles: longer than their projections, as a lab reports them.
    ang = _r(rng.uniform(0, 90, L.size), 1)
    meas = _r(L * (1 - 0.06 * np.sin(np.deg2rad(ang)) ** 2))
    for proj in PROJECTIONS:
        add(f"angles_{proj}", {"aft": {"age": a.age, "n_spontaneous": a.n_spontaneous, "lengths": meas, "angles": ang,
                                       "dpar": a.dpar, "projection": proj}, "grains": sample_to_dict(base)["grains"]})
    add("few_lengths", {"aft": {"age": 60.0, "n_spontaneous": 150, "lengths": _r(L[:8]), "dpar": 1.0}, "grains": []})
    add("many_lengths_dpar_high", {"aft": {"age": 60.0, "n_spontaneous": 150,
                                           "lengths": _r(rng.normal(13.5, 1.1, 180).clip(6, 16.5)), "dpar": 4.2},
                                   "grains": []})
    add("zero_tracks", {"aft": {"age": 12.0, "n_spontaneous": 0, "lengths": [], "dpar": 2.0},
                        "grains": [{"age": 9.0, "radius": 60.0, "eU": 30.0}]})
    add("odd_grains", {"aft": None, "grains": [
        {"age": 40.0, "radius": 20.0, "eU": 30.0, "age_sd": 2.6},
        {"age": 42.0, "radius": 140.0, "eU": 0.5, "age_sd": 0.4},
        {"age": 41.0, "radius": 60.0, "eU": 3000.0},
        {"age": 38.0, "radius": 60.0, "eU": 40.0, "age_sd": 10.0}]})
    add("fourteen_grains", {"aft": None, "grains": [
        {"age": float(30 + k), "radius": float(40 + 4 * k), "eU": float(10 + 3 * k)} for k in range(14)]})
    add("uncorrected_he", {"aft": None, "grains": [
        {"age": 24.0, "radius": 45.0, "eU": 30.0, "corrected": False},
        {"age": 31.0, "radius": 80.0, "eU": 12.0, "corrected": False},
        {"age": 36.0, "radius": 60.0, "eU": 55.0}]})
    add("uncorrected_one", {"aft": {"age": 45.0, "n_spontaneous": 200, "lengths": _r(L), "dpar": 2.0},
                            "grains": [{"age": 33.0, "radius": 70.0, "eU": 25.0, "corrected": False}]})
    add("five_small_grains", {"aft": None, "grains": [
        {"age": float(30 + k), "radius": 15.0 + k, "eU": 20.0} for k in range(5)]})
    add("past_100", {"aft": {"age": 130.0, "n_spontaneous": 400, "lengths": _r(L - 1.0), "dpar": 2.0},
                     "grains": [{"age": 118.0, "radius": 70.0, "eU": 20.0}, {"age": 104.0, "radius": 50.0, "eU": 12.0}]})
    add("past_250", {"aft": {"age": 280.0, "n_spontaneous": 900, "lengths": _r(L - 1.5), "dpar": 2.0}, "grains": [
        {"age": 262.0, "radius": 80.0, "eU": 15.0}, {"age": 210.0, "radius": 45.0, "eU": 60.0}]})
    add("old_odds_0p3", sims["old_cratonic"], prior_odds=0.3)
    add("odds_10", base, prior_odds=10.0)
    add("odds_0p1", sims["old_cratonic"], prior_odds=0.1)
    add("member_100_old", sims["old_cratonic"], member=100)
    add("member_250_young", sims["rapid_20ma"], member=250)

    passes = {"t": [20, 30], "T": [60, 90], "mode": "passes"}
    within = {"t": [0, 5], "T": [0, 25], "mode": "within"}
    soft = {"t": [40, 40], "T": [90, 110], "mode": "passes", "soft_sd": 10.0}
    add("box_passes", sims["rapid_20ma"], boxes=[passes])
    add("box_within_soft", base, boxes=[within, soft])
    add("box_past_100", sims["old_cratonic"], boxes=[{"t": [150, 180], "T": [70, 120], "mode": "passes"}])
    add("box_outside", base, boxes=[{"t": [90, 300], "T": [0, 50], "mode": "passes"}])
    add("box_walls", base, boxes=[{"t": [0, 0], "T": [40, 60], "mode": "within"}])
    add("box_rate", base, boxes=[{"t": [1, 1], "T": [150, 160], "mode": "passes", "label": "too fast"}])
    # Cooling only (QTQt's monotonic option): alone, with a box, on fission track alone, and one member asked for.
    add("cooling_only", base, cooling_only=True)
    add("cooling_box", sims["rapid_20ma"], boxes=[passes], cooling_only=True)
    add("cooling_aft_only", api.Sample(aft=a), cooling_only=True)
    add("cooling_member_250", sims["old_cratonic"], member=250, cooling_only=True)
    if present:
        # Other present days: the whole trained range, a warm one, the narrowest; and ranges the API refuses.
        p_lo, p_hi = family.long.T_present_range
        add("present_full_range", base, T_present=(p_lo, p_hi))
        add("present_warm", sims["rapid_20ma"], T_present=(p_hi - 5.0, p_hi))
        add("present_narrowest", base, T_present=(10.0, 10.0 + api.MIN_PRESENT_RANGE), boxes=[within])
        for name, rng_ in {"err_present_missing": None, "err_present_narrow": (10.0, 10.5),
                           "err_present_outside": (p_hi - 5.0, p_hi + 5.0),
                           "err_present_reversed": (12.0, 8.0)}.items():
            add(name, base, T_present=rng_)

    # Refusals: the API raises, and the port must say the same.
    for name, raw in {
        "err_no_data": {"aft": None, "grains": []},
        "err_age_sd_floor": {"aft": {"age": 45.0, "age_sd": 1.0, "lengths": [], "dpar": 2.0}, "grains": []},
        "err_long_track": {"aft": {"age": 45.0, "n_spontaneous": 100, "lengths": [12.0, 20.5], "dpar": 2.0},
                           "grains": []},
        "err_no_precision": {"aft": {"age": 45.0, "lengths": [], "dpar": 2.0}, "grains": []},
        "err_angles_count": {"aft": {"age": 45.0, "n_spontaneous": 100, "lengths": [12.0, 13.0], "angles": [10.0],
                                     "dpar": 2.0}, "grains": []},
        "err_angle_range": {"aft": {"age": 45.0, "n_spontaneous": 100, "lengths": [12.0, 13.0], "angles": [10.0, 95.0],
                                    "dpar": 2.0}, "grains": []},
        "err_grain_radius": {"aft": None, "grains": [{"age": 30.0, "radius": 0.0, "eU": 10.0}]},
        "err_raw_radius": {"aft": None, "grains": [{"age": 30.0, "radius": 9.0, "eU": 10.0, "corrected": False}]},
        "err_projection": {"aft": {"age": 45.0, "n_spontaneous": 100, "lengths": [12.0], "angles": [10.0],
                                   "dpar": 2.0, "projection": "nope"}, "grains": []},
    }.items():
        raw["name"] = name
        cases.append({"name": name, "raw": raw, "boxes": [], "cooling_only": False, "prior_odds": 1.0, "n": 2000,
                      "member": "auto", "T_present": T_present})
    return cases


#: The present-day range the fixtures of a family given it run at, °C.
FIXTURE_PRESENT = (8.0, 12.0)
