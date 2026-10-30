"""Training the amortized posterior.

Plain PyTorch: one optimiser, one schedule, a loop.  The run's shape is small
enough -- a million parameters, a 67-dimensional Gaussian -- that the data
pipeline is the cost, and that is in :mod:`chai.train.data`'s workers.

Everything a run needs is in :class:`TrainConfig`, and everything a run
produced is in its output directory: ``config.yaml`` (the config as run,
with the git SHA), ``history.jsonl`` (one line per validation), ``model.pt``
(the best validation checkpoint, loadable with
:meth:`AmortizedPosterior.load`) and ``diagnostics.npz`` (the per-node
summary of :func:`chai.eval.calibration.posterior_summary` on the fixed
validation set).

Validation is scored on **one fixed noisy observation** of the first rows of
the val split, drawn once with its own seed.  The training loader redraws
its noise every batch; the validation set does not, so a change in its
number is a change in the model.  The test split is not opened here.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from chai.data.observe import ObservationModel
from chai.eval.calibration import posterior_summary
from chai.models.encoder import EncoderConfig
from chai.models.flow import FlowConfig
from chai.models.gaussian import AmortizedPosterior, HeadConfig, head_config_from, residual_of
from chai.models.latent import Latent
from chai.train.data import (
    AFT_FLAGS,
    AFT_INPUTS,
    GRAIN_INPUTS,
    CorpusSplit,
    EffortPrior,
    NoisyBatches,
    WallDequantiser,
    fixed_batches,
    make_loader,
    mix_splits,
)

__all__ = ["TrainConfig", "train"]


@dataclass
class TrainConfig:
    """One run.  Paths are relative to the working directory."""

    corpus: str = "data/corpus_v22_100.h5"
    latent: str = "data/latent_v22_100.pt"
    observation: str | None = "configs/observation_v2.yaml"
    effort: EffortPrior = field(default_factory=EffortPrior)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig | FlowConfig = field(default_factory=HeadConfig)
    dequantise: WallDequantiser | None = None
    batch_size: int = 4096
    steps: int = 20_000
    lr: float = 1e-3
    lr_min: float = 1e-5
    warmup: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    val_every: int = 500
    val_rows: int = 50_000
    val_seed: int = 12345
    num_workers: int = 8
    seed: int = 0
    max_train_rows: int | None = None
    n_diag_samples: int = 64
    #: Train on empty samples only: the head becomes a density on the prior in
    #: the latent's (dequantised) coordinates.  Stage 13's importance weights.
    unconditional: bool = False
    #: Share of rows shown as empty samples, so the head is also a density on
    #: the prior -- in the same network as the posterior (Stage 13).
    p_empty: float = 0.0
    #: A second corpus whose train rows are appended (``mix_splits``): the
    #: episodic histories of Stage 13.  Validation stays on ``corpus``.
    mix_corpus: str | None = None
    #: A saved checkpoint (``model.pt``) of the same architecture whose weights
    #: start this run, in place of random initialisation -- the encoder's
    #: normalisation moments come with them.  The schedule (``lr``, ``warmup``,
    #: ``steps``) is the new run's own: a warm start continues the fit
    #: under a fresh cosine cycle (Stage 16 follow-up, ``docs/time-span.md``).
    init_from: str | None = None
    #: Train and validate on the rows whose stored path reheats by at most this
    #: many kelvin: a cooling-only head (``docs/cooling-coverage.md``).
    max_reheat: float | None = None
    #: Also save the current weights as ``model_<k>k.pt`` (``model_<step>.pt`` off the thousands) every this many steps,
    #: from ``save_from`` on (0 = off), whatever their validation NLL: snapshots
    #: to compare training lengths within one run.  ``model.pt`` stays the best.
    save_every: int = 0
    save_from: int = 0

    def to_config(self) -> dict[str, Any]:
        d = asdict(self)
        d["effort"] = self.effort.to_config()
        d["encoder"] = self.encoder.to_config()
        d["head"] = self.head.to_config()
        d["dequantise"] = None if self.dequantise is None else self.dequantise.to_config()
        return d

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TrainConfig:
        cfg = dict(cfg)
        cfg["effort"] = EffortPrior.from_config(cfg.get("effort", {}))
        cfg["encoder"] = EncoderConfig(**cfg.get("encoder", {}))
        cfg["head"] = head_config_from(cfg.get("head", {}))
        cfg["dequantise"] = WallDequantiser.from_config(cfg.get("dequantise"))
        return cls(**cfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            return cls.from_config(yaml.safe_load(f) or {})


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return ""


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * min(1.0, frac)))


def _to(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _observation(cfg: TrainConfig) -> ObservationModel:
    if cfg.observation is None:
        return ObservationModel()
    with open(cfg.observation) as f:
        return ObservationModel.from_config(yaml.safe_load(f))


@torch.no_grad()
def _validate(model: AmortizedPosterior, batches: list[dict[str, Any]], device: Any
              ) -> dict[str, float]:
    model.eval()
    tot = tot0 = 0.0
    n = 0
    resid_var = torch.zeros(model.dim, dtype=torch.float64, device=device)
    for raw in batches:
        b = _to(raw, device)
        q = model(b)
        tot += float(-q.log_prob(b["z"]).sum())
        tot0 += float(model.prior_nll(b["z"]).sum())
        r = residual_of(q, b["z"])
        resid_var += r.pow(2).sum(0).double()
        n += len(b["z"])
    model.train()
    rv = (resid_var / n).cpu().numpy()
    out = {"val_nll": tot / n, "val_nll_prior": tot0 / n,
           "resid_var_median": float(np.median(rv)), "resid_var_max": float(rv.max()),
           "resid_var_min": float(rv.min())}
    df = getattr(model.head, "df", None)
    if df is not None:
        out["df_min"], out["df_median"] = float(df.min()), float(df.median())
    return out


def train(cfg: TrainConfig, *, device: Any = "cuda", out_dir: str | Path = "results/run",
          log: Any = print) -> tuple[AmortizedPosterior, list[dict[str, Any]]]:
    """Train from scratch, validate on a fixed set, keep the best, write everything out."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.perf_counter()
    latent = Latent.load(cfg.latent)
    train_split = CorpusSplit(cfg.corpus, "train", latent=latent, max_rows=cfg.max_train_rows,
                              max_reheat=cfg.max_reheat)
    val_split = CorpusSplit(cfg.corpus, "val", latent=latent, max_rows=cfg.val_rows,
                            max_reheat=cfg.max_reheat)
    if cfg.mix_corpus:
        extra = CorpusSplit(cfg.mix_corpus, "train", max_rows=cfg.max_train_rows)
        n_base = len(train_split)
        train_split = mix_splits(train_split, extra, latent, np.random.default_rng([cfg.seed, 13]))
        log(f"mixed in {len(extra):,} train rows of {cfg.mix_corpus}: {len(extra) / len(train_split):.1%} of "
            f"{len(train_split):,} (base {n_base:,})")
    observation = _observation(cfg)
    log(f"loaded {len(train_split):,} train + {len(val_split):,} val rows in "
        f"{time.perf_counter() - t0:.1f} s; G_max = {train_split.g_max}, "
        f"n_grid = {train_split.n_grid}, t_max = {train_split.t_max:g} Ma")
    log(cfg.effort.describe())

    ds = NoisyBatches(train_split, observation, cfg.effort, seed=cfg.seed,
                      dequantise=cfg.dequantise, inputs=not cfg.unconditional, p_empty=cfg.p_empty)
    loader = make_loader(ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                         seed=cfg.seed)
    val_ds = NoisyBatches(val_split, observation, cfg.effort, seed=cfg.val_seed,
                          dequantise=cfg.dequantise, inputs=not cfg.unconditional)
    if cfg.p_empty > 0:                                # validated on real samples only: comparable NLL
        log(f"{cfg.p_empty:.0%} of training rows shown as empty samples: the head also carries the prior")
    if cfg.unconditional:
        log("unconditional: every sample is empty, the head is a density on the prior")
    if cfg.dequantise is not None:
        log(f"walls dequantised: offset U(0, {cfg.dequantise.width:g}) K per row and wall, "
            f"jitter U(0, {cfg.dequantise.jitter:g}) K per node; every node + N(0, "
            f"{cfg.dequantise.noise_sd:g}^2) K")
    val_batches = fixed_batches(val_ds, cfg.batch_size, seed=cfg.val_seed)

    model = AmortizedPosterior(
        latent, n_aft=len(AFT_INPUTS), n_bins=train_split.length_bins.size,
        n_grain=len(GRAIN_INPUTS), aft_flags=AFT_FLAGS, encoder=cfg.encoder, head=cfg.head,
        meta={"config": cfg.to_config(), "git_sha": _git_sha(),
              "corpus": train_split.meta, "aft_inputs": list(AFT_INPUTS),
              "grain_inputs": list(GRAIN_INPUTS)},
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model: {n_params:,} trainable parameters, latent dim {model.dim}")
    if cfg.init_from:
        start = torch.load(cfg.init_from, map_location=device, weights_only=False)
        model.load_state_dict(start["state_dict"])
        model.meta["init_from"] = {"path": str(cfg.init_from),
                                   "best_step": start["meta"].get("best_step"),
                                   "best_val_nll": start["meta"].get("best_val_nll")}
        log(f"warm start from {cfg.init_from} (its best step {start['meta'].get('best_step')}, "
            f"val NLL {start['meta'].get('best_val_nll', float('nan')):.3f})")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg.to_config(), "git_sha": _git_sha()}, f, sort_keys=False)
    hist_path = out / "history.jsonl"
    hist_path.write_text("")

    history: list[dict[str, Any]] = []
    best = math.inf
    ema = None
    step = 0
    t_start = time.perf_counter()
    model.train()
    done = False
    while not done:
        for raw in loader:
            b = _to(raw, device)
            if step == 0 and not cfg.unconditional and not cfg.init_from:
                model.fit_normalisation(b)             # a warm start keeps its moments
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, cfg)
            loss = model.nll(b).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            step += 1
            li = float(loss.detach())
            ema = li if ema is None else 0.98 * ema + 0.02 * li

            if step % cfg.val_every == 0 or step == cfg.steps:
                v = _validate(model, val_batches, device)
                rec = {"step": step, "epoch": step * cfg.batch_size / len(train_split),
                       "train_nll_ema": ema, "grad_norm": float(gn), "lr": _lr_at(step - 1, cfg),
                       "elapsed_s": time.perf_counter() - t_start,
                       "gpu_peak_gb": (torch.cuda.max_memory_allocated(device) / 1e9
                                       if device.type == "cuda" else 0.0), **v}
                history.append(rec)
                with open(hist_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                improved = v["val_nll"] < best
                if improved:
                    best = v["val_nll"]
                    model.meta["best_step"] = step
                    model.meta["best_val_nll"] = best
                    model.save(out / "model.pt")
                if cfg.save_every and step >= cfg.save_from and step % cfg.save_every == 0:
                    best_meta = (model.meta.get("best_step"), model.meta.get("best_val_nll"))
                    model.meta["best_step"], model.meta["best_val_nll"] = step, v["val_nll"]
                    tag = f"{step // 1000}k" if step % 1000 == 0 else str(step)
                    model.save(out / f"model_{tag}.pt")
                    model.meta["best_step"], model.meta["best_val_nll"] = best_meta
                df_txt = f"  df {v['df_min']:.1f}/{v['df_median']:.1f}" if "df_min" in v else ""
                log(f"step {step:6d}  ep {rec['epoch']:5.1f}  train {ema:7.3f}  "
                    f"val {v['val_nll']:7.3f} (prior {v['val_nll_prior']:.2f})  "
                    f"resid var {v['resid_var_median']:.3f} [{v['resid_var_min']:.2f}, "
                    f"{v['resid_var_max']:.2f}]{df_txt}  |g| {float(gn):6.2f}  "
                    f"{rec['elapsed_s'] / step * 1000:5.0f} ms/step{'  *' if improved else ''}")
            if step >= cfg.steps:
                done = True
                break

    # Diagnostics on the best checkpoint, on the same fixed validation set.
    best_model = AmortizedPosterior.load(out / "model.pt", map_location=device)
    if cfg.unconditional:                              # no data, so no per-modality summary
        log(f"done: best val NLL {best:.3f} at step {model.meta['best_step']}, "
            f"{(time.perf_counter() - t_start) / 60:.1f} min")
        return best_model, history
    diag = posterior_summary(best_model, val_batches, device=device,
                             n_samples=cfg.n_diag_samples, prior_sd=val_split.prior_sd())
    np.savez(out / "diagnostics.npz", **diag)
    log(f"done: best val NLL {best:.3f} at step {model.meta['best_step']}, "
        f"{(time.perf_counter() - t_start) / 60:.1f} min")
    return best_model, history
