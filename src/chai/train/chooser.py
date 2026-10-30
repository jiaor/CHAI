"""Training the member chooser.

Every step takes half a batch from each member's corpus, observes both halves
through the same observation model, effort prior and dropout as the heads were
trained on (:class:`~chai.train.data.NoisyBatches`), and labels each row with
the corpus it came from.  Balanced batches are the whole trick: the classifier's
optimum is then the likelihood ratio between the two priors' predictive
distributions, which is the Bayes factor (the model module says why).

Validation is scored on one fixed noisy observation of the first ``val_rows``
of each corpus's val split, drawn once, as for the heads.  Stage 12's rows
(val 100 000-150 000) stay untouched for the chooser's own calibration.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from chai.data.observe import ObservationModel
from chai.models.chooser import MemberChooser
from chai.models.encoder import EncoderConfig
from chai.train.data import (
    AFT_FLAGS,
    AFT_INPUTS,
    GRAIN_INPUTS,
    CorpusSplit,
    EffortPrior,
    NoisyBatches,
    fixed_batches,
    make_loader,
)
from chai.train.loop import _git_sha

__all__ = ["ChooserConfig", "join_batches", "train_chooser", "validate_chooser"]

#: The inputs the encoder reads; everything else a batch carries stays behind.
FEATURES = ("aft", "length_pdf", "grains", "grain_mask")


@dataclass
class ChooserConfig:
    """One chooser run.  ``corpora`` in logit order: the second is the positive class."""

    corpora: tuple[str, str] = ("data/corpus_v22_100.h5", "data/corpus_v22_250.h5")
    members: tuple[Any, Any] = (100, 250)
    observation: str = "configs/observation_v2.yaml"
    effort: EffortPrior = field(default_factory=EffortPrior)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    d_hidden: int = 256
    batch_size: int = 4096
    steps: int = 20_000
    lr: float = 1e-3
    lr_min: float = 1e-5
    warmup: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    val_every: int = 1000
    val_rows: int = 50_000
    val_seed: int = 12345
    num_workers: int = 6
    seed: int = 0
    #: Both corpora restricted to the rows that reheat by at most this many kelvin:
    #: the chooser for ``cooling_only=True``, since the Bayes factor between time spans
    #: under the constraint is between the conditionals (PLAN.md Stage 19c).
    max_reheat: float | None = None
    #: Read the present-day temperature as an input, as the members of a family given it do.
    present_input: bool = False

    def to_config(self) -> dict[str, Any]:
        d = asdict(self)
        d["corpora"], d["members"] = list(self.corpora), list(self.members)
        d["effort"] = self.effort.to_config()
        d["encoder"] = self.encoder.to_config()
        return d

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ChooserConfig:
        cfg = dict(cfg)
        cfg["corpora"] = tuple(cfg.get("corpora", cls.corpora))
        cfg["members"] = tuple(cfg.get("members", cls.members))
        cfg["effort"] = EffortPrior.from_config(cfg.get("effort", {}))
        cfg["encoder"] = EncoderConfig(**cfg.get("encoder", {}))
        return cls(**cfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChooserConfig:
        with open(path) as f:
            return cls.from_config(yaml.safe_load(f) or {})


def join_batches(a: dict[str, Any], b: dict[str, Any], *, present_input: bool = False
                 ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One member-0 batch and one member-1 batch as one labelled batch (features, and ``T0`` if asked)."""
    out = {k: torch.cat([a[k], b[k]]) for k in FEATURES + (("T0",) if present_input else ())}
    label = torch.cat([torch.zeros(len(a["aft"])), torch.ones(len(b["aft"]))])
    return out, label


def _lr_at(step: int, cfg: ChooserConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * min(1.0, frac)))


@torch.no_grad()
def validate_chooser(net: MemberChooser, batches: list[tuple[dict[str, Any], torch.Tensor]],
                     device: Any) -> dict[str, Any]:
    """Log loss (nats/row), accuracy, and the logits with their labels."""
    was = net.training
    net.eval()
    logits, labels = [], []
    for feats, y in batches:
        logits.append(net({k: v.to(device) for k, v in feats.items()}).float().cpu())
        labels.append(y)
    net.train(was)
    s, y = torch.cat(logits), torch.cat(labels)
    ll = torch.nn.functional.binary_cross_entropy_with_logits(s, y, reduction="mean")
    return {"val_logloss": float(ll), "val_acc": float(((s > 0).float() == y).float().mean()),
            "logit": s.numpy(), "label": y.numpy()}


def _val_batches(splits: list[CorpusSplit], observation: ObservationModel, cfg: ChooserConfig
                 ) -> list[tuple[dict[str, Any], torch.Tensor]]:
    half = cfg.batch_size // 2
    per = [fixed_batches(NoisyBatches(s, observation, cfg.effort, seed=cfg.val_seed + i), half,
                         seed=cfg.val_seed + i) for i, s in enumerate(splits)]
    return [join_batches(a, b, present_input=cfg.present_input) for a, b in zip(*per)]


def train_chooser(cfg: ChooserConfig, *, device: Any = "cuda", out_dir: str | Path = "results/chooser",
                  log: Any = print) -> tuple[MemberChooser, list[dict[str, Any]]]:
    """Train from scratch on balanced batches, keep the best validation log loss."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    t0 = time.perf_counter()
    trains = [CorpusSplit(c, "train", max_reheat=cfg.max_reheat) for c in cfg.corpora]
    vals = [CorpusSplit(c, "val", max_rows=cfg.val_rows, max_reheat=cfg.max_reheat)
            for c in cfg.corpora]
    for name in ("length_bins",):
        if not np.array_equal(getattr(trains[0], name), getattr(trains[1], name)):
            raise ValueError(f"the two corpora differ in {name}")
    if trains[0].g_max != trains[1].g_max:
        raise ValueError(f"the corpora carry {trains[0].g_max} and {trains[1].g_max} grains; the grain "
                         "axis must match or the classifier could read the member off the padding")
    with open(cfg.observation) as f:
        observation = ObservationModel.from_config(yaml.safe_load(f))
    log(f"loaded {len(trains[0]):,} + {len(trains[1]):,} train rows in {time.perf_counter() - t0:.1f} s: "
        + ", ".join(f"{m} = {s.path.name} ({s.t_max:g} Ma, {s.n_grid} nodes)" for m, s in zip(cfg.members, trains)))

    half = cfg.batch_size // 2
    loaders = [make_loader(NoisyBatches(s, observation, cfg.effort, seed=cfg.seed + 7919 * i), half,
                           shuffle=True, num_workers=cfg.num_workers, seed=cfg.seed + i)
               for i, s in enumerate(trains)]
    val_batches = _val_batches(vals, observation, cfg)

    net = MemberChooser(len(AFT_INPUTS), trains[0].length_bins.size, len(GRAIN_INPUTS), aft_flags=AFT_FLAGS,
                        members=cfg.members, encoder=cfg.encoder, present_input=cfg.present_input,
                        d_hidden=cfg.d_hidden,
                        meta={"config": cfg.to_config(), "git_sha": _git_sha(),
                              "corpora": [s.meta for s in trains]}).to(device)
    log(f"chooser: {sum(p.numel() for p in net.parameters()):,} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg.to_config(), "git_sha": _git_sha()}, f, sort_keys=False)
    hist_path = out / "history.jsonl"
    hist_path.write_text("")

    history: list[dict[str, Any]] = []
    best, ema, step = math.inf, None, 0
    t_start = time.perf_counter()
    iters = [iter(ld) for ld in loaders]

    def next_of(i: int) -> dict[str, Any]:
        try:
            return next(iters[i])
        except StopIteration:                    # epochs of the two corpora need not line up
            iters[i] = iter(loaders[i])
            return next(iters[i])

    net.train()
    while step < cfg.steps:
        feats, y = join_batches(next_of(0), next_of(1), present_input=cfg.present_input)
        feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
        y = y.to(device)
        if step == 0:
            net.fit_normalisation(feats)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        loss = net.loss(feats, y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        li = float(loss.detach())
        ema = li if ema is None else 0.98 * ema + 0.02 * li
        if step % cfg.val_every == 0 or step == cfg.steps:
            v = validate_chooser(net, val_batches, device)
            rec = {"step": step, "train_logloss_ema": ema, "grad_norm": float(gn), "lr": _lr_at(step - 1, cfg),
                   "elapsed_s": time.perf_counter() - t_start,
                   **{k: v[k] for k in ("val_logloss", "val_acc")}}
            history.append(rec)
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            improved = v["val_logloss"] < best
            if improved:
                best = v["val_logloss"]
                net.meta.update(best_step=step, best_val_logloss=best)
                net.save(out / "model.pt")
            log(f"step {step:6d}  train {ema:.4f}  val {v['val_logloss']:.4f} (ln 2 = 0.6931)  "
                f"acc {v['val_acc']:.3f}  |g| {float(gn):5.2f}  "
                f"{rec['elapsed_s'] / step * 1000:4.0f} ms/step{'  *' if improved else ''}")
    log(f"done: best val log loss {best:.4f} at step {net.meta['best_step']}, "
        f"{(time.perf_counter() - t_start) / 60:.1f} min")
    return MemberChooser.load(out / "model.pt", map_location=device), history
