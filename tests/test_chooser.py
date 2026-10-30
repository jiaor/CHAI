"""The member chooser: a balanced classifier is a Bayes factor. (PLAN.md Stage 16f)"""
from __future__ import annotations

import math

import numpy as np
import torch

from chai.models.chooser import MemberChooser
from chai.train.chooser import ChooserConfig, join_batches, train_chooser, validate_chooser
from chai.train.data import AFT_FLAGS, AFT_INPUTS, GRAIN_INPUTS


def _batch(n: int, g: int = 5, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    aft = torch.randn(n, len(AFT_INPUTS), generator=gen)
    aft[:, list(AFT_FLAGS)] = 1.0
    return {"aft": aft, "length_pdf": torch.softmax(torch.randn(n, 200, generator=gen), -1),
            "grains": torch.randn(n, g, len(GRAIN_INPUTS), generator=gen),
            "grain_mask": torch.ones(n, g, dtype=torch.bool)}


def test_an_untrained_chooser_is_even_odds_and_prior_odds_shift_the_logit_only():
    net = MemberChooser(len(AFT_INPUTS), 200, len(GRAIN_INPUTS), aft_flags=AFT_FLAGS)
    b = _batch(64)
    net.fit_normalisation(b)
    with torch.no_grad():
        s = net(b)
        assert s.abs().max() < 0.05
        assert torch.allclose(net.log_odds(b, prior_odds=4.0), s + math.log(4.0))
        assert torch.allclose(net.prob(b, 3.0), torch.sigmoid(s + math.log(3.0)))


def test_join_batches_labels_each_half_by_its_member():
    a, b = _batch(3, seed=1), _batch(5, seed=2)
    feats, y = join_batches({**a, "T": torch.zeros(3, 16)}, {**b, "T": torch.zeros(5, 24)})
    assert set(feats) == {"aft", "length_pdf", "grains", "grain_mask"}       # paths never reach it
    assert y.tolist() == [0.0] * 3 + [1.0] * 5
    assert torch.equal(feats["aft"][3:], b["aft"])


def test_the_chooser_learns_which_prior_drew_a_sample_and_round_trips(tiny_corpus, tiny_corpus_long, tmp_path):
    """On the tiny family the long member's ages run to 250 Ma: a chooser that
    cannot separate them has learned nothing, and one that is sure of every row
    has learned the corpus rather than the ratio."""
    cfg = ChooserConfig(corpora=(tiny_corpus, tiny_corpus_long), batch_size=512, steps=400, val_every=200,
                        val_rows=1000, num_workers=0, warmup=50, lr=2e-3)
    net, hist = train_chooser(cfg, device="cpu", out_dir=tmp_path, log=lambda *_: None)
    assert (tmp_path / "model.pt").exists() and len(hist) == 2
    best = min(h["val_logloss"] for h in hist)
    assert best < 0.55, "no better than a coin"
    assert best > 0.02, "a perfect score on overlapping priors is overfitting"
    again = MemberChooser.load(tmp_path / "model.pt")
    from chai.data.observe import ObservationModel
    from chai.train.chooser import _val_batches
    from chai.train.data import CorpusSplit
    vals = [CorpusSplit(c, "val", max_rows=500) for c in cfg.corpora]
    batches = _val_batches(vals, ObservationModel(), cfg)
    v1, v2 = validate_chooser(net, batches, "cpu"), validate_chooser(again, batches, "cpu")
    assert np.array_equal(v1["logit"], v2["logit"])
    assert again.members == (100, 250)
