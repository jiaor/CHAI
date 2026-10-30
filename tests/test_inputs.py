"""Input files: a sample, its constraints and the settings, saved and loaded back as the web page does."""
from __future__ import annotations

import json

import pytest

from chai import api, inputs
from chai.constraints import WITHIN, Box

SAMPLE = api.Sample(
    aft=api.AFT(age=42.0, n_spontaneous=180, lengths=[13.1, 12.45, 14.0], angles=[10.0, 55.5, 80.0], dpar=1.9,
                projection="donelick1999"),
    grains=[api.Grain(31.0, 62.0, 40.0, age_sd=2.0), api.Grain(24.0, 55.0, 30.0, corrected=False)],
    name="KL-07 (ü)",
)


def test_a_saved_input_loads_back_equal(tmp_path):
    inp = inputs.Input(SAMPLE, boxes=[Box((55, 60), (0, 25), WITHIN, soft_sd=5.0, label="unconformity"),
                                      Box((20, 30), (60, 90))],
                       cooling_only=True, member=250, n=4000, seed=None, prior_odds=0.3)
    path = inp.save(tmp_path / "in.chai.json")
    back = inputs.load(path)
    assert back == inp
    assert back.sample.aft.projected_lengths == SAMPLE.aft.projected_lengths
    assert back.sample.grains[1].corrected_age == SAMPLE.grains[1].corrected_age
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["format"] == "chai-input" and d["version"] == inputs.VERSION and d["chai"]
    assert d["constraints"]["cooling_only"] is True and d["settings"]["seed"] is None


def test_a_hand_written_file_needs_only_the_format_and_the_sample(tmp_path):
    # the shape the web page writes: nulls for what was not given, no projection
    (tmp_path / "web.json").write_text(json.dumps({
        "format": "chai-input", "version": 1, "chai": None,
        "sample": {"name": "", "aft": {"age": 45.0, "dpar": 2.0, "n_spontaneous": None, "age_sd": 3.1,
                                       "lengths": [], "angles": None, "projection": None}, "grains": []}}))
    inp = inputs.load(tmp_path / "web.json")
    assert inp.sample.aft.age_sd == 3.1 and inp.sample.aft.n_spontaneous is None
    assert (inp.boxes, inp.cooling_only, inp.member, inp.n, inp.seed, inp.prior_odds) == ((), False, "auto", 2000, 0, 1.0)


def test_the_present_day_range_is_saved_only_when_set(tmp_path):
    # a family given the present day reads it; a file without it is what it was before
    inp = inputs.Input(SAMPLE, T_present=[6, 14.5])
    assert inp.T_present == (6.0, 14.5)
    back = inputs.load(inp.save(tmp_path / "p.json"))
    assert back == inp and back.to_dict()["settings"]["T_present"] == [6.0, 14.5]
    assert "T_present" not in inputs.Input(SAMPLE).to_dict()["settings"]
    with pytest.raises(ValueError, match="T_present"):
        inputs.Input(SAMPLE, T_present=[5.0])


@pytest.mark.parametrize("doc, match", [
    ({"sample": {}}, "not a CHAI input file"),
    ([1, 2], "not a CHAI input file"),
    ({"format": "chai-input", "version": 99, "sample": {}}, "newer"),
    ({"format": "chai-input", "sample": {"aft": None, "grains": []}}, "no data"),
    ({"format": "chai-input", "sample": {"grains": [{"age": 30.0, "radius": 60.0, "eU": 10.0}]},
      "settings": {"prior_odds": 0}}, "prior_odds"),
    ({"format": "chai-input", "sample": {"grains": [{"age": 30.0, "radius": 60.0, "eU": 10.0}]},
      "constraints": {"boxes": [{"t": [30, 20], "T": [0, 50]}]}}, "t_young <= t_old"),
])
def test_what_is_not_an_input_is_refused(doc, match):
    with pytest.raises(ValueError, match=match):
        inputs.Input.from_dict(doc)


def test_the_export_module_still_offers_the_sample_json():
    from chai import export

    assert export.sample_to_dict is inputs.sample_to_dict and export.sample_from_dict is inputs.sample_from_dict
