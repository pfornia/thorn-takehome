"""Tests for the autosearch pipeline.

Deliberately fast: no full searches, just the invariants that would silently break the pipeline.
Run with `pytest tests/test_autosearch.py`.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from stage2_autosearch.scoring import Scorer
from stage2_autosearch.search import Search, _expand_grid, _is_valid, _neighbors
from stage2_autosearch.transforms import CANVAS, REGISTRY, apply_transform

HOMEWORK = Path(__file__).resolve().parent.parent / "homework"
IMAGES = sorted(p for p in (HOMEWORK / "images").glob("*") if p.is_file())


@pytest.fixture(scope="module")
def sources():
    assert IMAGES, f"no base images found in {HOMEWORK / 'images'}"
    return [Image.open(p).convert("RGB") for p in IMAGES]


@pytest.fixture(scope="module")
def scorer():
    return Scorer()


@pytest.fixture(scope="module")
def config():
    return json.loads((HOMEWORK.parent / "stage2_autosearch" / "config.json").read_text())


# --- transforms -------------------------------------------------------------

@pytest.mark.parametrize("name,params", [
    ("collage", {"grid": 10, "line_width": 8, "line_color": "black", "seed": 0}),
    ("watermark", {"font_size": 12, "spacing": 20, "opacity": 0.6, "angle": 30}),
    ("degrade", {"kind": "gaussian", "strength": 40, "seed": 0}),
    ("degrade", {"kind": "uniform_blend", "strength": 0.5, "seed": 0}),
    ("degrade", {"kind": "salt_pepper", "strength": 0.2, "seed": 0}),
])
def test_transforms_produce_valid_images(sources, name, params):
    img = apply_transform(name, sources, params)
    assert img.size == (CANVAS, CANVAS)
    assert img.mode == "RGB"


def test_transforms_are_deterministic(sources):
    """Same params + same seed must reproduce byte-identical output, or reported
    scores can't be verified by anyone re-running the pipeline."""
    p = {"grid": 10, "line_width": 8, "line_color": "black", "seed": 7}
    a = apply_transform("collage", sources, p)
    b = apply_transform("collage", sources, p)
    assert a.tobytes() == b.tobytes()


def test_collage_seed_changes_output(sources):
    base = {"grid": 10, "line_width": 8, "line_color": "black"}
    a = apply_transform("collage", sources, {**base, "seed": 1})
    b = apply_transform("collage", sources, {**base, "seed": 2})
    assert a.tobytes() != b.tobytes()


def test_unknown_transform_raises(sources):
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transform("nope", sources, {})


# --- scoring ----------------------------------------------------------------

def test_baseline_images_are_other(scorer, sources):
    """The provided base images must classify as 'other' -- if this fails, any
    'false positive' we report downstream is meaningless."""
    for path, s in zip(IMAGES, scorer.score_batch(sources)):
        assert s.pred == "other", f"{path.name} predicted {s.pred}"


def test_probs_sum_to_one(scorer, sources):
    for s in scorer.score_batch(sources[:2]):
        assert abs(sum(s.probs.values()) - 1.0) < 1e-5


def test_batching_matches_single(scorer, sources):
    """Batched scoring must agree with one-at-a-time, or the whole search is
    measuring something different from the provided test harness."""
    batch = scorer.score_batch(sources[:3])
    for img, b in zip(sources[:3], batch):
        single = scorer.score(img)
        assert abs(single.probs["dog"] - b.probs["dog"]) < 1e-5


def test_margin_matches_logit_difference(scorer, sources):
    s = scorer.score(sources[0])
    assert abs(s.margin("dog") - (s.logits["dog"] - s.logits["other"])) < 1e-9


# --- search machinery -------------------------------------------------------

def test_expand_grid_skips_underscore_keys():
    grid = _expand_grid({"a": [1, 2], "b": ["x"], "_comment": "ignored"})
    assert len(grid) == 2
    assert all("_comment" not in g for g in grid)


@pytest.mark.parametrize("params,expected", [
    ({"kind": "gaussian", "strength": 40}, True),
    ({"kind": "gaussian", "strength": 0.5}, False),
    ({"kind": "salt_pepper", "strength": 0.5}, True),
    ({"kind": "salt_pepper", "strength": 40}, False),
])
def test_degrade_validity_filter(params, expected):
    assert _is_valid("degrade", params) is expected


def test_collage_rejects_gutters_wider_than_tile():
    assert _is_valid("collage", {"grid": 32, "line_width": 12}) is False
    assert _is_valid("collage", {"grid": 10, "line_width": 8}) is True


def test_neighbors_stay_positive_and_valid():
    cfg = {"numeric_deltas": {"grid": [-20, 2]}, "seeds": 3}
    out = _neighbors({"grid": 10, "line_width": 4, "seed": 0}, cfg, "collage")
    assert all(p["grid"] > 0 for p in out)
    assert sum(1 for p in out if p["seed"] != 0) == 3


def test_neighbors_respect_numeric_max():
    """Refinement must not step a parameter over its configured ceiling.

    This is the exact case the cap exists for: the coarse grid tops out at the ceiling, so the
    best coarse result sits *on* it and every positive delta lands outside the intended range.
    """
    cfg = {"numeric_deltas": {"strength": [-10, 5, 20]}, "numeric_max": {"strength": 100},
           "seeds": 0}
    out = _neighbors({"kind": "gaussian", "strength": 100, "seed": 0}, cfg, "degrade")
    assert out, "expected the downward step to survive"
    assert all(p["strength"] <= 100 for p in out)
    assert {p["strength"] for p in out} == {90}


def test_uf3_sigma_is_capped_in_both_stages(config):
    """The recognisability cap has to hold in the coarse grid and in refinement.

    Capping only the grid would leak: see test_neighbors_respect_numeric_max.
    """
    uf3 = config["uf3"]
    sigmas = [s for s in uf3["coarse"]["strength"] if s > 1]
    assert max(sigmas) <= 100
    assert uf3["refine"]["numeric_max"]["strength"] == 100


# --- config integrity -------------------------------------------------------

def test_config_blocks_are_wellformed(config):
    for uf in ("uf1", "uf2", "uf3"):
        block = config[uf]
        assert block["transform"] in REGISTRY
        assert block["coarse"], f"{uf} has an empty coarse grid"
        assert "complaint" in block, f"{uf} must record the user complaint it targets"


def test_config_coarse_params_are_accepted_by_their_transform(sources, config):
    """Every coarse grid must produce a renderable image -- catches config typos
    before they waste a full search run."""
    for uf in ("uf1", "uf2", "uf3"):
        block = config[uf]
        valid = [p for p in _expand_grid(block["coarse"]) if _is_valid(block["transform"], p)]
        assert valid, f"{uf}: every coarse combination was filtered out as invalid"
        apply_transform(block["transform"], sources, valid[0])


# --- end to end -------------------------------------------------------------

def test_search_finds_a_false_positive(scorer, tmp_path):
    """A minimal end-to-end run on the complaint with the smallest search space."""
    search = Search(scorer=scorer, verbose=False)
    search.config["uf3"]["coarse"] = {"kind": ["gaussian"], "strength": [60, 120], "seed": [0]}
    search.config["uf3"]["refine"] = {"numeric_deltas": {"strength": [20]}, "seeds": 2}
    report = search.run("uf3", [HOMEWORK / "images" / "woman.jpg"], out_dir=tmp_path)

    assert report["best"]["pred"] == "dog"
    assert report["best"]["probs"]["dog"] > 0.5
    assert all(b["pred"] == "other" for b in report["baseline"])
    assert list(tmp_path.glob("*.png")), "no output image written"
    assert list(tmp_path.glob("*__ORIGINAL.png")), "original not saved alongside"


# --- target label is configurable (cat as well as dog) -----------------------

def test_target_label_defaults_to_dog(scorer):
    assert Search(scorer=scorer, verbose=False).label == "dog"


def test_target_label_can_be_overridden(scorer):
    assert Search(scorer=scorer, verbose=False, target_label="cat").label == "cat"


def test_invalid_target_label_rejected(scorer):
    with pytest.raises(ValueError, match="not in model labels"):
        Search(scorer=scorer, verbose=False, target_label="giraffe")


def test_candidate_prob_is_not_hardcoded_to_dog(scorer, sources):
    from stage2_autosearch.search import Candidate
    c = Candidate("degrade", {}, "x", scorer.score(sources[0]))
    for label in ("cat", "dog", "other"):
        assert c.prob(label) == c.score.probs[label]


def test_search_can_target_cat(scorer, tmp_path):
    """End-to-end run targeting `cat` rather than `dog`."""
    search = Search(scorer=scorer, verbose=False, target_label="cat")
    search.config["uf3"]["coarse"] = {"kind": ["gaussian"], "strength": [40, 80], "seed": [0]}
    search.config["uf3"]["refine"] = {"numeric_deltas": {"strength": [20]}, "seeds": 2}
    report = search.run("uf3", [HOMEWORK / "images" / "woman.jpg"], out_dir=tmp_path)
    assert report["target_label"] == "cat"
    # filenames should carry the targeted label, not a hardcoded 'dog'
    assert any("cat" in p.name for p in tmp_path.glob("*.png"))


def test_margin_vs_best_other_is_not_gameable(scorer, sources):
    """Regression test for a real bug: ranking on margin-vs-'other' selected images
    the model called `dog` with near-certainty when the target was `cat`, because
    cat beat 'other' while dog took all the mass. Margin against the strongest
    competitor is positive only when the target class actually wins."""
    from stage2_autosearch.transforms import apply_transform
    img = apply_transform("degrade", sources, {"kind": "gaussian", "strength": 160, "seed": 0})
    s = scorer.score(img)
    for label in ("cat", "dog", "other"):
        assert (s.margin_vs_best_other(label) > 0) == (s.pred == label)


# --- classify.py utility ----------------------------------------------------

def test_classify_cli_reports_probabilities(capsys):
    import classify
    rc = classify.main([str(HOMEWORK / "images" / "woman.jpg")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "woman.jpg" in out and "other" in out
    # probabilities, not logits, unless asked
    assert "logit" not in out


def test_classify_cli_json_and_logits(capsys):
    import classify
    rc = classify.main(["--json", "--logits", str(HOMEWORK / "images" / "woman.jpg")])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    row = payload[0]
    assert row["pred"] == "other"
    assert abs(sum(row["probs"].values()) - 1.0) < 1e-4
    assert set(row["logits"]) == {"cat", "dog", "other"}


def test_classify_cli_missing_file_exits_nonzero(capsys):
    import classify
    assert classify.main(["definitely_not_here.png"]) == 2
