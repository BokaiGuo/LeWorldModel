import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/pc_pointmaze_rgb_metric_20261103.py"
spec = importlib.util.spec_from_file_location("pc_pointmaze_rgb_metric", PATH)
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


def image(x, y):
    value = np.zeros((224, 224, 3), dtype=np.uint8)
    value[y:y+3, x:x+3, 1] = 200
    return value


def test_score_depends_only_on_image_marker():
    scores, positions, goal, failures = metric.score_images(
        np.stack([image(50, 70), image(53, 74)]), image(50, 70))
    np.testing.assert_allclose(scores, [0, 5])
    np.testing.assert_allclose(positions[1], [54, 75])
    np.testing.assert_allclose(goal, [51, 71])
    assert failures == []


def test_detection_failures_are_explicit_and_not_imputed():
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    scores, _, _, errors = metric.score_images(np.stack([image(50, 70), blank]), image(50, 70))
    assert scores[0] == 0 and np.isnan(scores[1])
    assert errors[0]["index"] == 1
    scores, _, goal, errors = metric.score_images(np.stack([image(50, 70)]), blank)
    assert np.isnan(scores[0]) and goal is None and errors[0]["index"] == "goal"


def test_ranking_ties_and_reversals():
    assert metric.ranking([0, 0, 2], [2, 1, 0]) == {
        "comparable": 2, "reversed": 2, "first_ties": 1, "second_ties": 0}
    with pytest.raises(ValueError):
        metric.ranking([0, np.nan], [0, 1])


def test_saved_source_pairing_and_rgb_recomputation():
    meta_path = metric.PREFIX.with_name(metric.PREFIX.name + "_scores.json")
    if not meta_path.exists():
        pytest.skip("score stage not yet run")
    import json
    meta = json.loads(meta_path.read_text())
    rgb, oracle = metric.receipt(metric.RGB), metric.receipt(metric.ORACLE)
    assert rgb["oracle_sha256"] == metric.sha(metric.ORACLE)
    assert len(meta["groups"]) == len(oracle["rows"]) == 12
    offset = 0
    with np.load(metric.RGB.with_suffix(".npz"), allow_pickle=False) as r, \
         np.load(metric.PLANS.with_suffix(".npz"), allow_pickle=False) as p, \
         np.load(metric.ORACLE.with_suffix(".npz"), allow_pickle=False) as o, \
         np.load(metric.PREFIX.with_name(metric.PREFIX.name + "_scores.npz"), allow_pickle=False) as s:
        for group, original in zip(meta["groups"], oracle["rows"]):
            assert all(group[k] == original[k] for k in ("root", "method", "phase", "n"))
            name, n = metric.key(group), group["n"]
            calc, _, _, errors = metric.score_images(r["rgb10"][offset:offset+n], p[f"r{group['root']}_goal_rgb"])
            np.testing.assert_array_equal(calc, s[name + "_scores"])
            assert errors == group["failures"]
            assert o[name + "_distances"].shape == (n,)
            offset += n
    assert offset == 624


def test_saved_evaluation_recomputes_all_rankings():
    import json
    report_path = metric.PREFIX.with_name(metric.PREFIX.name + "_evaluation.json")
    if not report_path.exists():
        pytest.skip("evaluation stage not yet run")
    report = json.loads(report_path.read_text())
    with np.load(metric.RGB.with_suffix(".npz"), allow_pickle=False) as r, \
         np.load(metric.ORACLE.with_suffix(".npz"), allow_pickle=False) as o, \
         np.load(metric.PREFIX.with_name(metric.PREFIX.name + "_scores.npz"), allow_pickle=False) as s:
        totals = {}
        for row in report["rows"]:
            assert not row["failures"]
            name = metric.key(row)
            geom, physical = s[name + "_scores"], o[name + "_distances"]
            predicted, actual = o[name + "_scores"], r[name + "_real_cost"]
            for label, a, b in (("rgb_vs_physical", geom, physical),
                                ("predicted_latent_vs_physical", predicted, physical),
                                ("actual_rgb_latent_vs_physical", actual, physical),
                                ("predicted_latent_vs_rgb", predicted, geom),
                                ("actual_rgb_latent_vs_rgb", actual, geom)):
                computed = metric.ranking(a, b)
                assert row["rankings"][label] == computed
                totals.setdefault(label, {k: 0 for k in computed})
                for k, v in computed.items():
                    totals[label][k] += v
        assert totals == report["pair_weighted"]
