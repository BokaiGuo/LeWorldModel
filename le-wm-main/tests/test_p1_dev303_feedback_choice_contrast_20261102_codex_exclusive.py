import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_feedback_choice_contrast_20261102_codex_exclusive as contrast
from test_p1_dev303_feedback_live_20261102_codex_exclusive import (
    FakeEnv, FakeScorer, capture, fixture_data,
)


def image():
    pixels = np.zeros((224, 224, 3), np.uint8)
    pixels[98:103, 18:23] = contrast.feedback.rgb.BLUE
    pixels[98:103, 80:85] = contrast.feedback.rgb.BLOCK
    return pixels


class SyntheticScorer(contrast.feedback.CUDARealScorer):
    """Mock scores only; this test executes no model or CUDA."""
    def __init__(self):
        self.calls = 0

    def begin(self, frames, actions):
        self.history = actions.copy()

    def score(self, variants):
        self.calls += 1
        result = np.ones((64, 5), np.float32)
        result[0] = 100
        return result

    def finish(self):
        assert self.calls == 5
        return {"synthetic_scoring_batches": self.calls}

    def close(self):
        pass


def controller():
    return contrast.feedback.Feedback([image()] * 11, np.zeros((10, 2), np.float32),
                                      frame_ids=list(range(11)))


def test_same_scoring_and_fixed_intervention():
    model, direct = controller(), controller()
    model_scorer, direct_scorer = SyntheticScorer(), SyntheticScorer()
    a = contrast.decision(model, model_scorer, "model_choice")
    b = contrast.decision(direct, direct_scorer, "direct_approach")
    assert model_scorer.calls == direct_scorer.calls == 5
    assert a["model_decision"] == b["model_decision"]
    assert a["action"] == a["model_decision"]["action"]
    assert b["action"] == b["model_decision"]["approach_candidate"]
    assert a["action"] != b["action"]
    assert b["seed_passed_existing_geometry_guard"]
    assert np.array_equal(direct.pending, np.asarray(b["model_decision"]["action"], np.float32))


def test_bad_method_never_calls_model():
    scorer = SyntheticScorer()
    with pytest.raises(ValueError):
        contrast.decision(controller(), scorer, "choose_after_outcome")
    assert scorer.calls == 0


def test_clipped_geometry_abstains_before_selection():
    pixels = image()
    pixels[100, 0] = contrast.feedback.rgb.BLOCK
    control = contrast.feedback.Feedback([pixels] * 11, np.zeros((10, 2)),
                                         frame_ids=list(range(11)))
    scorer = SyntheticScorer()
    with pytest.raises(ValueError):
        contrast.decision(control, scorer, "direct_approach")
    assert scorer.calls == 0 and control.closed


def test_contract_source_and_fixed_four_arm_design():
    contract = contrast.build()
    assert [(x["prefix"], x["method"]) for x in contract["arms_in_fixed_order"]] == [
        ("rgb_chunk", "model_choice"), ("rgb_chunk", "direct_approach"),
        ("zero_prefix", "model_choice"), ("zero_prefix", "direct_approach")]
    assert not contract["physical_execution_authorized_by_contract"]
    assert not contract["cuda_execution_authorized_by_contract"]
    assert not contract["full_cost_certified"] and not contract["P1_admission"]


@pytest.mark.parametrize("method", ["model_choice", "direct_approach"])
def test_complete_synthetic_branch_logs_frames_actions_and_cost(tmp_path, monkeypatch, method):
    role, source = fixture_data()
    monkeypatch.setattr(contrast.base, "capture", capture)
    monkeypatch.setattr(contrast.live, "RecordedScorer", FakeScorer)
    store = contrast.base.Store(tmp_path / "out")
    row = {"arm": role["arm"], "prefix": role["arm"], "method": method,
           "status": "running", "transitions": [],
           "observations": [], "decisions": [], "error": None}
    env = FakeEnv()
    store.emit("branch_started", arm=role["arm"], prefix=role["arm"], method=method)
    contrast.branch(env, store, role, method, source, {"state": [0.]},
                    np.zeros((2, 2, 3), np.uint8), None, None, None, row)
    store.emit("branch_closed", arm=role["arm"], status=row["status"])
    assert row["status"] == "feedback_limit_completed" and env.n == 20
    assert len(row["observations"]) == 21 and len(row["transitions"]) == 20
    assert len(row["decisions"]) == 5
    assert len([e for e in store.events if e["kind"] == "score_batch"]) == 25
    for entry, transition in zip(row["decisions"], row["transitions"][-5:], strict=True):
        assert entry["method"] == method and entry["action"] == transition["action"]
        assert entry["model_decision"]["budget"]["predict_samples"] == 3200
    spec = {"id": role["arm"], "prefix": role["arm"], "method": method}
    assert contrast.saved.verify_branch(store.path, role, spec, row, store.events,
                                        source["history"])["decisions"] == 5
    score_event = next(e for e in store.events if e["kind"] == "score_batch")
    (store.path / score_event["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        contrast.saved.verify_branch(store.path, role, spec, row, store.events, source["history"])


def test_source_mismatch_blocks_all_planning(tmp_path, monkeypatch):
    role, source = fixture_data()
    role["prefix_checkpoints"][-1]["pixel_sha256"] = "mismatch"
    monkeypatch.setattr(contrast.base, "capture", capture)
    monkeypatch.setattr(contrast.live, "RecordedScorer", FakeScorer)
    store = contrast.base.Store(tmp_path / "out")
    row = {"arm": role["arm"], "status": "running", "transitions": [],
           "observations": [], "decisions": [], "error": None}
    with pytest.raises(ValueError):
        contrast.branch(FakeEnv(), store, role, "direct_approach", source, {"state": [0.]},
                        np.zeros((2, 2, 3), np.uint8), None, None, None, row)
    assert not any(e["kind"] == "planning_started" for e in store.events)
