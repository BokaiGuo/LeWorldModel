from copy import deepcopy

import pytest

from experiments.spectral_reachability.p1_backtrack_rgb_feedback_analysis_20261030 import analyze


def fixture():
    prep = dict(fixed_rows=16, model_hash="hash", rows=[])
    report = dict(schema="p1_backtrack_rgb_feedback_cuda_20261030.report", fixed_rows=16,
                  model_before="hash", model_after="hash",
                  calls=dict(encoder=80, projector=80, predictor=32, action_encoder=32),
                  training=False, P1_admission=False, freshness=False, rows=[])
    for i in range(16):
        responses = [dict(step=j, native_success=False, native_metrics=dict(
            object_position_error=1.0, wrapped_object_angle_error=0.5)) for j in range(1, 16)]
        r = dict(index=i, slot=i // 4, arm=("a", "b")[(i // 2) % 2],
                 role=("incumbent", "model_selected")[i % 2], native_responses=responses,
                 open_predictions=[[float(i)] * 192 for _ in range(3)])
        prep["rows"].append(dict(**{k: deepcopy(r[k]) for k in
                                    ("index", "slot", "arm", "role", "native_responses")},
                                 open_endpoints=deepcopy(r["open_predictions"])))
        r.update(actual_latents=[[0.] * 192 for _ in range(4)], goal_latent=[0.] * 192,
                 feedback_predictions=[[0.] * 192 for _ in range(2)])
        report["rows"].append(r)
    return report, prep


def test_fixed_pairs_and_mse():
    report, prepared = fixture()
    result = analyze(report, prepared)
    assert len(result["pairs"]) == 8
    assert result["rows"][1]["times"][1]["open_mse"] == 1.0
    assert result["rows"][1]["times"][1]["feedback_mse"] == 0.0
    assert result["pairs"][0]["times"][0]["selected_minus_incumbent"]["open_goal_distance"] > 0


def test_rejects_mutated_source_or_nonfinite():
    report, prepared = fixture()
    report["rows"][0]["open_predictions"][0][0] = 4.
    with pytest.raises(ValueError, match="identity"):
        analyze(report, prepared)
    report, prepared = fixture()
    report["rows"][0]["actual_latents"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analyze(report, prepared)
