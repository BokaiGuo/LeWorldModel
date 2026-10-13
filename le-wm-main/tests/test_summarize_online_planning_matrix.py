import json

import pytest

from experiments.summarize_online_planning_matrix import summarize


def _run(tmp_path, normalization, offset=0.0):
    path = tmp_path / f"{normalization}.json"
    policies = []
    for policy, base in (("zero", 10.0), ("vanilla", 8.0)):
        policies.append({
            "policy": policy,
            "episodes": [
                {"episode": i, "success": i == 0 and policy == "vanilla", "final_distance": base + i + offset, "block_xy_distance": base + i, "return": -(base + i), "clip_fraction": 0.0, "steps": 2}
                for i in range(4)
            ],
            "summary": {},
        })
    path.write_text(json.dumps({
        "protocol": {
            "action_normalization": normalization,
            "episodes": 4,
            "seed": 7,
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "samples": 8,
            "cem_steps": 1,
            "topk": 4,
            "policies": ["zero", "vanilla"],
        },
        "reports": policies,
    }))
    return path


def test_matched_summary_reports_paired_delta(tmp_path):
    report = summarize([_run(tmp_path, "identity"), _run(tmp_path, "live_zscore", offset=1.0)], draws=100)
    assert report["protocol"]["status"] == "partial_cpu_pilot"
    assert report["sanity_checks"]["finite_all"]
    assert report["paired_vs_zero"]["identity"]["vanilla"]["success"]["estimate"] == 0.25


def test_summary_rejects_protocol_mismatch(tmp_path):
    first = _run(tmp_path, "identity")
    second = _run(tmp_path, "live_zscore")
    data = json.loads(second.read_text())
    data["protocol"]["horizon"] = 1
    second.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol-matched"):
        summarize([first, second], draws=20)
