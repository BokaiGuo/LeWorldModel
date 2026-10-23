from experiments.decision_benchmark.freeze_pointmaze_probe_margin import freeze


def test_freeze_reads_one_margin_per_unique_design_root() -> None:
    report = {"rows": [
        {
            "method": "local", "episode_id": root,
            "active_probe_trace": [{
                "probe_decision": True,
                "probe_conditioned_visual_margin": 0.1 + 0.01 * root,
            }],
        }
        for root in range(8)
    ]}
    frozen = freeze(report, "local", [0.25, 0.50, 0.75])
    assert frozen["design_root_count"] == 8
    assert [item["label"] for item in frozen["thresholds"]] == ["q25", "q50", "q75"]
    assert frozen["thresholds"][0]["margin"] < frozen["thresholds"][-1]["margin"]
