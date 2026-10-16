import json

import pytest

from experiments.summarize_tworoom_method_comparison import main


def _panel(path, metric, successes, random=(0, 0, 0, 0)):
    seeds = list(range(10, 10 + len(successes)))
    payload = {
        "schema": "live_tworoom_planning.v1",
        "protocol": {
            "checkpoint": "frozen",
            "checkpoint_frozen": True,
            "device": "cuda",
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "cem": {"num_samples": 64, "n_steps": 8, "topk": 8},
            "candidate_bank": True,
            "runtime_input": "rgb",
            "runtime_state_access": False,
            "same_seed_pairing": True,
            "cost_metric": metric,
        },
        "random": {"seeds": seeds, "episode_successes": list(random)},
        "frozen_lewm": {"seeds": seeds, "episode_successes": list(successes)},
        "paired": {},
    }
    path.write_text(json.dumps(payload))


def test_method_comparison_is_paired_and_reports_delta(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    method = tmp_path / "method.json"
    _panel(baseline, "euclidean", [0, 1, 0, 0])
    _panel(method, "latent_action_ranker", [1, 1, 0, 0])
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--baseline", str(baseline), "--methods", str(method), "--output", str(output)],
    )
    main()
    report = json.loads(output.read_text())
    item = report["comparisons"][0]
    assert item["wins"] == 1
    assert item["losses"] == 0
    assert item["paired_success_rate_delta"] == 0.25


def test_method_comparison_rejects_seed_mismatch(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    method = tmp_path / "method.json"
    _panel(baseline, "euclidean", [0, 1, 0, 0])
    _panel(method, "latent_action_ranker", [1, 1, 0, 0])
    payload = json.loads(method.read_text())
    payload["frozen_lewm"]["seeds"][0] = 999
    payload["random"]["seeds"][0] = 999
    method.write_text(json.dumps(payload))
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--baseline", str(baseline), "--methods", str(method), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="seed panel mismatch"):
        main()
