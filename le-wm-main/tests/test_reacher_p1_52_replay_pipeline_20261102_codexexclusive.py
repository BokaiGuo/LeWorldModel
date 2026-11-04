import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_p1_52_replay_contract_20261102_codexexclusive as contract
from experiments.spectral_reachability import reacher_p1_52_replay_consumer_20261102_codexexclusive as consumer
from experiments.spectral_reachability import reacher_p1_52_replay_host_20261102_codexexclusive as runner


def test_action_packet_time_order_roundtrip_and_bounds():
    actions = np.arange(30, dtype=np.float32).reshape(15, 2) / 100
    mean = np.linspace(-.1, .1, 10, dtype=np.float32)
    std = np.linspace(.5, 1.5, 10, dtype=np.float32)
    got = contract.normalized_suffix(actions, mean, std, np.full(10, -3), np.full(10, 3))
    np.testing.assert_allclose(got * std + mean, actions[5:].reshape(2, 10), atol=1e-7)
    with pytest.raises(ValueError):
        contract.normalized_suffix(actions, mean, std, np.full(10, -3), np.zeros(10))
    with pytest.raises(ValueError):
        contract.normalized_suffix(actions[:-1], mean, std, np.full(10, -3), np.full(10, 3))


def test_metric_shape_and_nonfinite_fail_closed():
    values = contract.metrics([1, 2], [0, 2], [0, 0], 3)
    assert values["prediction_error"] == 1
    assert values["physical_distance"] == 3
    with pytest.raises(ValueError):
        contract.metrics([1], [1, 2], [0], 1)
    with pytest.raises(ValueError):
        contract.metrics([float("nan")], [1], [0], 1)


def test_failure_before_physics_recorded_without_model_or_step():
    rows = json.loads(Path(contract.ROSTER).read_text())["rows"]
    row = rows[0]
    events = []
    journal = SimpleNamespace(emit=events.append)
    env = SimpleNamespace(step=lambda action: pytest.fail("must not step"))

    def fail_warmup(*args, **kwargs):
        raise RuntimeError("synthetic reset failure")

    setup = SimpleNamespace(collect_executed_context=fail_warmup, SOURCE_WARMUP_POLICY="fake")
    runner.replay_arm(row, {"warmup_packets": np.zeros((3, 5, 2)).tolist()}, env,
                      setup, None, None, None, None, None, None, None, journal, None)
    assert [e["kind"] for e in events] == ["arm_started", "arm_failed"]
    assert events[-1]["returned_steps"] == 0
    assert events[-1]["error"] == "synthetic reset failure"
    assert callable(env.step)


def test_partial_consumer_retains_failed_and_51_missing():
    rows = json.loads(Path(contract.ROSTER).read_text())["rows"]
    row = rows[0]
    key = {k: row[k] for k in ("group", "seed", "arm")}
    events = [{"kind": "run_started", "plan_sha256": "plan"},
              {"kind": "arm_started", **key, "action_sha256": row["action_sha256"]},
              {"kind": "arm_failed", **key, "returned_steps": 0, "error": "x"},
              {"kind": "run_error", "error": "x"}]
    roots = [{"seed": seed, "warmup_packets": np.zeros((3, 5, 2)).tolist()}
             for seed in (20263160, 20263161)]
    result = consumer.consume_events(events, rows, "plan", "model", roots, np.zeros(10), np.ones(10))
    assert result["status"] == "partial"
    assert result["rows"][0]["status"] == "failed"
    assert sum(r["status"] == "not_started" for r in result["rows"]) == 51
    events[1]["action_sha256"] = "wrong"
    with pytest.raises(ValueError):
        consumer.consume_events(events, rows, "plan", "model", roots, np.zeros(10), np.ones(10))
