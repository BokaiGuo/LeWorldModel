import json

import numpy as np
import pytest

from experiments import pc_pusht_continuous_unprivileged_plan_dev_20261105_codex as run


def test_decision_requires_selected_full_chain_in_scored_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    path = tmp_path / "root_9_cem"
    (path / "input").mkdir(parents=True)
    (path / "output").mkdir()
    for name in ("context.npz", "config.json", "model_metadata.json"):
        (path / "input" / name).write_bytes(name.encode())
    selected = np.zeros((15, 2), np.float32).tolist()
    record = {"arm": "cem", "step": 0, "worker_sha256": run.sha(run.WORKER),
              "input_sha256": run.sha(path / "input/context.npz"),
              "config_sha256": run.sha(path / "input/config.json"),
              "metadata_sha256": run.sha(path / "input/model_metadata.json"),
              "decision": {"selected": {"chain": selected, "terminal_score": 1.0}},
              "calls": [{"prediction_tokens": 3}], "resource": {"prediction_tokens": 3},
              "evaluations": [{"actions": [selected], "scores": [1.0]}]}
    (path / "output/decision.json").write_text(json.dumps(record))
    _, chain = run.check_decision(9, "cem")
    np.testing.assert_array_equal(chain, np.zeros((15, 2), np.float32))
    record["evaluations"][0]["scores"][0] = 2.0
    (path / "output/decision.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not bound"):
        run.check_decision(9, "cem")
