import hashlib
import json

import mujoco
import numpy as np

from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as exp


def test_frozen_bank_and_roots():
    assert exp.bank().shape == (32, 2)
    assert np.all(np.abs(exp.bank()) <= 1)
    assert [s.tolist() for s in exp.obstacle_intersection_starts()] == [
        [0.8, 0.8], [0.8, 1.8], [2.8, 0.8], [2.8, 1.8]
    ]


def test_receipt_replays_persistent_actions_and_stopping():
    result = json.loads(exp.OUTPUT.read_text())
    assert result["mujoco"] == mujoco.__version__ == "3.1.6"
    assert result["protocol_sha256"] == hashlib.sha256(exp.PROTOCOL.read_bytes()).hexdigest()
    assert result["script_sha256"] == exp.digest(exp.Path(exp.__file__))
    assert result["xml_sha256"] == exp.digest(exp.XML)
    assert len(result["rows"]) == 8
    model = mujoco.MjModel.from_xml_string(exp.XML.read_text())
    for row in result["rows"]:
        data = mujoco.MjData(model)
        data.qpos[:] = row["start"]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        assert 1 <= len(row["trace"]) <= 24
        for index, record in enumerate(row["trace"]):
            assert record["action_index"] == int(np.argmin(record["candidate_scores"]))
            np.testing.assert_allclose(record["action"], exp.bank()[record["action_index"]], atol=1e-14)
            exp.step(model, data, np.asarray(record["action"]))
            np.testing.assert_allclose(data.qpos[:2], record["position"], atol=1e-12)
            np.testing.assert_allclose(data.qvel[:2], record["velocity"], atol=1e-12)
            assert abs(np.linalg.norm(data.qpos[:2] - exp.GOAL) - record["distance"]) < 1e-12
            assert record["contact_count"] == data.ncon
            if index < len(row["trace"]) - 1:
                assert record["distance"] > 0.25
        assert row["success"] == (row["trace"][-1]["distance"] <= 0.25)
        assert row["stop"] == ("success" if row["success"] else "horizon")
