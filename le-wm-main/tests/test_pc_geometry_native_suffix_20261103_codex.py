import json
import os
import subprocess

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_geometry_native_suffix_20261103_codex as experiment
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base


def test_native_receipt_contract_and_full_physical_replay(tmp_path):
    receipt = json.loads(experiment.OUTPUT.read_text())
    assert receipt["binding"] == "2.1.2.14"
    assert receipt["protocol_sha256"] == base.digest(experiment.PROTOCOL)
    assert receipt["worker_sha256"] == base.digest(experiment.WORKER)
    assert receipt["launcher_sha256"] == base.digest(experiment.Path(experiment.__file__))
    assert len(receipt["rows"]) == 48
    common = np.asarray(receipt["common_suffixes"])
    assert common.shape == (6, 10, 2)
    np.testing.assert_array_equal(common[0], 0)
    requests, expected = [], []
    for row in receipt["rows"]:
        prefix = np.asarray(row["prefix_actions"])
        assert prefix.shape == (5, 2)
        assert row["non_improving"] == (row["prefix_distance"] >= row["initial_distance"])
        assert len(row["search"]["evaluations"]) == 100
        values = [x["distance"] for x in row["search"]["evaluations"]]
        assert row["search"]["best_index"] == int(np.argmin(values))
        assert row["search"]["best_distance"] == min(values)
        np.testing.assert_allclose(row["common_change_vs_zero"],
                                   np.asarray(row["common_distances"]) - row["common_distances"][0])
        for suffix, distance in zip(common, row["common_distances"], strict=True):
            requests.append({"start": row["start"], "policy": row["prefix_name"],
                             "actions": np.concatenate((prefix, suffix)).tolist()})
            expected.append(distance)
        winner = row["search"]["evaluations"][row["search"]["best_index"]]
        requests.append({"start": row["start"], "policy": row["prefix_name"],
                         "actions": np.concatenate((prefix, winner["suffix"])).tolist()})
        expected.append(winner["distance"])
    request, xml, output = tmp_path / "input.json", tmp_path / "official.xml", tmp_path / "result.json"
    request.write_text(json.dumps({"rows": requests}))
    xml.write_text(official_xml())
    assert base.digest(xml) == receipt["xml_sha256"]
    env = os.environ.copy()
    env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
    env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                       "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
    subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                    str(request), str(xml), str(output)], check=True, env=env, timeout=120)
    replayed = json.loads(output.read_text())
    assert len(replayed["rows"]) == len(expected) == 336
    for row, distance in zip(replayed["rows"], expected, strict=True):
        assert len(row["trace"]) == 16
        terminal = np.asarray(row["trace"][-1]["state"][:2])
        np.testing.assert_allclose(np.linalg.norm(terminal - base.GOAL), distance, atol=1e-12)
