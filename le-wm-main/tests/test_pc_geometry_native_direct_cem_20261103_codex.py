import json
import os
import subprocess

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_geometry_native_direct_cem_20261103_codex as experiment
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base


def test_direct_curve_and_winner_physical_replay(tmp_path):
    receipt = json.loads(experiment.OUTPUT.read_text())
    assert receipt["binding"] == "2.1.2.14"
    assert receipt["protocol_sha256"] == base.digest(experiment.PROTOCOL)
    assert receipt["worker_sha256"] == base.digest(experiment.WORKER)
    assert receipt["launcher_sha256"] == base.digest(experiment.Path(experiment.__file__))
    request, expected = [], []
    assert len(receipt["rows"]) == 4
    for row in receipt["rows"]:
        assert len(row["evaluations"]) == 1200
        assert [point["evaluations"] for point in row["budget_curve"]] == [100, 300, 600, 1200]
        for point in row["budget_curve"]:
            evaluations = row["evaluations"][:point["evaluations"]]
            index = int(np.argmin([item["distance"] for item in evaluations]))
            assert point["best_index"] == index
            assert point["best_distance"] == evaluations[index]["distance"]
            request.append({"start": row["start"], "policy": "full_cem",
                            "actions": evaluations[index]["actions"]})
            expected.append(point["best_distance"])
    request_path, xml, output = tmp_path / "request.json", tmp_path / "official.xml", tmp_path / "output.json"
    request_path.write_text(json.dumps({"rows": request}))
    xml.write_text(official_xml())
    assert base.digest(xml) == receipt["xml_sha256"]
    env = os.environ.copy()
    env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
    env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                       "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
    subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                    str(request_path), str(xml), str(output)], env=env, check=True, timeout=120)
    rows = json.loads(output.read_text())["rows"]
    assert len(rows) == len(expected) == 16
    for row, distance in zip(rows, expected, strict=True):
        terminal = np.asarray(row["trace"][-1]["state"][:2])
        np.testing.assert_allclose(np.linalg.norm(terminal-base.GOAL), distance, atol=1e-12)
