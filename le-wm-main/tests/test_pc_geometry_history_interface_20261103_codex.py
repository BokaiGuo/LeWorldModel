import json
import os
import subprocess

import numpy as np

from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments import pc_geometry_history_interface_20261103_codex as audit
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base


def test_history_receipt_and_native_zero_warmup(tmp_path):
    report = json.loads(audit.OUTPUT.read_text())
    assert report["runner_sha256"] == base.digest(audit.Path(audit.__file__))
    assert report["protocol_sha256"] == base.digest(audit.PROTOCOL)
    assert report["decision_sha256"] == base.digest(audit.old.OUTPUT)
    assert report["physical_sha256"] == base.digest(audit.physical.OUTPUT)
    assert report["arrays_sha256"] == base.digest(audit.ARRAYS)
    request = {"rows": [{"start": list(start), "policy": "zero_history",
                          "actions": [[0., 0.]]*15} for start in audit.old.STARTS]}
    req, xml, out = tmp_path / "request.json", tmp_path / "xml", tmp_path / "output.json"
    req.write_text(json.dumps(request))
    xml.write_text(official_xml())
    env = os.environ.copy()
    env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
    env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                       "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
    subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                    str(req), str(xml), str(out)], env=env, check=True, timeout=120)
    physical = json.loads(out.read_text())
    with np.load(audit.ARRAYS, allow_pickle=False) as images, \
            np.load(audit.old.ARRAYS, allow_pickle=False) as original:
        for root, row in enumerate(physical["rows"]):
            initial = np.asarray(row["trace"][0]["state"])
            for state in row["trace"]:
                np.testing.assert_array_equal(state["state"], initial)
            history = images[f"root{root}_history_rgb"]
            assert history.shape == (16, 224, 224, 3)
            np.testing.assert_array_equal(history[-1], original[f"root{root}_current_rgb"])
    for row in report["rows"]:
        assert row["single_source_max_error"] <= 1e-5
        for arm in row["arms"].values():
            pred = np.asarray(arm["predicted_cost"])
            true = np.asarray(row["actual_physical_distance"])
            assert arm["selected_index"] == int(np.argmin(pred))
            np.testing.assert_allclose(arm["absolute_proxy_vs_physical_error"], abs(pred-true))
            reversals = sum((pred[i]-pred[j])*(true[i]-true[j]) < 0
                            for i in range(12) for j in range(i+1, 12))
            assert arm["pairwise_reversals"] == reversals
