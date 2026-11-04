"""Rebuild direct spectral probe comparisons from sealed arrays."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_reacher_direct_spectral_probe_v3_20261104_codex as run
from experiments.pc_v4_reacher_actual_fd_physical_audit_20261104_codex import vector_metrics
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.live.host.bridge.sha_file
    manifest, result = [json.loads(p.read_text()) for p in (run.MANIFEST, run.RESULT)]
    seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(run.MANIFEST) or result["arrays_sha256"] != sha(run.ARRAYS):
        raise ValueError("binding mismatch")
    decision = json.loads(run.prev.DECISION.read_text())
    with np.load(run.ARRAYS, allow_pickle=False) as f:
        a = {k: f[k].copy() for k in f.files}
    with np.load(run.old.ARRAYS, allow_pickle=False) as f:
        old_frames = f["frames"].copy()
    if a["actions"].shape != (18, 5, 2):
        raise ValueError("incorrect shape")
    np.testing.assert_array_equal(a["frames"][0], old_frames[0, 0])
    np.testing.assert_array_equal(a["frames"][9], old_frames[1, 0])
    np.testing.assert_allclose(a["chart"], decision["chart"], atol=1e-7, rtol=0)
    pred_pos, obs_pos = a["predicted"] @ a["chart"], a["observed"] @ a["chart"]
    pred_score = ((a["predicted"] - a["target"]) ** 2).mean(axis=1)
    obs_score = ((a["observed"] - a["target"]) ** 2).mean(axis=1)
    distance = np.asarray([s["distance"] for s in result["states"]])
    reports = []
    for k in range(2):
        matrix = np.asarray(decision["result"]["first_response"][k]["matrix"])
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        nominal = np.asarray(decision["result"]["first_response"][k]["nominal"])
        for mode in range(2):
            direction = vh[mode].reshape(5, 2) / np.max(np.abs(vh[mode]))
            np.testing.assert_allclose(direction, a["directions"][k, mode], rtol=0, atol=1e-12)
            for si, scale in enumerate((.05, .5)):
                minus = k * 9 + 1 + mode * 4 + si * 2
                plus = minus + 1
                for index, sign in ((minus, -1), (plus, 1)):
                    request = nominal + sign * scale * direction
                    np.testing.assert_allclose(a["requested"][index], request, rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(a["actions"][index], np.clip(request, -1, 1).astype(np.float32))
                    state = result["states"][index]
                    np.testing.assert_allclose(distance[index], run.live.p0._qpos_distance(
                        np.asarray(state["qpos"]), np.asarray(state["target_qpos"])), rtol=0, atol=1e-12)
                model = (pred_pos[plus] - pred_pos[minus]) / (2 * scale)
                physical = (obs_pos[plus] - obs_pos[minus]) / (2 * scale)
                reports.append({"nominal": k, "mode": mode, "scale": scale,
                    "clipped": result["metadata"][minus]["clipped_coordinates"] + result["metadata"][plus]["clipped_coordinates"],
                    "model_vs_actual": vector_metrics(model, physical),
                    "jvp_vs_model_fd": vector_metrics(a["jvps"][k, mode], model),
                    "pred_mse_minus_plus": [float(pred_score[minus]), float(pred_score[plus])],
                    "actual_mse_minus_plus": [float(obs_score[minus]), float(obs_score[plus])],
                    "distance_minus_plus": [float(distance[minus]), float(distance[plus])]})
    seal.verify(manifest["seal"])
    run.live.host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
        "arrays_sha256": sha(run.ARRAYS), "script_sha256": sha(Path(__file__)), "reports": reports})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
