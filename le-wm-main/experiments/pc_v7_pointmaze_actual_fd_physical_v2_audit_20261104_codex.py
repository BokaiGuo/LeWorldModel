"""Independently recompute saved probe metrics and verify recorded lineage."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_pointmaze_actual_fd_physical_v2_dev_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    run.configured.configure()
    live = run.configured.live
    manifest, result = (json.loads(path.read_text()) for path in (run.MANIFEST, run.RESULT))
    seal.verify(manifest["seal"])
    sha = seal.digest
    if result["manifest_sha256"] != sha(run.MANIFEST) or result["arrays_sha256"] != sha(run.ARRAYS):
        raise ValueError("result hash mismatch")
    decisions, images, execution = live.paths("response")
    decision = json.loads(decisions[2].read_text())
    receipt = json.loads(execution.read_text())
    camera = json.loads(live.h.CAMERA.read_text())["metadata"]
    offset = json.loads(live.h.OLD.read_text())["body_offset_from_static_scene"]
    with np.load(run.ARRAYS, allow_pickle=False) as data:
        actions, frames, estimated, states, prefix = (data[k] for k in
            ("actions", "frames", "estimated", "native_states", "prefix"))
        if frames.shape != (2, 21, 5, 224, 224, 3):
            raise ValueError("wrong frame count")
        np.testing.assert_array_equal(prefix, np.asarray(live.old.WARMUP.tolist()
                                                        + receipt["executed_actions"][:10]))
        estimates = np.asarray([live.h.causal_state(b[-2], b[-1], camera, offset, .05)
                                for b in frames.reshape(42, 5, 224, 224, 3)]).reshape(2, 21, 4)
        np.testing.assert_array_equal(estimates, estimated)
        for i, info in enumerate(decision["result"]["first_response"]):
            nominal = np.asarray(info["nominal"])
            axes = np.eye(10).reshape(10, 5, 2)
            requested = np.concatenate((nominal[None], nominal[None]+.05*axes, nominal[None]-.05*axes))
            batches = [b for b in decision["trace"] if np.array_equal(np.asarray(b["requested"]), requested)]
            if len(batches) != 1:
                raise ValueError("ambiguous original probe")
            batch = batches[0]
            np.testing.assert_array_equal(actions[i], np.asarray(batch["raw"], np.float32))
            for j in range(21):
                row = result["native_rows"][i*21+j]
                np.testing.assert_array_equal(states[i, j], np.asarray([t["state"] for t in row["trace"]]))
            def matrix(values):
                return ((values[1:11, :2] - values[11:21, :2]) / .1).T
            model = matrix(np.asarray(batch["positions"]))
            rgb = matrix(estimated[i])
            physical = matrix(states[i, :, -1])
            row = result["rows"][i]
            for name, value in (("model_matrix", model), ("rgb_matrix", rgb), ("native_matrix", physical)):
                np.testing.assert_allclose(row[name], value, atol=1e-12, rtol=0)
            for name, left, right in (("model_vs_rgb", model, rgb), ("model_vs_native", model, physical),
                                      ("rgb_vs_native", rgb, physical)):
                actual = run.compare(left, right)
                for key, val in actual.items():
                    if val is None:
                        if row[name][key] is not None:
                            raise ValueError("undefined metric changed")
                    else:
                        np.testing.assert_allclose(row[name][key], val, atol=1e-12, rtol=0)
            scores = np.asarray(batch["scores"])
            native_dist = np.linalg.norm(states[i, :, -1, :2] - [1.8, 2.8], axis=1)
            rgb_dist = np.linalg.norm(estimated[i, :, :2] - [1.8, 2.8], axis=1)
            for name, distance in (("native", native_dist), ("rgb", rgb_dist)):
                np.testing.assert_allclose(row[name+"_distances"], distance, atol=1e-12, rtol=0)
                if row["model_vs_"+name+"_ordering"] != run.ordering(scores, distance):
                    raise ValueError("ordering mismatch")
    live.old.write_json(OUT, {"result_sha256": sha(run.RESULT), "arrays_sha256": sha(run.ARRAYS),
                              "audit_script_sha256": sha(Path(__file__)),
                              "checked": ["dependency seal", "42 original action packets", "saved RGB causal estimates",
                                          "native saved states", "matrices", "distances", "pairwise ordering"],
                              "limits": "does not independently rerender branches or replay native worker"})
    print("Audited sealed probes, saved frames, states, and metrics.", flush=True)


if __name__ == "__main__":
    main()
