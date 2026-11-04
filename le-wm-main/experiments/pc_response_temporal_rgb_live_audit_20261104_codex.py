"""Verify saved RGB feedback decisions; separately replay native physics."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_rgb_live_diag_20261104_codex as run
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
PHYSICAL = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    if AUDIT.exists() or PHYSICAL.exists():
        raise FileExistsError("audit or replay already exists")
    sha = run.old.prior.sha
    manifest = json.loads(run.MANIFEST.read_text())
    require(manifest["dependencies"] == run.dependencies(), "dependency mismatch")
    summaries, requests = [], []
    for arm in run.ARMS:
        decisions, images, receipt_path = run.paths(arm)
        receipt = json.loads(receipt_path.read_text())
        require(receipt["images_sha256"] == sha(images) and receipt["manifest_sha256"] == sha(run.MANIFEST)
                and receipt["arm"] == arm and len(receipt["rows"]) == 3, "receipt binding")
        with np.load(images, allow_pickle=False) as archive:
            frames = archive["rgb"]
            require(len(frames) == 31, "frame count")
            actions = run.old.WARMUP.tolist()
            carry, steps = None, []
            for step, (path, item) in enumerate(zip(decisions, receipt["rows"], strict=True)):
                data = json.loads(path.read_text())
                binding, result, trace = data["binding"], data["result"], data["trace"]
                require(data["manifest_sha256"] == sha(run.MANIFEST) and item["decision_sha256"] == sha(path)
                        and data["context_id"] == run.bridge.digest_context(binding), "decision binding")
                ticks = binding["ticks"]
                require(ticks == list(run.bridge.runner.previous.prior.context_ticks(16 + 5*step))
                        and binding["rgb_sha256"] == [run.old.digest_image(frames[t]) for t in ticks]
                        and binding["previous_rgb_sha256"] == [run.old.digest_image(frames[t-1]) for t in ticks]
                        and binding["history_actions"] == actions[-10:] and binding["carry"] == carry,
                        "context mismatch")
                chain = np.asarray(result["selected"]["chain"])
                require(chain.shape == (15 - 5*step, 2)
                        and item["actions"] == chain[:5].astype(np.float32).tolist()
                        and data["dispatch_float32"] == item["actions"], "dispatch mismatch")
                require(item["sealed_ns"] <= item["times_ns"][0][0] and len(item["times_ns"]) == 5
                        and item["rgb_sha256"] == [run.old.digest_image(x) for x in
                                                    frames[16 + 5*step:21 + 5*step]], "execution clock")
                total, samples, values, found = 0, 0, [], False
                for batch in trace:
                    proposed = np.asarray(batch["requested"])
                    raw = np.asarray(batch["raw"], dtype=np.float32)
                    require(np.array_equal(proposed.astype(np.float32), raw), "float32 mismatch")
                    tokens = len(raw) * raw.shape[1] // 5
                    require(tokens == batch["tokens"] and len(batch["scores"]) == len(raw)
                            and len(batch["positions"]) == len(raw), "batch count mismatch")
                    total += tokens
                    samples += len(raw)
                    if raw.shape[1] == len(chain):
                        values.extend(batch["scores"])
                        for index in np.flatnonzero(np.all(proposed == chain, axis=(1, 2))):
                            found |= batch["scores"][index] == result["selected"]["terminal_score"]
                counts = data["counts"]
                require(found and total <= run.method.CAP and total == counts["prediction_token_samples"]
                        and samples == counts["trajectory_samples"] and len(trace) == counts["prediction_calls"],
                        "selected or accounting mismatch")
                if step:
                    require(result["carry"] == carry and trace[0]["requested"] == [carry]
                            and result["carry_score"] == trace[0]["scores"][0], "carry mismatch")
                minimum = min(values)
                steps.append({"step": step, "tokens": total, "prediction_calls": len(trace),
                              "selected_score": result["selected"]["terminal_score"],
                              "minimum_scored": minimum,
                              "selected_global_scored_min": abs(minimum - result["selected"]["terminal_score"]) <= 1e-12,
                              "abstained": result["abstained"], "planning_seconds": data["planning_seconds"],
                              "selected_name": result["selected"]["name"]})
                actions.extend(item["actions"])
                carry = chain[5:].tolist() if step < 2 else None
            require(actions[15:] == receipt["executed_actions"], "full action mismatch")
        summaries.append({"root": 0, "arm": arm, "steps": steps, "execution_sha256": sha(receipt_path),
                          "images_sha256": sha(images), "decisions_sha256": [sha(p) for p in decisions]})
        requests.append({"start": list(run.old.STARTS[0]), "policy": arm, "actions": actions})
    run.old.write_json(AUDIT, {"manifest_sha256": sha(run.MANIFEST), "rows": summaries,
                              "all_scored_min": all(s["selected_global_scored_min"]
                                                    for row in summaries for s in row["steps"]),
                              "limitations": ["does not independently recompute model predictions",
                                              "hashes only listed dependencies"]})
    with tempfile.TemporaryDirectory(prefix="pc_temporal_rgb_replay_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "request.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        require(sha(xml) == "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d",
                "scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        physical = json.loads(out.read_text())
    require(physical["binding"] == "2.1.2.14" and len(physical["rows"]) == len(summaries), "backend mismatch")
    rows = []
    for summary, replay in zip(summaries, physical["rows"], strict=True):
        require(replay["policy"] == summary["arm"] and len(replay["trace"]) == 31, "replay clock")
        distances = np.linalg.norm(np.asarray([x["state"][:2] for x in replay["trace"]]) - [1.8, 2.8], axis=1)
        rows.append({"arm": summary["arm"], "distances": [float(distances[t]) for t in (15, 20, 25, 30)],
                     "trace": replay["trace"]})
    run.old.write_json(PHYSICAL, {"audit_sha256": sha(AUDIT), "rows": rows, "backend": physical["binding"],
                                  "worker_sha256": sha(backend.WORKER), "script_sha256": sha(Path(__file__)),
                                  "limitations": ["physical replay is not independently matched frame-by-frame",
                                                  "one exposed root; not performance confirmation"]})
    print("RGB decisions audited and two native trajectories replayed.", flush=True)


if __name__ == "__main__":
    main()
