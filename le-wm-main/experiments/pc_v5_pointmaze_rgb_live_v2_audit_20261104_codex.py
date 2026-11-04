"""Audit V5 RGB records and replay the physical actions with the native backend."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments import pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex as configured
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

PREFIX, run = configured.PREFIX, configured.live
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")


def main():
    configured.configure()
    if AUDIT.exists() or PHYSICAL.exists():
        raise FileExistsError("prior audit or physical replay")
    sha = run.old.prior.sha
    manifest = json.loads(run.MANIFEST.read_text())
    if manifest["dependencies"] != run.dependencies() or manifest["arms"] != ["response"]:
        raise ValueError("dependency or arm mismatch")
    decisions, images, receipt_path = run.paths("response")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["images_sha256"] != sha(images) or receipt["manifest_sha256"] != sha(run.MANIFEST)
            or len(receipt["rows"]) != 3):
        raise ValueError("receipt mismatch")
    steps, actions, carry = [], run.old.WARMUP.tolist(), None
    with np.load(images, allow_pickle=False) as archive:
        frames = archive["rgb"]
        if len(frames) != 31:
            raise ValueError("frame count")
        for step, (path, item) in enumerate(zip(decisions, receipt["rows"], strict=True)):
            data = json.loads(path.read_text())
            binding, result, trace = data["binding"], data["result"], data["trace"]
            ticks = list(run.bridge.runner.previous.prior.context_ticks(16 + 5*step))
            if (data["manifest_sha256"] != sha(run.MANIFEST) or item["decision_sha256"] != sha(path)
                    or data["context_id"] != run.bridge.digest_context(binding) or binding["ticks"] != ticks
                    or binding["rgb_sha256"] != [run.old.digest_image(frames[t]) for t in ticks]
                    or binding["previous_rgb_sha256"] != [run.old.digest_image(frames[t-1]) for t in ticks]
                    or binding["history_actions"] != actions[-10:] or binding["carry"] != carry):
                raise ValueError("context, history or carry mismatch")
            chain = np.asarray(result["selected"]["chain"])
            if (chain.shape != (15-5*step, 2) or np.any(np.abs(chain)>1)
                    or chain[:5].astype(np.float32).tolist() != data["dispatch_float32"]
                    or data["dispatch_float32"] != item["actions"]
                    or item["sealed_ns"] > item["times_ns"][0][0]
                    or len(item["times_ns"]) != 5
                    or item["rgb_sha256"] != [run.old.digest_image(x) for x in frames[16+5*step:21+5*step]]):
                raise ValueError("dispatch, bounds or clock mismatch")
            if step and (trace[0]["requested"] != [carry] or result["carry_score"] != trace[0]["scores"][0]):
                raise ValueError("carry rescore mismatch")
            values, n, tokens, found = [], 0, 0, False
            for batch in trace:
                requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
                if not np.array_equal(requested.astype(np.float32), raw):
                    raise ValueError("conversion mismatch")
                cost = len(raw)*raw.shape[1]//5
                if cost != batch["tokens"] or len(raw) != len(batch["scores"]):
                    raise ValueError("batch size mismatch")
                tokens, n = tokens + cost, n + len(raw)
                if raw.shape[1] == len(chain):
                    values.extend(batch["scores"])
                    found |= any(batch["scores"][j] == result["selected"]["terminal_score"]
                                 for j in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
            counts = data["counts"]
            if (not found or min(values) != result["selected"]["terminal_score"] or tokens > 1920
                    or tokens != counts["prediction_token_samples"] or n != counts["trajectory_samples"]
                    or len(trace) != counts["prediction_calls"]):
                raise ValueError("selection or accounting mismatch")
            steps.append({"step": step, "tokens": tokens, "samples": n, "batches": len(trace),
                          "selected_score": result["selected"]["terminal_score"],
                          "abstained": result["abstained"]})
            actions.extend(item["actions"])
            carry = chain[5:].tolist() if step < 2 else None
    if actions[15:] != receipt["executed_actions"]:
        raise ValueError("action receipt mismatch")
    run.old.write_json(AUDIT, {"manifest_sha256": sha(run.MANIFEST), "execution_sha256": sha(receipt_path),
                              "images_sha256": sha(images), "steps": steps,
                              "limitations": ["model predictions not independently recomputed",
                                              "listed dependencies only"]})
    with tempfile.TemporaryDirectory(prefix="v5_pointmaze_native_replay_") as directory:
        temp = Path(directory)
        request, xml, output = temp / "request.json", temp / "scene.xml", temp / "out.json"
        request.write_text(json.dumps({"rows": [{"start": list(run.old.STARTS[0]), "policy": "response",
                                                "actions": actions}]}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(request), str(xml), str(output)], env=env, check=True, timeout=180)
        replay = json.loads(output.read_text())
    if replay["binding"] != "2.1.2.14" or len(replay["rows"]) != 1:
        raise ValueError("native backend mismatch")
    row = replay["rows"][0]
    if row["policy"] != "response" or len(row["trace"]) != 31:
        raise ValueError("native clock mismatch")
    distances = np.linalg.norm(np.asarray([entry["state"][:2] for entry in row["trace"]])-[1.8, 2.8], axis=1)
    run.old.write_json(PHYSICAL, {"audit_sha256": sha(AUDIT), "trace": row["trace"],
        "distances": [float(distances[i]) for i in (15, 20, 25, 30)], "backend": replay["binding"],
        "worker_sha256": sha(backend.WORKER), "script_sha256": sha(Path(__file__)),
        "limitations": ["replay not matched against every original state or frame", "one exposed root"]})
    print("Audited 3 V5 decisions and replayed 30 native physical actions.", flush=True)


if __name__ == "__main__":
    main()
