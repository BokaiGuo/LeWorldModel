"""Audit PointMaze CEM records, pairing and native physical replay."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments import pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex as configured
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

ROOT, run = configured.ROOT, configured.live
PREFIX = configured.PREFIX
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")


def main():
    configured.configure()
    if AUDIT.exists() or PHYSICAL.exists():
        raise FileExistsError("existing or partial audit")
    sha = run.old.prior.sha
    manifest = json.loads(run.MANIFEST.read_text())
    if (manifest["dependencies"] != run.dependencies() or manifest["arms"] != ["response"]
            or manifest["method"] != configured.METHOD):
        raise ValueError("dependency, method or arm mismatch")
    decisions, images, receipt_path = run.paths("response")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["images_sha256"] != sha(images) or receipt["manifest_sha256"] != sha(run.MANIFEST)
            or len(receipt["rows"]) != 3):
        raise ValueError("receipt mismatch")
    v7prefix = configured.v7.PREFIX
    v7manifest = v7prefix.with_name(v7prefix.name + "_manifest.json")
    v7audit = v7prefix.with_name(v7prefix.name + "_audit.json")
    if json.loads(v7audit.read_text())["manifest_sha256"] != sha(v7manifest):
        raise ValueError("v7 audit mismatch")
    v7images = v7prefix.with_name(v7prefix.name + "_response_images.npz")
    v7dec0 = json.loads(v7prefix.with_name(v7prefix.name + "_response_decision0.json").read_text())
    actions, carry, steps = run.old.WARMUP.tolist(), None, []
    with np.load(images, allow_pickle=False) as archive, np.load(v7images, allow_pickle=False) as original:
        frames = archive["rgb"]
        if len(frames) != 31:
            raise ValueError("frame count")
        np.testing.assert_array_equal(frames[:16], original["rgb"][:16])
        for step, (path, item) in enumerate(zip(decisions, receipt["rows"], strict=True)):
            data = json.loads(path.read_text())
            binding, result, trace = data["binding"], data["result"], data["trace"]
            ticks = list(run.bridge.runner.previous.prior.context_ticks(16+5*step))
            if (data["manifest_sha256"] != sha(run.MANIFEST) or item["decision_sha256"] != sha(path)
                    or data["context_id"] != run.bridge.digest_context(binding) or binding["ticks"] != ticks
                    or binding["rgb_sha256"] != [run.old.digest_image(frames[t]) for t in ticks]
                    or binding["previous_rgb_sha256"] != [run.old.digest_image(frames[t-1]) for t in ticks]
                    or binding["history_actions"] != actions[-10:] or binding["carry"] != carry):
                raise ValueError("context, history or carry mismatch")
            if step == 0:
                for key in ("ticks", "rgb_sha256", "previous_rgb_sha256", "history_actions", "goal_sha256",
                            "goal_position", "model_hash", "checkpoint_sha256", "action_mean", "action_std"):
                    if binding[key] != v7dec0["binding"][key]:
                        raise ValueError(f"initial context mismatch: {key}")
            chain = np.asarray(result["selected"]["chain"])
            if (chain.shape != (15-5*step, 2) or np.any(np.abs(chain) > 1)
                    or chain[:5].astype(np.float32).tolist() != data["dispatch_float32"]
                    or data["dispatch_float32"] != item["actions"]
                    or item["sealed_ns"] > item["times_ns"][0][0] or len(item["times_ns"]) != 5
                    or item["rgb_sha256"] != [run.old.digest_image(x) for x in frames[16+5*step:21+5*step]]):
                raise ValueError("dispatch or clock mismatch")
            if step:
                seeds, scores = trace[0]["raw"], trace[0]["scores"]
                np.testing.assert_array_equal(np.asarray(seeds[0], np.float32), np.zeros_like(np.asarray(carry)))
                np.testing.assert_array_equal(np.asarray(seeds[1], np.float32), np.asarray(carry, np.float32))
                if result["carry"] != carry or scores != [result["zero_score"], result["carry_score"]]:
                    raise ValueError("CEM seed binding mismatch")
                restarts = result["restarts"]
                counts = [x["samples"] for x in restarts]
                if ([x["name"] for x in restarts] != ["zero", "carry"] or abs(counts[0]-counts[1]) > 1
                        or sum(counts)+2 != 1920//(3-step)
                        or any(sum(g["size"] for g in r["generations"]) != r["samples"] for r in restarts)):
                    raise ValueError("restart allocation mismatch")
            values, samples, tokens, found = [], 0, 0, False
            for batch in trace:
                requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
                if (not np.array_equal(requested.astype(np.float32), raw) or not np.isfinite(raw).all()
                        or np.any(np.abs(raw) > 1) or not np.isfinite(batch["scores"]).all()
                        or len(raw) != len(batch["scores"]) or len(raw) != len(batch["positions"])
                        or batch["tokens"] != len(raw)*raw.shape[1]//5):
                    raise ValueError("batch mismatch")
                tokens += batch["tokens"]
                samples += len(raw)
                if raw.shape[1] == len(chain):
                    values.extend(batch["scores"])
                    found |= any(batch["scores"][j] == result["selected"]["terminal_score"]
                                 for j in np.flatnonzero(np.all(requested == chain, axis=(1, 2))))
            if (not found or min(values) != result["selected"]["terminal_score"] or tokens != 1920
                    or tokens != data["counts"]["prediction_token_samples"]
                    or samples != data["counts"]["trajectory_samples"]
                    or len(trace) != data["counts"]["prediction_calls"]):
                raise ValueError("selection or accounting mismatch")
            steps.append({"step": step, "tokens": tokens, "samples": samples,
                          "batches": len(trace), "selected_score": result["selected"]["terminal_score"]})
            actions.extend(item["actions"])
            carry = chain[5:].tolist() if step < 2 else None
    if actions[15:] != receipt["executed_actions"]:
        raise ValueError("action receipt mismatch")
    run.old.write_json(AUDIT, {"manifest_sha256": sha(run.MANIFEST), "execution_sha256": sha(receipt_path),
        "images_sha256": sha(images), "v7_audit_sha256": sha(v7audit), "initial_context_matches": True,
        "own_carry_and_restart_verified": True, "steps": steps,
        "scope": "record and shared initial context; prediction and native replay separate"})
    with tempfile.TemporaryDirectory(prefix="v7_pointmaze_cem_native_") as directory:
        tmp = Path(directory)
        request, xml, output = tmp / "request.json", tmp / "scene.xml", tmp / "out.json"
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
    distances = np.linalg.norm(np.asarray([x["state"][:2] for x in row["trace"]])-[1.8, 2.8], axis=1)
    run.old.write_json(PHYSICAL, {"audit_sha256": sha(AUDIT), "trace": row["trace"],
        "distances": [float(distances[i]) for i in (15, 20, 25, 30)], "backend": replay["binding"],
        "worker_sha256": sha(backend.WORKER), "script_sha256": sha(Path(__file__)),
        "limitations": ["native replay not matched against every original state or RGB", "one exposed root"]})
    print("Audited paired initial context, CEM records and 30 native actions.", flush=True)


if __name__ == "__main__":
    main()
