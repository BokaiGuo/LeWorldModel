"""Audit v7 paired choices and replay PointMaze actions on native MuJoCo."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments import pc_v7_pointmaze_rgb_live_dev_20261104_codex as configured
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

PREFIX, run = configured.PREFIX, configured.live
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")


def main():
    configured.configure()
    if AUDIT.exists() or PHYSICAL.exists():
        raise FileExistsError("existing audit or physical replay")
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
            if (chain.shape != (15-5*step, 2) or np.any(np.abs(chain) > 1)
                    or chain[:5].astype(np.float32).tolist() != data["dispatch_float32"]
                    or data["dispatch_float32"] != item["actions"]
                    or item["sealed_ns"] > item["times_ns"][0][0] or len(item["times_ns"]) != 5
                    or item["rgb_sha256"] != [run.old.digest_image(x) for x in frames[16+5*step:21+5*step]]):
                raise ValueError("dispatch, bounds or clock mismatch")
            if step and (trace[0]["requested"] != [carry] or result["carry_score"] != trace[0]["scores"][0]):
                raise ValueError("carry rescore mismatch")
            tokens = samples = 0
            for batch in trace:
                requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
                if (not np.array_equal(requested.astype(np.float32), raw) or np.any(np.abs(raw) > 1)
                        or not np.isfinite(raw).all() or not np.isfinite(batch["scores"]).all()
                        or not np.isfinite(batch["positions"]).all()
                        or len(raw) != len(batch["scores"]) or len(raw) != len(batch["positions"])
                        or batch["tokens"] != len(raw)*raw.shape[1]//5):
                    raise ValueError("trace mismatch")
                tokens += batch["tokens"]
                samples += len(raw)
            pair, last = result["paired_comparison"], trace[-1]
            requested = np.asarray(last["requested"])
            raw = np.asarray(last["raw"], np.float32)
            scores = last["scores"]
            expected = np.stack((np.asarray(result["precomparison"]["chain"]), np.asarray(result["comparison_chain"])))
            same = np.array_equal(raw[0], raw[1])
            abstain = bool(same or scores[1] <= scores[0])
            if (not np.array_equal(requested, expected) or result["abstained"] != abstain
                    or pair["same_actual_float32_actions"] != same
                    or pair["incumbent_score"] != scores[0] or pair["zero_prefix_score"] != scores[1]
                    or pair["reserved_token_samples"] != 2*(3-step)
                    or not np.array_equal(chain, requested[int(abstain)])
                    or result["selected"]["terminal_score"] != scores[int(abstain)]
                    or tokens > 1920 or tokens != data["counts"]["prediction_token_samples"]
                    or samples != data["counts"]["trajectory_samples"]
                    or len(trace) != data["counts"]["prediction_calls"]):
                raise ValueError("paired choice or budget mismatch")
            generations = result["refinement"]
            boundary = len(trace)-len(generations)-2
            earlier = [(a, s) for b in trace[:boundary] if len(b["requested"][0]) == len(chain)
                       for a, s in zip(b["requested"], b["scores"], strict=True)]
            start = result["refinement_start"]
            if (boundary < 1 or len(earlier) != start["full_samples_seen"]
                    or min(s for _, s in earlier) != start["terminal_score"]
                    or (start["chain"], start["terminal_score"]) not in earlier
                    or any(g["size"] != len(b["scores"])
                           for g, b in zip(generations, trace[boundary:-2], strict=True))):
                raise ValueError("refinement accounting mismatch")
            steps.append({"step": step, "tokens": tokens, "samples": samples, "batches": len(trace),
                          "selected_score": scores[int(abstain)], "abstained": abstain, "same_actions": same})
            actions.extend(item["actions"])
            carry = chain[5:].tolist() if step < 2 else None
    if actions[15:] != receipt["executed_actions"]:
        raise ValueError("action receipt mismatch")
    run.old.write_json(AUDIT, {"manifest_sha256": sha(run.MANIFEST), "execution_sha256": sha(receipt_path),
                               "images_sha256": sha(images), "steps": steps,
                               "scope": "paired decision and causal records; prediction replay separate"})
    with tempfile.TemporaryDirectory(prefix="v7_pointmaze_native_replay_") as directory:
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
    distances = np.linalg.norm(np.asarray([x["state"][:2] for x in row["trace"]])-[1.8, 2.8], axis=1)
    run.old.write_json(PHYSICAL, {"audit_sha256": sha(AUDIT), "trace": row["trace"],
        "distances": [float(distances[i]) for i in (15, 20, 25, 30)], "backend": replay["binding"],
        "worker_sha256": sha(backend.WORKER), "script_sha256": sha(Path(__file__)),
        "limitations": ["native replay not matched against every original state or RGB frame", "one exposed root"]})
    print("Audited v7 paired choices and replayed 30 native actions.", flush=True)


if __name__ == "__main__":
    main()
