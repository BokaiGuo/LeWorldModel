"""Isolated v22/CEM plan and physical execution from continuous warmup."""
import argparse
import importlib.util
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_continuous_warmup_score_dev_20261105_codex as src

ROOT, sha = src.ROOT, src.sha
OUT = ROOT / "results/pc_pusht_continuous_unprivileged_plan_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_CONTINUOUS_UNPRIVILEGED_PLAN_DEV_PROTOCOL_20261105_CODEX.md"
WORKER = ROOT / "experiments/pc_jepa_v22_contact_feedback_worker_20261104_codex.py"


def save(path, payload):
    with path.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    src.verify()
    OUT.mkdir()
    record = {"status": "running", "bindings": {str(p): sha(p) for p in
              (Path(__file__), PROTOCOL, WORKER, Path(src.__file__), src.OUT / "physical.json")}, "runs": []}
    try:
        for seq in (9, 11):
            source = src.OUT / f"root_{seq}/input"
            for arm in ("response", "cem"):
                path = OUT / f"root_{seq}_{arm}"
                (path / "input").mkdir(parents=True)
                (path / "output").mkdir()
                for name in ("context.npz", "model_metadata.json"):
                    shutil.copyfile(source / name, path / "input" / name)
                with np.load(path / "input/context.npz", allow_pickle=False) as f:
                    expected = {"rgb_history", "goal_rgb", "normalized_proprio", "normalized_history_packets",
                                "action_mean", "action_std", "carry"}
                    if set(f.files) != expected or f["carry"].shape != (0, 2):
                        raise ValueError("unexpected context or carry")
                save(path / "input/config.json", {"arm": arm, "step": 0})
                record["runs"].append({"sequence": seq, "arm": arm,
                                       "inputs": {p.name: sha(p) for p in sorted((path / "input").iterdir())}})
        record["status"] = "completed"
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
    save(OUT / "prepared.json", record)
    if record["status"] != "completed":
        raise RuntimeError(record["error"])


def verify_prepared():
    src.verify()
    r = json.loads((OUT / "prepared.json").read_text())
    if r["status"] != "completed" or [(x["sequence"], x["arm"]) for x in r["runs"]] != [
            (9, "response"), (9, "cem"), (11, "response"), (11, "cem")]:
        raise ValueError("prepared run roster")
    if any(sha(Path(p)) != h for p, h in r["bindings"].items()):
        raise ValueError("dependency changed")
    for row in r["runs"]:
        path = OUT / f"root_{row['sequence']}_{row['arm']}" / "input"
        if set(p.name for p in path.iterdir()) != set(row["inputs"]) or any(
                sha(path / name) != h for name, h in row["inputs"].items()):
            raise ValueError("planner input changed")
    return r


def check_decision(seq, arm):
    path = OUT / f"root_{seq}_{arm}"
    result = json.loads((path / "output/decision.json").read_text())
    if result["arm"] != arm or result["step"] != 0 or result["worker_sha256"] != sha(WORKER):
        raise ValueError("worker identity")
    for key, name in (("input_sha256", "context.npz"), ("config_sha256", "config.json"),
                      ("metadata_sha256", "model_metadata.json")):
        if result[key] != sha(path / "input" / name):
            raise ValueError("worker input mismatch")
    chain = np.asarray(result["decision"]["selected"]["chain"], dtype=np.float32)
    if chain.shape != (15, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
        raise ValueError("invalid selected chain")
    tokens = sum(call["prediction_tokens"] for call in result["calls"])
    if tokens != result["resource"]["prediction_tokens"] or tokens > 1920:
        raise ValueError("inconsistent token count")
    # The selected final chain must appear in a full-horizon scored trace.
    selected_score = float(result["decision"]["selected"]["terminal_score"])
    matches = []
    for trace in result["evaluations"]:
        actions = np.asarray(trace["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[1:] == (15, 2):
            for i in np.flatnonzero(np.all(actions == chain, axis=(1, 2))):
                matches.append(float(trace["scores"][int(i)]))
    if not matches or not any(np.isclose(score, selected_score, rtol=1e-5, atol=1e-7) for score in matches):
        raise ValueError("selected chain not bound to scored full-horizon trace")
    return result, chain


def plan():
    verify_prepared()
    for seq in (9, 11):
        for arm in ("response", "cem"):
            path = OUT / f"root_{seq}_{arm}"
            if not (path / "output/decision.json").exists():
                src.visible.core.old.neural.launch(path, "experiments.pc_jepa_v22_contact_feedback_worker_20261104_codex")
            check_decision(seq, arm)
            print(f"sequence {seq} {arm}: sealed decision verified", flush=True)
    verify_prepared()
    save(OUT / "plan_seal.json", {"status": "completed", "prepared_sha256": sha(OUT / "prepared.json"),
                                 "decisions": {f"{seq}_{arm}": sha(OUT / f"root_{seq}_{arm}/output/decision.json")
                                               for seq in (9, 11) for arm in ("response", "cem")}})


def execute():
    verify_prepared()
    seal = json.loads((OUT / "plan_seal.json").read_text())
    if seal["status"] != "completed" or seal["prepared_sha256"] != sha(OUT / "prepared.json"):
        raise ValueError("invalid plan seal")
    for seq in (9, 11):
        for arm in ("response", "cem"):
            if sha(OUT / f"root_{seq}_{arm}/output/decision.json") != seal["decisions"][f"{seq}_{arm}"]:
                raise ValueError("decision changed")
            check_decision(seq, arm)
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact physics overlay required")
    import gym
    import pymunk
    import skimage
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))):
        raise ValueError("physics dependency mismatch")
    base = src.visible.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("unprivileged_execution_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                          for name in ("states.pth", "velocities.pth")]
    rows = []
    for seq in (9, 11):
        source = src.OUT / f"root_{seq}"
        with np.load(source / "physical.npz", allow_pickle=False) as f:
            warm, target, history = (f[k].copy() for k in ("warm", "goal_state", "history_states"))
        with np.load(source / "input/context.npz", allow_pickle=False) as f:
            rgb_history = f["rgb_history"].copy()
        initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
        for arm in ("response", "cem"):
            path = OUT / f"root_{seq}_{arm}"
            if (path / "physical.npz").exists():
                raise FileExistsError(path / "physical.npz")
            decision, chain = check_decision(seq, arm)
            env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
            try:
                env.seed(0)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], rgb_history[0])
                for t, action in enumerate(warm):
                    obs, _, done, info = env.step(action.astype(np.float64))
                    if done:
                        raise ValueError("early warmup done")
                    if t in (4, 9):
                        h = 1 if t == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], rgb_history[h])
                trajectory, images = [info["state"].copy()], [obs["visual"].copy()]
                for action in chain:
                    obs, _, done, info = env.step(action.astype(np.float64))
                    if done:
                        raise ValueError("early execution done")
                    trajectory.append(info["state"].copy())
                    images.append(obs["visual"].copy())
            finally:
                env.close()
            trajectory = np.stack(trajectory)
            with (path / "physical.npz").open("xb") as f:
                np.savez_compressed(f, states=trajectory, images=np.stack(images), actions=chain, target=target)
            end = trajectory[-1]
            joint = float(np.linalg.norm(end[:4] - target[:4]))
            obj = float(np.linalg.norm(end[2:4] - target[2:4]))
            angle = float(abs((end[4] - target[4] + np.pi) % (2*np.pi) - np.pi))
            rows.append({"sequence": seq, "arm": arm, "success": bool(joint < 20 and angle < np.pi/9),
                         "joint_error": joint, "object_error": obj, "angle_error": angle,
                         "center_outside_frames": int(np.sum(np.any((trajectory[:, :4] < 40) | (trajectory[:, :4] > 472), axis=1))),
                         "predicted_score": decision["decision"]["selected"]["terminal_score"],
                         "prediction_tokens": decision["resource"]["prediction_tokens"],
                         "encoding_seconds": decision["encoding_seconds"], "planning_seconds": decision["planning_seconds"],
                         "physical_sha256": sha(path / "physical.npz"), "decision_sha256": sha(path / "output/decision.json")})
            print(f"sequence {seq} {arm}: physical execution complete", flush=True)
    save(OUT / "report.json", {"status": "completed", "plan_seal_sha256": sha(OUT / "plan_seal.json"),
                               "env_sha256": sha(ROOT / runtime.ENV_REL), "rows": rows,
                               "scope": "two exposed development tasks, single open-loop decision"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "plan", "execute"))
    phase = parser.parse_args().phase
    print(f"Nonprivileged continuous-warmup planning: {phase}", flush=True)
    {"prepare": prepare, "plan": plan, "execute": execute}[phase]()
