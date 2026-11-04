"""Constructed-feasible native PushT feedback interface diagnostic."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_jepa_native_v12_feedback_dev_20261104_codex"
OLD = ROOT / "results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1"
PREV = ROOT / "results/pc_jepa_native_boundary_replay_20261104_codex"
META = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex/input/model_metadata.json"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_NATIVE_V12_FEASIBLE_FEEDBACK_DEV_PROTOCOL_20261104_CODEX.md"
LAUNCH = "experiments.pc_jepa_native_v12_feedback_launch_20261104_codex"
SEED = 2026110412
WITNESS_SEED = 2026110413


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def fixed_goal(seed):
    rng = np.random.RandomState(seed)
    def generate():
        return np.array([rng.randint(50, 450), rng.randint(50, 450),
                         rng.randint(100, 400), rng.randint(100, 400),
                         rng.randn() * 2 * np.pi - np.pi, 0, 0], dtype=np.float64)
    generate()
    return generate()


def metrics(state, goal):
    state, goal = np.asarray(state, dtype=float), np.asarray(goal, dtype=float)
    angle = abs((state[4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
    joint = float(np.linalg.norm(state[:4] - goal[:4]))
    return {"agent_error": float(np.linalg.norm(state[:2] - goal[:2])),
            "object_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "angle_error_radians": float(angle), "joint_xy_error": joint,
            "success": bool(joint < 20 and angle < np.pi / 9)}


def main():
    print("Native JEPA v12/CEM: constructed-feasible interface diagnostic.", flush=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact physics-only overlay required")
    import gym
    import pymunk
    import skimage
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("native runtime mismatch")
    if any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage)):
        raise ValueError("native module origin mismatch")
    base = prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")
    if probe["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime probe failure")
    replay = json.loads((PREV / "result.json").read_text())
    audit = json.loads((PREV / "audit.json").read_text())
    if replay["status"] != "exact_boundary_replay" or audit["status"] != "saved_replay_matches_prior_journal":
        raise ValueError("physical replay not verified")
    if audit["new_result_sha256"] != sha(PREV / "result.json"):
        raise ValueError("physical replay audit mismatch")
    stats = prior.native_stats(ROOT)
    warm = np.asarray(prior.SCHEDULE, dtype=np.float64).reshape(15, 2)[:10]
    spec = importlib.util.spec_from_file_location("pc_jepa_feedback_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    cls = native.PushTEnv
    base.mkdir(OUT)
    manifest = {"protocol_sha256": sha(PROTOCOL), "collector_sha256": sha(Path(__file__)),
        "worker_sha256": sha(ROOT / "experiments/pc_jepa_native_v12_feedback_worker_20261104_codex.py"),
        "launcher_sha256": sha(ROOT / "experiments/pc_jepa_native_v12_feedback_launch_20261104_codex.py"),
        "v12_sha256": sha(ROOT / "experiments/spectral_reachability/pc_recursive_response_v12.py"),
        "v6_sha256": sha(ROOT / "experiments/spectral_reachability/pc_target_reach_action_dim_v6.py"),
        "metadata_sha256": sha(META), "native_sha256": sha(ROOT / runtime.ENV_REL),
        "boundary_result_sha256": sha(PREV / "result.json"), "boundary_audit_sha256": sha(PREV / "audit.json"),
        "seed": SEED, "witness_seed": WITNESS_SEED,
        "warmup": warm.tolist(), "arms": ["response", "cem"], "steps": [0, 1, 2],
        "action_stats": {k: v.tolist() for k, v in stats.items()}, "cap": 1920}
    base.save_json(OUT / "manifest.json", manifest)
    base.save_json(OUT / "runtime_probe.json", probe)
    journal = base.Journal(OUT / "journal.jsonl")
    calls = base.NativeCalls(cls, pymunk)
    calls.start()
    goal_env = None
    try:
        witness = np.random.default_rng(WITNESS_SEED).uniform(-.45, .45, (15, 2)).astype(np.float32)
        base.save_array(OUT / "witness_actions_evaluation_only.npy", witness)
        witness_runs = []
        for name in ("witness", "witness_replay"):
            goal_env = base.operation(journal, calls, name, "construct", {"config": base.CTOR}, lambda: cls(**base.CTOR))
            base.operation(journal, calls, name, "seed", {"seed": SEED}, lambda: goal_env.seed(SEED))
            obs, state = base.operation(journal, calls, name, "reset", {}, goal_env.reset)
            boundaries = [(obs["visual"].copy(), obs["proprio"].copy(), state.copy())]
            for index, action in enumerate(np.concatenate((warm, witness))):
                obs, _, done, info = base.operation(journal, calls, name, "step",
                    {"frame": index + 1, "action": action.tolist()}, lambda: goal_env.step(action.copy()))
                if done:
                    raise ValueError("witness terminated early")
                state = info["state"]
                boundaries.append((obs["visual"].copy(), obs["proprio"].copy(), state.copy()))
            base.operation(journal, calls, name, "close", {}, goal_env.close)
            goal_env = None
            witness_runs.append(boundaries)
        for first, replay_boundary in zip(*witness_runs, strict=True):
            for a, b in zip(first, replay_boundary, strict=True):
                np.testing.assert_array_equal(a, b)
        goal_rgb, _, goal_state = witness_runs[0][-1]
        if not metrics(goal_state, goal_state)["success"]:
            raise ValueError("witness terminal not successful")
        for key, index in (("rgb", 0), ("proprio", 1), ("state", 2)):
            base.save_array(OUT / f"witness_{key}_evaluation_only.npy",
                            np.stack([boundary[index] for boundary in witness_runs[0]]))
        base.save_array(OUT / "goal_rgb.npy", goal_rgb)
        base.save_array(OUT / "goal_state_evaluation_only.npy", goal_state)
        results = {}
        for arm in ("response", "cem"):
            env, error, decisions, events = None, None, [], []
            try:
                env = base.operation(journal, calls, arm, "construct", {"config": base.CTOR}, lambda: cls(**base.CTOR))
                base.operation(journal, calls, arm, "seed", {"seed": SEED}, lambda: env.seed(SEED))
                obs, state = base.operation(journal, calls, arm, "reset", {}, env.reset)
                timeline = OUT / (arm + "_physical")
                base.mkdir(timeline)
                images, props, actions = [], [], []
                names = ("rgb.npy", "state.npy", "proprio.npy", "full_public_state.json", "transition.json")

                def capture(frame, obs, state, transition):
                    record = base.capture(env, obs, state, transition, timeline / f"frame_{frame:02d}")
                    if frame <= 10:
                        expected = witness_runs[0][frame]
                        for observed, reference in zip((obs["visual"], obs["proprio"], state), expected, strict=True):
                            np.testing.assert_array_equal(observed, reference)
                    images.append(obs["visual"].copy())
                    props.append(obs["proprio"].copy())
                    events.append({"frame": frame, "record": record, "metrics": metrics(state, goal_state)})
                    journal.append({"phase": "capture", "arm": arm, "frame": frame,
                                    "files": record["files"], "metrics": events[-1]["metrics"]})

                capture(0, obs, state, {"kind": "reset"})
                for index, u in enumerate(warm):
                    details = {"frame": index + 1, "action": u.tolist(), "source": "fixed_warmup"}
                    obs, reward, done, info = base.operation(journal, calls, arm, "step", details, lambda: env.step(u.copy()))
                    actions.append(u.copy())
                    capture(index + 1, obs, info["state"], {"kind": "step", "reward": reward, "done": done, "info": info})
                    if done:
                        raise ValueError("warmup done")
                carry = None
                for step in range(3):
                    current = 10 + 5 * step
                    decision_dir = OUT / f"{arm}_{step}"
                    base.mkdir(decision_dir)
                    base.mkdir(decision_dir / "input")
                    base.mkdir(decision_dir / "output")
                    history_rgb = np.stack([images[current - 10], images[current - 5], images[current]])
                    raw_prop = np.stack([props[current - 10], props[current - 5], props[current]]).astype(np.float32)
                    prop = (raw_prop - stats["PROPRIO_MEAN"]) / stats["PROPRIO_STD"]
                    # Reproduce dataset projection: native argument -> pixel float32 -> /100.
                    history_u = (100 * np.asarray(actions[-10:], dtype=np.float64)).astype(np.float32) / np.float32(100)
                    packets = ((history_u - stats["ACTION_MEAN"]) / stats["ACTION_STD"]).reshape(2, 10)
                    c = np.empty((0, 2), np.float32) if carry is None else np.asarray(carry, np.float32)
                    with (decision_dir / "input/context.npz").open("xb") as f:
                        np.savez_compressed(f, rgb_history=history_rgb, goal_rgb=goal_rgb[None],
                            normalized_proprio=prop, normalized_history_packets=packets,
                            action_mean=stats["ACTION_MEAN"], action_std=stats["ACTION_STD"], carry=c)
                    base.save_json(decision_dir / "input/config.json", {"arm": arm, "step": step})
                    base.save_json(decision_dir / "input/model_metadata.json", json.loads(META.read_text()))
                    journal.append({"phase": "plan_intent", "arm": arm, "step": step,
                                    "input_sha256": sha(decision_dir / "input/context.npz")})
                    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", LAUNCH, str(decision_dir)],
                                   check=True, cwd=ROOT, timeout=600)
                    path = decision_dir / "output/decision.json"
                    result = json.loads(path.read_text())
                    if result["input_sha256"] != sha(decision_dir / "input/context.npz") or result["arm"] != arm or result["step"] != step:
                        raise ValueError("decision binding mismatch")
                    chain = np.asarray(result["decision"]["selected"]["chain"], dtype=np.float32)
                    if chain.shape != (15 - 5 * step, 2) or not np.isfinite(chain).all() or np.any(np.abs(chain) > 1):
                        raise ValueError("selected action invalid")
                    decision_sha = sha(path)
                    journal.append({"phase": "decision_sealed", "arm": arm, "step": step,
                                    "decision_sha256": decision_sha, "chain": chain.tolist()})
                    decisions.append({"step": step, "decision_sha256": decision_sha,
                                      "meter": result["meter"], "selected": result["decision"]["selected"]})
                    for j, u in enumerate(chain[:5]):
                        u = u.astype(np.float64)
                        frame = current + j + 1
                        details = {"frame": frame, "action": u.tolist(), "decision_sha256": decision_sha,
                                   "agent_xy_before": list(env.agent.position),
                                   "absolute_target": (np.asarray(env.agent.position) + 100 * u).tolist()}
                        base.save_json(timeline / f"command_{frame:02d}.json", details)
                        obs, reward, done, info = base.operation(journal, calls, arm, "step", details, lambda: env.step(u.copy()))
                        actions.append(u.copy())
                        capture(frame, obs, info["state"], {"kind": "step", "reward": reward, "done": done, "info": info})
                        if done:
                            raise ValueError("unexpected physical done")
                    carry = chain[5:].copy()
                    print(f"{arm} completed decision {step}; {events[-1]['metrics']}", flush=True)
            except Exception:
                error = traceback.format_exc()
            finally:
                if env is not None:
                    try:
                        base.operation(journal, calls, arm, "close", {}, env.close)
                    except Exception:
                        error = (error or "") + traceback.format_exc()
            results[arm] = {"status": "completed" if error is None and len(decisions) == 3 else "failed",
                            "error": error, "decisions": decisions, "events": events}
            base.save_json(OUT / f"{arm}_result.json", results[arm])
    finally:
        if goal_env is not None:
            goal_env.close()
        calls.stop()
        receipt = calls.receipt()
        journal.close()
        base.save_json(OUT / "native_calls.json", receipt)
    final = {"results": {arm: {"status": r["status"], "error": r["error"],
                                "boundary_metrics": [r["events"][i]["metrics"] for i in (10, 15, 20, 25)
                                                     if len(r["events"]) > i]} for arm, r in results.items()},
             "manifest_sha256": sha(OUT / "manifest.json"), "journal_sha256": sha(OUT / "journal.jsonl"),
             "scope": "one constructed-feasible root, RGB plus proprio; interface diagnostic only", "pc_gate_pass": False}
    base.save_json(OUT / "summary.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
