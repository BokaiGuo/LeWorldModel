"""Exact native physical warmup replay, without planning or neural calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1"
OUT = ROOT / "results/pc_jepa_native_boundary_replay_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_NATIVE_BOUNDARY_REPLAY_PROTOCOL_20261104_CODEX.md"
EXPECTED = {"env_construct": 1, "seed": 2, "reset": 1, "control_step": 10,
            "space_step": 101, "native_render": 11, "native_get_obs": 11, "close": 1}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def run():
    print("Replaying the fixed native 10-step warmup; no planning or neural execution.", flush=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact isolated physics overlay required")
    import gym
    import pymunk
    import skimage
    if any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage)):
        raise ValueError("native runtime module origin mismatch")
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("native runtime versions differ")
    base = prior.helpers(ROOT)
    runtime, manifest = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT, manifest, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")
    if probe["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("native runtime probe failed")
    old_report = json.loads((OLD / "report.json").read_text())
    if old_report["status"] != "fresh_native_nonzero_factual_recorded" or not old_report["source_hashes_unchanged"]:
        raise ValueError("prior factual run invalid")
    if not all(row["all_exact"] for row in old_report["rows"][:11]):
        raise ValueError("prior paired replay mismatch")
    schedule = np.asarray(prior.SCHEDULE, dtype=np.float64).reshape(15, 2)[:10].copy()
    if not np.array_equal(schedule, np.load(OLD / "a/consumer/raw_env_commands.npy", allow_pickle=False)[:10]):
        raise ValueError("prior action mismatch")
    source = ROOT / runtime.ENV_REL
    spec = importlib.util.spec_from_file_location("pc_jepa_boundary_native", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.PushTEnv
    base.mkdir(OUT)
    bindings = {"protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                "prior_report_sha256": sha(OLD / "report.json"), "native_source_sha256": sha(source),
                "prior_inputs_sha256": sha(OLD / "a/consumer/raw_env_commands.npy"),
                "commands": schedule.tolist(), "seed": prior.SEED, "constructor": base.CTOR}
    base.save_json(OUT / "bindings.json", bindings)
    base.save_json(OUT / "runtime_probe.json", probe)
    journal = base.Journal(OUT / "journal.jsonl")
    calls = base.NativeCalls(cls, pymunk)
    calls.start()
    env, problem, comparisons = None, None, []
    try:
        env = base.operation(journal, calls, "replay", "construct", {"config": base.CTOR}, lambda: cls(**base.CTOR))
        base.operation(journal, calls, "replay", "seed", {"seed": prior.SEED}, lambda: env.seed(prior.SEED))
        obs, state = base.operation(journal, calls, "replay", "reset", {"expected_space_steps": 1}, env.reset)
        for frame in range(11):
            if frame:
                u = schedule[frame - 1]
                command = {"frame": frame, "raw_env_argument": u.tolist(),
                           "agent_xy_before": list(env.agent.position),
                           "absolute_target": (np.array(env.agent.position) + 100 * u).tolist(),
                           "expected_substeps": 10}
                base.save_json(OUT / f"command_{frame:02d}.json", command)
                obs, reward, done, info = base.operation(journal, calls, "replay", "step", command,
                                                        lambda: env.step(u.copy()))
                state = info["state"]
                transition = {"kind": "step", "reward": reward, "done": done, "info": info}
                if done:
                    raise ValueError("unexpected native done")
            else:
                transition = {"kind": "reset"}
            record = base.capture(env, obs, state, transition, OUT / f"frame_{frame:02d}")
            names = ("rgb.npy", "state.npy", "proprio.npy", "full_public_state.json", "transition.json")
            expected = {name: sha(OLD / f"a/frame_{frame:02d}" / name) for name in names}
            equal = {name: record["files"][name]["sha256"] == expected[name] for name in names}
            comparisons.append({"frame": frame, "exact": equal, "expected_sha256": expected,
                                "actual_sha256": {name: record["files"][name]["sha256"] for name in names}})
            journal.append({"phase": "frame_comparison", "frame": frame, "exact": equal})
            if not all(equal.values()):
                raise ValueError(f"frame {frame} differs from prior physical replay")
            print(f"frame {frame}: exact", flush=True)
    except Exception:
        problem = traceback.format_exc()
    finally:
        if env is not None:
            try:
                base.operation(journal, calls, "replay", "close", {}, env.close)
            except Exception:
                problem = (problem or "") + traceback.format_exc()
        calls.stop()
        receipt = calls.receipt()
        journal.close()
    observed = receipt["python_call_entries"]
    count_checks = {key: observed.get(key, 0) == value for key, value in EXPECTED.items()}
    dt_ok = len(receipt["space_step_events"]) == 101 and all(x["dt"] == .01 for x in receipt["space_step_events"])
    status = "exact_boundary_replay" if problem is None and len(comparisons) == 11 and all(count_checks.values()) and dt_ok else "failed"
    result = {"status": status, "bindings": bindings, "comparisons": comparisons,
              "calls": receipt, "count_checks": count_checks, "physics_dt_check": dt_ok,
              "journal_sha256": sha(OUT / "journal.jsonl"), "error": problem,
              "planning_executed": False, "neural_executed": False, "pc_gate_pass": False}
    base.save_json(OUT / "result.json", result)
    print(json.dumps({"status": status, "frames": len(comparisons), "count_checks": count_checks,
                      "error": problem}, indent=2), flush=True)
    if status != "exact_boundary_replay":
        raise RuntimeError("replay failed; evidence retained")


if __name__ == "__main__":
    run()
