"""Replay a predeclared slice of an already sealed v11 prediction bank."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np

from experiments import pc_jepa_native_v12_contact_feedback_dev_20261104_codex as feedback
from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior

ROOT, SOURCE = feedback.ROOT, feedback.OUT
OUT = ROOT / "results/pc_jepa_v12_contact_score_bank_posthoc_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V12_CONTACT_SCORE_BANK_POSTHOC_PROTOCOL_20261104_CODEX.md"


def order(model, physical):
    agree = disagree = tied = 0
    for i in range(len(model)):
        for j in range(i + 1, len(model)):
            a, b = model[i] - model[j], physical[i] - physical[j]
            if a == 0 or b == 0:
                tied += 1
            elif np.sign(a) == np.sign(b):
                agree += 1
            else:
                disagree += 1
    return {"agree": agree, "disagree": disagree, "tied": tied,
            "valid": agree + disagree, "fraction": agree / (agree + disagree) if agree + disagree else None}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("exact physics overlay required")
    import gym
    import pymunk
    import skimage
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("physics version mismatch")
    base = prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")
    if probe["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime probe failure")
    spec = importlib.util.spec_from_file_location("pc_jepa_v11_bank_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    audit = json.loads((SOURCE / "audit.json").read_text())
    replay = json.loads((SOURCE / "physical_replay.json").read_text())
    if (audit["status"] != "saved_data_audit_passed" or replay["status"] != "exact"
            or audit["summary_sha256"] != feedback.sha(SOURCE / "summary.json")
            or replay["audit_sha256"] != feedback.sha(SOURCE / "audit.json")):
        raise ValueError("unverified source")
    decision_path = SOURCE / "response_0/output/decision.json"
    decision = json.loads(decision_path.read_text())
    bank = []
    for batch_index, batch in enumerate(decision["evaluations"]):
        if len(batch["actions"][0]) != 15:
            continue
        for index, (action, score) in enumerate(zip(batch["actions"], batch["scores"], strict=True)):
            bank.append((batch_index, index, np.asarray(action, np.float32), float(score)))
    if len(bank) < 56:
        raise ValueError("short saved bank")
    bank = [bank[i] for i in np.linspace(0, len(bank) - 1, 56, dtype=int)]
    witness_path = SOURCE / "witness_actions_evaluation_only.npy"
    witness = np.load(witness_path, allow_pickle=False).astype(np.float32)
    privileged = [(witness * scale).astype(np.float32) for scale in (0., .25, .5, .75, 1.)]
    privileged.extend((np.concatenate((witness[:5], np.zeros((10, 2), np.float32))),
                       np.concatenate((witness[:10], np.zeros((5, 2), np.float32))),
                       witness[::-1].copy()))
    bank.extend((None, i, action, None) for i, action in enumerate(privileged))
    goal = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False)
    boundary = SOURCE / "response_physical/frame_10"
    expected = {key: np.load(boundary / f"{key}.npy", allow_pickle=False)
                for key in ("rgb", "state", "proprio")}
    source_files = [PROTOCOL, Path(__file__), decision_path, SOURCE / "manifest.json",
                    SOURCE / "summary.json", SOURCE / "audit.json", SOURCE / "physical_replay.json",
                    SOURCE / "goal_state_evaluation_only.npy", witness_path, ROOT / runtime.ENV_REL,
                    *(boundary / f"{key}.npy" for key in expected)]
    bindings = {str(p): feedback.sha(p) for p in source_files}
    OUT.mkdir()
    rows, frames, states = [], [], []
    try:
        for i, (batch, index, actions, score) in enumerate(bank):
            env = native.PushTEnv(**base.CTOR)
            try:
                env.seed(manifest["seed"])
                obs, state = env.reset()
                for action in np.asarray(manifest["warmup"], np.float64):
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("warmup terminated")
                    state = info["state"]
                for key, value in (("rgb", obs["visual"]), ("state", state), ("proprio", obs["proprio"])):
                    np.testing.assert_array_equal(value, expected[key])
                for action in actions.astype(np.float64):
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("candidate terminated")
                    state = info["state"]
                frames.append(obs["visual"].copy())
                states.append(state.copy())
                rows.append({"index": i, "batch": batch, "batch_index": index, "model_score": score,
                             "group": "planner_slice" if i < 56 else "privileged_witness_transform",
                             "status": "completed", "metrics": feedback.metrics(state, goal)})
            finally:
                env.close()
            if (i + 1) % 16 == 0:
                print(f"replayed {i + 1}/64 frozen chains", flush=True)
        if any(feedback.sha(Path(path)) != value for path, value in bindings.items()):
            raise ValueError("source changed")
        actions = np.stack([item[2] for item in bank])
        with (OUT / "arrays.npz").open("xb") as handle:
            np.savez_compressed(handle, actions=actions, terminal_rgb=np.stack(frames), states=np.stack(states))
        model = np.asarray([row["model_score"] for row in rows[:56]])
        comparisons = {key: order(model, np.asarray([row["metrics"][key] for row in rows[:56]]))
                       for key in ("joint_xy_error", "object_error", "angle_error_radians")}
        result = {"status": "complete", "bindings": bindings, "count": len(rows),
                  "rows": rows, "ordering": comparisons, "arrays_sha256": feedback.sha(OUT / "arrays.npz"),
                  "scope": "post-hoc correlated bank, one constructed root; no planner feedback"}
    except Exception:
        result = {"status": "failed", "bindings": bindings, "rows": rows, "error": traceback.format_exc()}
    with (OUT / "result.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result.get(key) for key in ("status", "count", "ordering", "error")}), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("physical bank diagnostic failed; records retained")


if __name__ == "__main__":
    main()
