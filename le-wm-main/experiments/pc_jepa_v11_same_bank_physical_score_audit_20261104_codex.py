"""Rebuild frozen bank, replay it and independently verify rank arithmetic."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_jepa_native_v11_feasible_feedback_dev_20261104_codex as feedback
from experiments.pc_jepa_v11_same_bank_physical_score_diag_20261104_codex import OUT, ROOT, SOURCE
from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior


def independent_order(x, y):
    i, j = np.triu_indices(len(x), 1)
    product = (x[i] - x[j]) * (y[i] - y[j])
    agree, disagree, ties = (int(np.count_nonzero(product > 0)),
                             int(np.count_nonzero(product < 0)),
                             int(np.count_nonzero(product == 0)))
    return {"agree": agree, "disagree": disagree, "tied": ties,
            "valid": agree + disagree, "fraction": agree / (agree + disagree) if agree + disagree else None}


def main():
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    result = json.loads((OUT / "result.json").read_text())
    encoded = json.loads((OUT / "encoder/output/encoded.json").read_text())
    report = json.loads((OUT / "encoder/report.json").read_text())
    if (result["status"] != "complete" or report["status"] != "complete"
            or result["arrays_sha256"] != feedback.sha(OUT / "arrays.npz")
            or report["encoded_sha256"] != feedback.sha(OUT / "encoder/output/encoded.json")):
        raise ValueError("result binding")
    for path, digest in result["bindings"].items():
        if feedback.sha(Path(path)) != digest:
            raise ValueError("changed source")
    decision = json.loads((SOURCE / "response_0/output/decision.json").read_text())
    bank = [(bi, i, a, s) for bi, batch in enumerate(decision["evaluations"])
            if len(batch["actions"][0]) == 15
            for i, (a, s) in enumerate(zip(batch["actions"], batch["scores"], strict=True))][:64]
    with np.load(OUT / "arrays.npz", allow_pickle=False) as package:
        actions, frames, states = (package[k].copy() for k in ("actions", "terminal_rgb", "states"))
    np.testing.assert_array_equal(actions, np.asarray([row[2] for row in bank], np.float32))
    for row, source in zip(result["rows"], bank, strict=True):
        if (row["batch"], row["batch_index"], row["model_score"]) != (source[0], source[1], source[3]):
            raise ValueError("bank ordering")
    with np.load(OUT / "encoder/input/images.npz", allow_pickle=False) as images:
        if set(images.files) != {"goal_rgb", "terminal_rgb"}:
            raise ValueError("encoder input leaked fields")
        np.testing.assert_array_equal(images["terminal_rgb"], frames)
        np.testing.assert_array_equal(images["goal_rgb"], np.load(SOURCE / "goal_rgb.npy", allow_pickle=False))
    if encoded["image_sha256"] != feedback.sha(OUT / "encoder/input/images.npz"):
        raise ValueError("encoder input binding")
    base = prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("v11_bank_independent_replay", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    goal = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False).astype(float)
    physical = []
    for action_chain, frame, saved_state in zip(actions, frames, states, strict=True):
        env = native.PushTEnv(**base.CTOR)
        try:
            env.seed(manifest["seed"])
            obs, state = env.reset()
            for action in np.concatenate((np.asarray(manifest["warmup"], np.float64), action_chain.astype(np.float64))):
                obs, _, done, info = env.step(action)
                if done:
                    raise ValueError("early termination")
                state = info["state"]
            np.testing.assert_array_equal(obs["visual"], frame)
            np.testing.assert_array_equal(state, saved_state)
        finally:
            env.close()
        state = state.astype(float)
        angle = abs((state[4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
        physical.append((float(np.linalg.norm(state[:4] - goal[:4])),
                         float(np.linalg.norm(state[2:4] - goal[2:4])), float(angle)))
    model = np.asarray([item[3] for item in bank])
    actual = np.asarray(encoded["scores"])
    physical = np.asarray(physical)
    for col, key in enumerate(("joint_xy_error", "object_error", "angle_error_radians")):
        np.testing.assert_allclose(physical[:, col], [row["metrics"][key] for row in result["rows"]], atol=0, rtol=0)
        if independent_order(model, physical[:, col]) != result["ordering"][key]:
            raise ValueError("physical ordering")
        if independent_order(actual, physical[:, col]) != report["actual_encoded_vs_physical"][key]:
            raise ValueError("encoder ordering")
    if independent_order(model, actual) != report["model_vs_actual_encoded"]:
        raise ValueError("model ordering")
    audit = {"status": "passed", "count": 64, "result_sha256": feedback.sha(OUT / "result.json"),
             "report_sha256": feedback.sha(OUT / "encoder/report.json"),
             "scope": "same-backend replay and independent rank arithmetic; no neural replay"}
    with (OUT / "audit.json").open("x") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
