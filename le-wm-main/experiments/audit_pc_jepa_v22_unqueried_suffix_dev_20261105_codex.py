"""Independent score arithmetic and same-backend replay of fixed suffix diagnostic."""
import importlib.util
import json
import os

import numpy as np

from experiments import pc_jepa_v22_unqueried_suffix_dev_20261105_codex as run


def independent_order(x, y):
    i, j = np.triu_indices(len(x), 1)
    product = (x[i] - x[j]) * (y[i] - y[j])
    agree, disagree, tied = (int(np.sum(product > 0)), int(np.sum(product < 0)), int(np.sum(product == 0)))
    valid = agree + disagree
    return {"agree": agree, "disagree": disagree, "tied": tied, "valid": valid,
            "fraction": agree / valid if valid else None}


def main():
    out, source, sha = run.OUT, run.SOURCE, run.sha
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    manifest = run.verify()
    physical = json.loads((out / "physical.json").read_text())
    report = json.loads((out / "report.json").read_text())
    pred_path, enc_path = out / "neural/predict/output/decision.json", out / "neural/encode/output/encoded.json"
    p, e = json.loads(pred_path.read_text()), json.loads(enc_path.read_text())
    if (physical["status"] != "completed" or report["status"] != "completed"
            or report["physical_sha256"] != sha(out / "physical.json")
            or report["predict_sha256"] != sha(pred_path) or report["encode_sha256"] != sha(enc_path)
            or physical["arrays_sha256"] != sha(out / "physical_arrays.npz")):
        raise ValueError("result hash mismatch")
    decisions = [json.loads((source / f"{arm}_0/output/decision.json").read_text()) for arm in ("response", "cem")]
    prefixes, suffixes, bank = run.make_bank(*[np.asarray(d["decision"]["selected"]["chain"], np.float32) for d in decisions])
    with np.load(out / "bank.npz", allow_pickle=False) as data:
        for key, value in (("prefixes", prefixes), ("suffixes", suffixes), ("actions", bank)):
            np.testing.assert_array_equal(data[key], value)
    old = {np.asarray(a, np.float32).tobytes() for d in decisions for batch in d["evaluations"]
           for a in batch["actions"] if np.asarray(a).shape == (15, 2)}
    overlap = [i for i, a in enumerate(bank) if a.tobytes() in old]
    if overlap != manifest["exact_overlap_zero_tail_indices"] or any(i % 16 for i in overlap):
        raise ValueError("unqueried check failed")
    np.testing.assert_array_equal(bank, np.load(out / "neural/predict/input/bank.npy", allow_pickle=False))
    if (p["bank_sha256"] != sha(out / "neural/predict/input/bank.npy")
            or p["input_sha256"] != sha(source / "response_0/input/context.npz")
            or e["image_sha256"] != sha(out / "neural/encode/input/images.npz")):
        raise ValueError("worker input binding")
    with np.load(out / "physical_arrays.npz", allow_pickle=False) as data:
        frames, states = data["terminal_rgb"].copy(), data["states"].copy()
    with np.load(out / "neural/encode/input/images.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["terminal_rgb"], frames)
        np.testing.assert_array_equal(data["goal_rgb"], np.load(source / "goal_rgb.npy", allow_pickle=False))
    if os.environ.get("PYTHONPATH") != str((run.ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    base = run.prior.helpers(run.ROOT)
    runtime, _ = base.load_runtime(run.ROOT)
    if physical["environment_sha256"] != sha(run.ROOT / runtime.ENV_REL):
        raise ValueError("environment changed")
    spec = importlib.util.spec_from_file_location("v22_unqueried_independent_env", run.ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    source_manifest = json.loads((source / "manifest.json").read_text())
    goal = np.load(source / "goal_state_evaluation_only.npy", allow_pickle=False)
    for index, actions in enumerate(bank):
        env = native.PushTEnv(**base.CTOR)
        try:
            env.seed(source_manifest["seed"])
            obs, state = env.reset()
            for action in np.concatenate((np.asarray(source_manifest["warmup"], np.float64), actions.astype(np.float64))):
                obs, _, done, info = env.step(action)
                if done:
                    raise ValueError("unexpected termination")
                state = info["state"]
            np.testing.assert_array_equal(obs["visual"], frames[index])
            np.testing.assert_array_equal(state, states[index])
            if run.feedback.metrics(state, goal) != physical["rows"][index]["metrics"]:
                raise ValueError("physical metric mismatch")
        finally:
            env.close()
    predicted, encoded = np.asarray(p["decision"]["scores"]), np.asarray(e["scores"])
    for summary in report["summaries"]:
        start = summary["prefix"] * 16
        section = slice(start + 1, start + 8) if summary["group"] == "construction" else slice(start + 8, start + 16)
        rows = physical["rows"][section]
        for key, x, y in (
            ("predicted_vs_encoded", predicted[section], encoded[section]),
            ("predicted_vs_object_error", predicted[section], np.asarray([r["metrics"]["object_error"] for r in rows])),
            ("predicted_vs_angle_error", predicted[section], np.asarray([r["metrics"]["angle_error_radians"] for r in rows])),
        ):
            if summary[key] != independent_order(x, y):
                raise ValueError("rank arithmetic mismatch")
        error = np.mean(np.abs((predicted[start]-predicted[section])-(encoded[start]-encoded[section])))
        if abs(error - summary["mean_abs_relative_improvement_error"]) > 1e-12:
            raise ValueError("relative error mismatch")
    counts = [len(np.unique(states[i*16:(i+1)*16, 2:5], axis=0)) for i in range(4)]
    if counts != report["pose_counts"]:
        raise ValueError("pose counts mismatch")
    receipt = {"status": "passed", "count": len(bank), "report_sha256": sha(out / "report.json"),
               "physical_sha256": sha(out / "physical.json"), "exact_overlap_indices": overlap,
               "pose_counts": counts, "scope": "same-backend physical replay and CPU arithmetic; no independent neural replay"}
    with (out / "audit.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
