"""Audit model and real encoded trajectories for sealed CEM/B-tail pairs."""
import json

import numpy as np
import torch

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments import pc_tworoom_selected_prefix_equal_cem_20261107_codex as cem
from experiments import pc_tworoom_selected_prefix_intervention_20261107_codex as fixed
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = base.ROOT, base.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CEM_SELECTION_ALIGNMENT_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_tworoom_cem_selection_alignment_20261107_codex.json"
FEATURES = ROOT / "results/pc_tworoom_cem_selection_alignment_features_20261107_codex.npz"


def validate_trace(actual, prior):
    a, b = np.asarray(actual, float), np.asarray(prior, float)
    if a.shape != (25, 3) or b.shape != (25, 3):
        raise ValueError("invalid trace shape")
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-5, rtol=0)
    np.testing.assert_array_equal(a[:, 1:], b[:, 1:])


def reconstruct(old, arm):
    p, t = old["selected"]["descriptor"]
    prefix = np.asarray(old["prefixes"][p], np.float32)
    if arm == "reversed":
        prefix = prefix[::-1]
    elif arm != "selected":
        raise ValueError("unknown arm")
    return np.concatenate((prefix, np.asarray(old["tails"][5 + t], np.float32)))


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists() or FEATURES.exists():
        raise FileExistsError("audit artifact exists")
    cp, cr, old, fixed_result = (json.loads(p.read_text()) for p in
                                 (cem.PLAN, cem.RESULT, base.PLAN, fixed.OUTPUT))
    for obj, key, path in ((cp, "prior_plan_sha256", base.PLAN), (cp, "public_sha256", PANEL.PUBLIC),
                           (cp, "protocol_sha256", cem.PROTOCOL), (cp, "runner_sha256", cem.__file__),
                           (cr, "plan_sha256", cem.PLAN), (cr, "truth_sha256", PANEL.TRUTH),
                           (fixed_result, "plan_sha256", base.PLAN), (fixed_result, "protocol_sha256", fixed.PROTOCOL),
                           (fixed_result, "runner_sha256", fixed.__file__),
                           (old, "protocol_sha256", base.PROTOCOL), (old, "runner_sha256", base.__file__),
                           (old, "public_sha256", PANEL.PUBLIC), (old, "wide_sha256", base.wide.__file__),
                           (old, "repair_sha256", base.wide.repair.__file__)):
        if obj[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    if not all(len(x["rows"]) == 40 for x in (cp, cr, old, fixed_result)):
        raise ValueError("panel drift")
    model, scaler, transform, normalizer, boundary = base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != cp["model_sha256"] or normalizer != cp["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows, arrays = [], {}
    print("Recompute paired frozen-model and real-image trajectories, verifying original physical traces.", flush=True)
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, (p, r, source, fr) in enumerate(zip(cp["rows"], cr["rows"], old["rows"], fixed_result["rows"])):
                if any(x["seed"] != int(public["seeds"][i]) for x in (p, r, source, fr)):
                    raise ValueError("root mismatch")
                for arm in ("selected", "reversed"):
                    optimized = np.asarray(p["arms"][arm]["actions"], np.float32)
                    reference_chain = reconstruct(source, arm)
                    np.testing.assert_array_equal(optimized[:10], reference_chain[:10])
                    first = np.asarray(p["arms"][arm]["generations"][0]["tails"][:4], np.float32)
                    if not any(np.array_equal(reference_chain[10:], t) for t in first):
                        raise ValueError("reference tail not in first CEM population")
                    actions = np.stack([optimized, reference_chain])
                    ctx = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                           "goal": transform(public["goal"][i])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    normal = scaler.transform(actions.reshape(-1, 2)).reshape(1, 2, 5, 10)
                    endpoint, out = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    pred, goal = out["predicted_emb"][0, :, 1:], out["goal_emb"][0, -1]
                    np.testing.assert_allclose(float(endpoint[0, 0]), p["arms"][arm]["cost"], atol=1e-3, rtol=1e-5)
                    images, traces = [], []
                    for chain in actions:
                        env.reset(seed=p["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                        actor = env.unwrapped
                        np.testing.assert_array_equal(actor.render(), public["current"][i])
                        trace = []
                        for step, action in enumerate(chain):
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append([float(info["distance_to_target"]), bool(term), bool(trunc)])
                            if (step + 1) % 5 == 0:
                                images.append(actor.render().copy())
                        traces.append(trace)
                    validate_trace(traces[0], r["traces"][arm])
                    validate_trace(traces[1], fr["traces"][arm][p["old_tail_index"]])
                    batch = torch.stack([transform(im) for im in images]).cuda()
                    actual = boundary.encode({"pixels": batch[:, None]})["emb"][:, 0].reshape(2, 5, -1)
                    model_cost = (pred - goal).square().sum(-1)
                    actual_cost = (actual - goal).square().sum(-1)
                    arrays[f"pred_{i}_{arm}"] = pred.cpu().numpy()
                    arrays[f"actual_{i}_{arm}"] = actual.cpu().numpy()
                    arrays[f"goal_{i}_{arm}"] = goal.cpu().numpy()
                    rows.append({"seed": p["seed"], "arm": arm, "model_cost": model_cost.cpu().tolist(),
                                 "actual_cost": actual_cost.cpu().tolist(), "traces": traces,
                                 "success": [any(s[1] for s in t) for t in traces]})
                print(f"root={i} paired arms verified", flush=True)
    finally:
        env.close()
    if len(rows) != 80 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    with FEATURES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-outcome exposed development audit", "rows": rows,
                   "plan_sha256": file_hash(cem.PLAN), "physical_sha256": file_hash(cem.RESULT),
                   "fixed_sha256": file_hash(fixed.OUTPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "features_sha256": file_hash(FEATURES), "runner_sha256": file_hash(__file__)}, f, allow_nan=False)


if __name__ == "__main__":
    main()
