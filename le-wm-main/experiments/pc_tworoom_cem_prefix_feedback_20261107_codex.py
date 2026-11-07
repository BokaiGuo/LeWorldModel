"""Seal RGB-feedback tail scores before comparing prior physical outcomes."""
import hashlib
import json
import sys

import numpy as np
import torch

from experiments import pc_tworoom_cem_selection_alignment_20261107_codex as aligned
from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments import pc_tworoom_selected_prefix_equal_cem_20261107_codex as cem
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = base.ROOT, base.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_CEM_PREFIX_FEEDBACK_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_cem_prefix_feedback_plan_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_cem_prefix_feedback_result_20261107_codex.json"


def choose(cost):
    values = np.asarray(cost, float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("two finite costs required")
    return int(np.argmin(values))


def plan():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if PLAN.exists():
        raise FileExistsError(PLAN)
    cp, old = json.loads(cem.PLAN.read_text()), json.loads(base.PLAN.read_text())
    for obj, key, path in ((cp, "prior_plan_sha256", base.PLAN), (cp, "public_sha256", PANEL.PUBLIC),
                           (cp, "protocol_sha256", cem.PROTOCOL), (cp, "runner_sha256", cem.__file__),
                           (old, "public_sha256", PANEL.PUBLIC), (old, "protocol_sha256", base.PROTOCOL),
                           (old, "runner_sha256", base.__file__), (old, "wide_sha256", base.wide.__file__),
                           (old, "repair_sha256", base.wide.repair.__file__)):
        if obj[key] != file_hash(path):
            raise ValueError(f"input drift: {key}")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    model, scaler, transform, normalizer, boundary = base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != cp["model_sha256"] or normalizer != cp["normalizer_sha256"]:
        raise ValueError("model drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Score both existing tails after actual prefix RGB observation; do not read future outcomes.", flush=True)
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth, torch.inference_mode():
            for i, (p, source) in enumerate(zip(cp["rows"], old["rows"])):
                if p["seed"] != source["seed"] or p["seed"] != int(public["seeds"][i]):
                    raise ValueError("root mismatch")
                for arm in ("selected", "reversed"):
                    optimized = np.asarray(p["arms"][arm]["actions"], np.float32)
                    old_chain = aligned.reconstruct(source, arm)
                    np.testing.assert_array_equal(optimized[:10], old_chain[:10])
                    tails = np.stack((optimized[10:], old_chain[10:]))
                    env.reset(seed=p["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                    actor = env.unwrapped
                    np.testing.assert_array_equal(actor.render(), public["current"][i])
                    prefix_trace = []
                    for action in optimized[:10]:
                        _, _, term, trunc, info = actor.step(action.copy())
                        prefix_trace.append([float(info["distance_to_target"]), bool(term), bool(trunc)])
                    observed = actor.render().copy()
                    ctx = {"pixels": transform(observed)[None, None, None].cuda(),
                           "goal": transform(public["goal"][i])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    normal = scaler.transform(tails.reshape(-1, 2)).reshape(1, 2, 3, 10)
                    scores, _ = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    values = scores[0].cpu().tolist()
                    rows.append({"seed": p["seed"], "arm": arm, "prefix_trace": prefix_trace,
                                 "observed_rgb_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                                 "scores": values, "selected": choose(values), "query_chains": 2,
                                 "model_packets": 6})
                print(f"root={i} reobservation scored", flush=True)
    finally:
        env.close()
    if len(rows) != 80 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    with PLAN.open("x") as f:
        json.dump({"track": "primary_rgb", "rows": rows, "cem_plan_sha256": file_hash(cem.PLAN),
                   "public_sha256": file_hash(PANEL.PUBLIC), "truth_sha256": file_hash(PANEL.TRUTH),
                   "model_sha256": fingerprint, "normalizer_sha256": normalizer,
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__)}, f, allow_nan=False)


def compare():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan, prior = json.loads(PLAN.read_text()), json.loads(aligned.OUTPUT.read_text())
    for obj, key, path in ((plan, "cem_plan_sha256", cem.PLAN), (plan, "public_sha256", PANEL.PUBLIC),
                           (plan, "truth_sha256", PANEL.TRUTH), (plan, "protocol_sha256", PROTOCOL),
                           (plan, "runner_sha256", __file__), (prior, "plan_sha256", cem.PLAN),
                           (prior, "physical_sha256", cem.RESULT), (prior, "fixed_sha256", aligned.fixed.OUTPUT),
                           (prior, "protocol_sha256", aligned.PROTOCOL), (prior, "runner_sha256", aligned.__file__),
                           (prior, "features_sha256", aligned.FEATURES)):
        if obj[key] != file_hash(path):
            raise ValueError(f"artifact drift: {key}")
    if len(plan["rows"]) != 80 or len(prior["rows"]) != 80:
        raise ValueError("pair count drift")
    rows = []
    for row, old in zip(plan["rows"], prior["rows"]):
        if (row["seed"], row["arm"]) != (old["seed"], old["arm"]):
            raise ValueError("pair mismatch")
        a, b = np.asarray(row["prefix_trace"], float), np.asarray(old["traces"][0][:10], float)
        np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(a[:, 1:], b[:, 1:])
        rows.append({"seed": row["seed"], "arm": row["arm"], "selected": row["selected"],
                     "previous_success": old["success"][0], "b_success": old["success"][1],
                     "feedback_success_from_prior_replay": old["success"][row["selected"]],
                     "feedback_scores": row["scores"], "prior_model_scores": [v[-1] for v in old["model_cost"]],
                     "real_endpoint_scores": [v[-1] for v in old["actual_cost"]]})
    with RESULT.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "prior_audit_sha256": file_hash(aligned.OUTPUT),
                   "rows": rows}, f, allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "compare"):
        raise SystemExit("expected plan|compare")
    {"plan": plan, "compare": compare}[sys.argv[1]]()
