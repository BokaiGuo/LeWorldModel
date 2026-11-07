"""Paired fixed-candidate RGB CEM selection; scoring and truth are separate phases."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_history_feedback_20261107_codex import RGBFeedbackPlanner
from experiments import pc_rgb_true_history_alignment_20261107_codex as source

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_CEM_INCUMBENT_SELECTION_PROTOCOL_20261107_CODEX.md"
SCORES = ROOT / "results/pc_rgb_cem_incumbent_selection_scores_20261107_codex.json"
EVAL = ROOT / "results/pc_rgb_cem_incumbent_selection_eval_20261107_codex.json"
ROOTS = (0, 1, 8, 9)


class TracingRGB(RGBLeWM):
    def __init__(self, model, audit):
        super().__init__(model, audit)
        self.batches = []

    def score(self, inputs, actions):
        costs, context = super().score(inputs, actions)
        self.batches.append({"actions": actions.detach().cpu().numpy().tolist(),
                             "costs": costs.detach().cpu().numpy().tolist()})
        return costs, context


def choose(batches):
    if len(batches) != 3:
        raise ValueError("expected two CEM rounds and final options")
    actions = np.concatenate([np.asarray(b["actions"])[0] for b in batches])
    scores = np.concatenate([np.asarray(b["costs"])[0] for b in batches])
    if actions.shape != (66, 3, 10) or scores.shape != (66,) or not np.isfinite(scores).all():
        raise ValueError("invalid candidate trace")
    return actions, scores, int(np.argmin(scores)), 64 + int(np.argmin(scores[64:]))


def score():
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import IMAGES
    if SCORES.exists():
        raise FileExistsError(SCORES)
    model, scaler, transform, action_hash = source.model_and_scaler()
    fingerprint = model_fingerprint(model)
    runner = Path(__file__).with_name("pc_rgb_history_feedback_20261107_codex.py")
    contract = Path(__file__).with_name("pc_rgb_history_contract_20261107_codex.py")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, runner, contract, IMAGES],
                         preprocessing="PushT ImageNet transform and expert action scaler",
                         clock="one RGB input; three future 5-action packets")
    rows = []
    with np.load(IMAGES, allow_pickle=False) as images:
        for root in ROOTS:
            frame = images[f"r{root}_white_1_plus_initial"].copy()
            goal = images[f"r{root}_white_goal"].copy()
            wrapper = TracingRGB(model, audit)
            planner = RGBFeedbackPlanner(wrapper, scaler, transform, seed=4200 + root)
            original_raw, _, meta = planner.plan([frame], goal, [], None)
            actions, costs, best, old = choose(wrapper.batches)
            np.testing.assert_allclose(actions[old], meta["normalized_selected_chain"], atol=0, rtol=0)
            np.testing.assert_allclose(costs[64:], meta["option_costs"], atol=0, rtol=0)
            raw = scaler.inverse_transform(actions.reshape(-1, 2)).reshape(66, 3, 5, 2)
            if not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1 + 1e-7:
                raise ValueError("invalid action")
            np.testing.assert_allclose(raw[old], original_raw, atol=1e-7, rtol=0)
            rows.append({"root": root, "batches": wrapper.batches, "receipts": wrapper.receipts,
                         "old_index": old, "best_index": best, "old_raw": raw[old].tolist(),
                         "best_raw": raw[best].tolist(), "planner_meta": meta})
            print(f"root={root} old={costs[old]:.6f} best={costs[best]:.6f}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with SCORES.open("x") as stream:
        json.dump({"rows": rows, "model_sha256": fingerprint, "normalizer_source_sha256": action_hash,
                   "images_sha256": file_hash(IMAGES), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "track": RGB}, stream, allow_nan=False)


def evaluate():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import PHYSICAL, IMAGES
    from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot

    if EVAL.exists():
        raise FileExistsError(EVAL)
    sealed = json.loads(SCORES.read_text())
    if (sealed["runner_sha256"] != file_hash(__file__) or sealed["protocol_sha256"] != file_hash(PROTOCOL)
            or sealed["images_sha256"] != file_hash(IMAGES)):
        raise ValueError("scoring source drift")
    physical = json.loads(PHYSICAL.read_text())
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor, rows = env.unwrapped, []
    try:
        with np.load(IMAGES, allow_pickle=False) as images:
            for row in sealed["rows"]:
                root, task = row["root"], physical["rows"][row["root"]]
                condition = next(c for c in task["conditions"] if c["side"] == 1 and c["appearance"] == "white")
                arms = {}
                for arm in ("old", "best"):
                    env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                           "goal_state": np.asarray(task["goal"])})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    np.testing.assert_allclose(snapshot(actor), condition["actions"]["plus"]["initial_snapshot"],
                                               atol=1e-9, rtol=0)
                    np.testing.assert_array_equal(actor.render(), images[f"r{root}_white_1_plus_initial"])
                    trace = []
                    for action in np.asarray(row[f"{arm}_raw"], np.float32).reshape(15, 2):
                        _, reward, terminated, truncated, _ = actor.step(action.copy())
                        trace.append({"action": action.tolist(), "reward": float(reward),
                                      "terminated": bool(terminated), "truncated": bool(truncated),
                                      "snapshot": snapshot(actor).tolist()})
                        if terminated or truncated:
                            break
                    arms[arm] = trace
                rows.append({"root": root, "arms": arms})
                print(f"root={root} old={arms['old'][-1]['reward']:.4f} best={arms['best'][-1]['reward']:.4f}", flush=True)
    finally:
        env.close()
    with EVAL.open("x") as stream:
        json.dump({"rows": rows, "scores_sha256": file_hash(SCORES), "physical_sha256": file_hash(PHYSICAL),
                   "runner_sha256": file_hash(__file__),
                   "scope": "exposed RGB candidate selection development; no feedback or independent confirmation"},
                  stream, allow_nan=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "evaluate"):
        raise SystemExit("use score or evaluate")
    {"score": score, "evaluate": evaluate}[sys.argv[1]]()
