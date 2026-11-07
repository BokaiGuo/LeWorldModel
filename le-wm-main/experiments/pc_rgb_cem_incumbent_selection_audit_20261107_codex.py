"""Rescore all stored candidates and replay paired physical executions."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash, tensor_receipt
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments import pc_rgb_cem_incumbent_selection_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_rgb_cem_incumbent_selection_audit_20261107_codex.json"


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_geometric_contact_equivalence_dev_20261106_codex import PHYSICAL, IMAGES
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    score, evaluation = json.loads(run.SCORES.read_text()), json.loads(run.EVAL.read_text())
    if (evaluation["scores_sha256"] != file_hash(run.SCORES)
            or evaluation["physical_sha256"] != file_hash(PHYSICAL)
            or score["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("source drift")
    model, scaler, transform, action_hash = run.source.model_and_scaler()
    if model_fingerprint(model) != score["model_sha256"] or action_hash != score["normalizer_source_sha256"]:
        raise ValueError("model or normalizer changed")
    physical = json.loads(PHYSICAL.read_text())
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor, rows = env.unwrapped, []
    try:
        with np.load(IMAGES, allow_pickle=False) as images, torch.inference_mode():
            for row, ev in zip(score["rows"], evaluation["rows"], strict=True):
                root = row["root"]
                if ev["root"] != root:
                    raise ValueError("root mismatch")
                actions, costs, best, old = run.choose(row["batches"])
                if (best, old) != (row["best_index"], row["old_index"]) or costs[best] > costs[old]:
                    raise ValueError("invalid selection")
                pixel = transform(images[f"r{root}_white_1_plus_initial"])[None, None, None].cuda()
                goal = transform(images[f"r{root}_white_goal"])[None, None, None].cuda()
                max_cost_error = 0.
                for batch, receipt in zip(row["batches"], row["receipts"], strict=True):
                    tensor = torch.tensor(batch["actions"], dtype=torch.float32, device="cuda")
                    if (receipt["inputs"]["pixels"] != tensor_receipt(pixel)
                            or receipt["inputs"]["goal"] != tensor_receipt(goal)
                            or receipt["inputs"]["candidate_actions"] != tensor_receipt(tensor)):
                        raise ValueError("scoring inputs differ")
                    context = {"pixels": pixel, "goal": goal,
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    actual = model.get_cost(context, tensor)
                    max_cost_error = max(max_cost_error, float(np.max(np.abs(
                        actual.cpu().numpy() - np.asarray(batch["costs"])))))
                if max_cost_error > 1e-3:
                    raise ValueError("candidate model cost mismatch")
                raw = scaler.inverse_transform(actions.reshape(-1, 2)).reshape(66, 3, 5, 2)
                task = physical["rows"][root]
                condition = next(c for c in task["conditions"] if c["side"] == 1 and c["appearance"] == "white")
                arms = {}
                for arm, index in (("old", old), ("best", best)):
                    np.testing.assert_allclose(raw[index], row[f"{arm}_raw"], atol=1e-7, rtol=0)
                    env.reset(seed=881001 + root, options={"state": np.asarray(condition["requested_state"]),
                                                           "goal_state": np.asarray(task["goal"])})
                    actor.variation_space["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
                    np.testing.assert_array_equal(actor.render(), images[f"r{root}_white_1_plus_initial"])
                    trace, max_state, max_reward = ev["arms"][arm], 0., 0.
                    for step, saved in enumerate(trace):
                        expected = np.asarray(raw[index], np.float32).reshape(15, 2)[step]
                        np.testing.assert_array_equal(expected, np.asarray(saved["action"], np.float32))
                        _, reward, terminated, truncated, _ = actor.step(expected.copy())
                        max_state = max(max_state, float(np.max(np.abs(snapshot(actor) - saved["snapshot"]))))
                        max_reward = max(max_reward, abs(float(reward) - saved["reward"]))
                        if (bool(terminated), bool(truncated)) != (saved["terminated"], saved["truncated"]):
                            raise ValueError("termination mismatch")
                    if max_state > 1e-9 or max_reward > 1e-9:
                        raise ValueError("physical replay mismatch")
                    arms[arm] = {"steps": len(trace), "reward_5": trace[min(4, len(trace)-1)]["reward"],
                                 "final_reward": trace[-1]["reward"],
                                 "success": any(x["terminated"] for x in trace),
                                 "max_state_error": max_state, "max_reward_error": max_reward}
                rows.append({"root": root, "old_score": float(costs[old]), "best_score": float(costs[best]),
                             "score_error": max_cost_error, "arms": arms})
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "scores_sha256": file_hash(run.SCORES),
                   "eval_sha256": file_hash(run.EVAL), "audit_sha256": file_hash(__file__)},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    audit()
