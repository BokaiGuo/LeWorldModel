"""Evaluate the complete 3-prefix x 2-tail pool at matched recorded RGB decisions."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as run
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = run.ROOT.parent / "docs/PC_RGB_RETENTION_COUNTERFACTUAL_POOL_PROTOCOL_20261107_CODEX.md"
OUTPUT = run.ROOT / "results/pc_tworoom_retention_counterfactual_pool_20261107_codex.json"


def matched_decisions(row):
    """Yield only shared histories and old tails before first dispatched-action divergence."""
    old_a = old_b = None
    history = []
    for a, b in zip(row["arms"]["lookahead"]["decisions"], row["arms"]["effect"]["decisions"]):
        if a["iteration"] != b["iteration"] or a["remaining"] != b["remaining"]:
            raise ValueError("decision clock mismatch")
        if old_a == old_b and a["remaining"] > 5:
            if a["candidate_seed"] != b["candidate_seed"]:
                raise ValueError("candidate identity mismatch")
            yield a, b, history.copy(), old_a
        old_a = a["actions"][len(a["dispatched"]):]
        old_b = b["actions"][len(b["dispatched"]):]
        if a["dispatched"] != b["dispatched"]:
            break
        history.extend(a["dispatched"])


def candidate_pool(decision, old_tail):
    remaining = decision["remaining"]
    prefix = min(10, remaining - 5)
    parents, _, tails = run.program.base.timed.build(decision["candidate_seed"], old_tail, prefix)
    parents = parents[:, :remaining]
    tails = tails[:, :remaining-prefix]
    chains = np.stack([np.concatenate((parents[i, :prefix], tails[j])) for i in range(3) for j in range(2)])
    if chains.shape != (6, remaining, 2) or not np.isfinite(chains).all() or np.abs(chains).max() > 1:
        raise ValueError("invalid counterfactual pool")
    return prefix, chains


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = json.loads(run.OUTPUT.read_text())
    if (original["runner_sha256"] != file_hash(run.__file__)
            or original["protocol_sha256"] != file_hash(run.PROTOCOL)
            or original["input_sha256"] != file_hash(run.INPUT)):
        raise ValueError("sealed source drift")
    model, scaler, transform, normalizer = run.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, run.OUTPUT, run.INPUT,
                                      run.common.CHECKPOINT / "weights.pt", run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224, dataset action scaler",
                         clock="remaining deadline, five environment steps per model packet")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    entries = []
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.common.DATA, "r") as data, torch.inference_mode()):
            for idx, row in enumerate(original["rows"]):
                if row["row"] != int(inputs["rows"][idx]) or row["episode"] != int(inputs["episodes"][idx]):
                    raise ValueError("episode binding mismatch")

                def reset_and_history(history):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    for k, action in enumerate(history):
                        _, _, term, trunc, info = actor.step(np.asarray(action, dtype=np.float32))
                        expected = row["arms"]["lookahead"]["trace"][k]
                        if (bool(term) != expected["terminated"] or bool(trunc) != expected["truncated"]
                                or abs(float(info["distance_to_target"]) - expected["distance"]) > 1e-5):
                            raise ValueError("physical history mismatch")
                    return actor

                goal = transform(inputs["goal"][idx])[None, None, None].cuda()
                for look, effect, history, old_tail in matched_decisions(row):
                    prefix, chains = candidate_pool(look, old_tail)
                    actor = reset_and_history(history)
                    initial_success = bool(float((actor.agent_position - actor.target_position).norm()) < 16.)
                    ctx = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    normal = scaler.transform(chains.reshape(-1, 2)).reshape(1, 6, look["remaining"] // 5, 10)
                    boundary = RGBLeWM(model, audit)
                    endpoint, context = boundary.score(ctx, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    pred, target = context["predicted_emb"], context["goal_emb"][:, -1]
                    if pred.shape[2] != look["remaining"] // 5 + 1:
                        raise ValueError("prediction clock mismatch")
                    temporal = (pred[:, :, 1:] - target[:, None, None]).square().sum(-1).min(-1).values[0].cpu().numpy()
                    endpoint = endpoint[0].cpu().numpy()
                    for decision in (look, effect):
                        for slot, retained in enumerate(decision["retained"]):
                            np.testing.assert_allclose(temporal[2*retained:2*retained+2],
                                                       decision["temporal_scores"][2][2*slot:2*slot+2],
                                                       rtol=0, atol=1e-3)
                            np.testing.assert_allclose(endpoint[2*retained:2*retained+2],
                                                       decision["tail_cost"][2*slot:2*slot+2],
                                                       rtol=0, atol=1e-3)
                    physical = []
                    for chain in chains:
                        actor = reset_and_history(history)
                        trace = []
                        for action in chain:
                            _, _, term, trunc, info = actor.step(action.copy())
                            trace.append({"distance": float(info["distance_to_target"]), "success": bool(term),
                                          "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        physical.append(trace)
                    entries.append({"episode": row["episode"], "iteration": look["iteration"],
                                    "remaining": look["remaining"], "history": history, "prefix": prefix,
                                    "initial_success": initial_success, "actions": chains.tolist(),
                                    "temporal_cost": temporal.tolist(), "endpoint_cost": endpoint.tolist(),
                                    "retained": {"lookahead": look["retained"], "effect": effect["retained"]},
                                    "choice": {"lookahead": look["choice"], "effect": effect["choice"]},
                                    "physical": physical})
                print(f"episode={row['episode']} pooled={len(entries)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(run.OUTPUT), "input_sha256": file_hash(run.INPUT),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "model_sha256": fingerprint, "entries": entries}, stream, allow_nan=False)
    print(f"sealed {len(entries)} matched decision pools", flush=True)


if __name__ == "__main__":
    main()
