"""Physical replay of the first sealed prefix/probe/tail pool."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_rgb_hold_retention_feedback_20261107_codex as run
from experiments.pc_input_permissions import file_hash

PROTOCOL = run.ROOT.parent / "docs/PC_RGB_HOLD_RETENTION_FIRST_POOL_PHYSICAL_PROTOCOL_20261107_CODEX.md"
OUT = run.ROOT / "results/pc_tworoom_hold_first_pool_physical_20261107_codex.json"


def sign_report(predicted, physical, eps=1e-6):
    predicted = np.asarray(predicted, dtype=float)
    physical = np.asarray(physical, dtype=float)
    if predicted.shape != physical.shape or not np.isfinite(predicted).all() or not np.isfinite(physical).all():
        raise ValueError("invalid effects")
    informative = (np.abs(predicted) > eps) & (np.abs(physical) > eps)
    return {"total": int(predicted.size), "nontie": int(informative.sum()),
            "correct": int((np.sign(predicted[informative]) == np.sign(physical[informative])).sum()),
            "ties": int((~informative).sum())}


def common_first(root_row):
    decisions = {arm: root_row["arms"][arm]["decisions"][0] for arm in run.search.ARMS}
    base = decisions["lookahead"]
    if base["iteration"] != 0 or base["remaining_steps"] != 25 or base["prefix_steps"] != 10:
        raise ValueError("first-round clock mismatch")
    keys = ("candidate_seed", "old_tail_sha256", "parent_cost", "probe_cost")
    ref_receipts = base["receipts"][:2]
    for d in decisions.values():
        if any(d[k] != base[k] for k in keys):
            raise ValueError("first-round common candidates or scores mismatch")
        for a, b in zip(d["receipts"][:2], ref_receipts, strict=True):
            if a["inputs"] != b["inputs"]:
                raise ValueError("first-round inputs differ")
    if base["candidate_seed"] != 27121000 + root_row["root"] * 10:
        raise ValueError("seed mismatch")
    return decisions


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    saved = json.loads(run.OUT.read_text())
    for key, path in (("runner_sha256", run.__file__), ("search_sha256", run.search.__file__),
                      ("protocol_sha256", run.PROTOCOL), ("input_sha256", run.prior.source.block.prior.inputs.INPUT)):
        if saved[key] != file_hash(path):
            raise ValueError("sealed source drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay sealed first-round candidate probes and independent tail bank.", flush=True)
    try:
        with (np.load(run.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.prior.source.block.prior.common.DATA, "r") as db):
            for row in saved["rows"]:
                root, task = row["root"], row["row"]
                if task != int(inputs["rows"][root]):
                    raise ValueError("row mismatch")
                decisions = common_first(row)
                parents, probes, tails = run.search.timed.build(decisions["effect"]["candidate_seed"], None, 10)
                bank = np.concatenate((np.zeros_like(probes[:1]), probes, tails), axis=0)
                physical, drifts, prefix_success = [], [], []
                initial_success = None
                for i in range(3):
                    entries = []
                    for j in range(5):
                        env.reset(seed=42)
                        actor = env.unwrapped
                        actor._set_state(db["proprio"][task])
                        actor._set_goal_state(db["proprio"][task + 25])
                        initial = bool(float((actor.agent_position - actor.target_position).norm()) < 16.)
                        if initial_success is not None and initial != initial_success:
                            raise ValueError("initial state differs")
                        initial_success = initial
                        chain = np.concatenate((parents[i, :10], bank[j]))
                        distances, successes, positions = [], [], []
                        for action in chain:
                            _, _, term, trunc, info = actor.step(action.copy())
                            if trunc:
                                raise ValueError("unexpected truncation")
                            distances.append(float(info["distance_to_target"]))
                            successes.append(bool(term))
                            positions.append(actor.agent_position.detach().cpu().numpy().copy())
                        positions = np.asarray(positions)
                        drift = float(np.linalg.norm(positions[10:] - positions[9], axis=1).max()) if j == 0 else None
                        entries.append({"distance": distances, "success": successes, "zero_drift": drift})
                    p_success = any(entries[0]["success"][:10])
                    if any(any(e["success"][:10]) != p_success for e in entries):
                        raise ValueError("identical prefix diverged")
                    prefix_success.append(p_success)
                    drifts.append(entries[0]["zero_drift"])
                    physical.append(entries)
                distances = np.asarray([[entry["distance"][-1] for entry in prefix] for prefix in physical])
                actual_effect = distances[:, :1] - distances[:, 1:3]
                predicted_effect = np.asarray(decisions["effect"]["effects"])
                useful = [not initial_success and not prefix_success[i] and any(
                    any(physical[i][j]["success"][10:]) for j in (3, 4)) for i in range(3)]
                recalls = {arm: {"useful_retained": sum(useful[i] for i in d["retained"]),
                                  "useful_total": sum(useful),
                                  "retained_any_useful": any(useful[i] for i in d["retained"])}
                           for arm, d in decisions.items()}
                rows.append({"root": root, "row": task, "initial_success": initial_success, "prefix_success": prefix_success,
                             "zero_max_drifts": drifts, "predicted_effects": predicted_effect.tolist(),
                             "physical_distance_effects": actual_effect.tolist(),
                             "sign_agreement": sign_report(predicted_effect, actual_effect),
                             "useful_prefixes_independent_tails": useful, "recall": recalls,
                             "physical": physical})
                print(f"root={root} hold_drift={max(drifts):.6g} useful={sum(useful)}", flush=True)
    finally:
        env.close()
    with OUT.open("x") as stream:
        json.dump({"scope": "post-hoc physical counterfactual on exposed roots; not deployable",
                   "source_sha256": file_hash(run.OUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
