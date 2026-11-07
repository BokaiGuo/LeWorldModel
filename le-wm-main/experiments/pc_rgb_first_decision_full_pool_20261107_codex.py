"""Reconstruct first-decision scored pools and replay missing parent chains."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_hold_first_pool_physical_20261107_codex as prior
from experiments.pc_input_permissions import file_hash
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored

ROOT = prior.run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_FIRST_DECISION_FULL_POOL_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_tworoom_first_decision_full_pool_20261107_codex.json"


def build_scored(decision, parents, physical, parent_traces):
    kept = decision["retained"]
    if len(kept) != 2 or len(set(kept)) != 2 or any(i not in range(3) for i in kept):
        raise ValueError("invalid retained indices")
    scores = (np.asarray(decision["parent_cost"]), np.asarray(decision["probe_cost"]).reshape(-1),
              np.asarray(decision["tail_cost"]))
    if tuple(map(len, scores)) != (3, 9, 4):
        raise ValueError("wrong scored pool size")
    candidates = ([ScoredCandidate("parent", i, float(v)) for i, v in enumerate(scores[0])] +
                  [ScoredCandidate("probe", i, float(v)) for i, v in enumerate(scores[1])] +
                  [ScoredCandidate("tail", i, float(v)) for i, v in enumerate(scores[2])])
    traces = (list(parent_traces) + [physical[i][j] for i in range(3) for j in range(3)] +
              [physical[i][j] for i in kept for j in (3, 4)])
    if any(len(t["success"]) != 25 or len(t["distance"]) != 25 for t in traces):
        raise ValueError("invalid trace lengths")
    winner = select_scored(candidates)
    expected = decision["choice"]
    if (winner.source, winner.index, winner.cost) != (expected["source"], expected["index"], expected["cost"]):
        raise ValueError("logged choice does not match candidate scores")
    offset = {"parent": 0, "probe": 3, "tail": 12}[winner.source]
    idx = offset + winner.index
    ever = [any(t["success"]) for t in traces]
    endpoint = [t["success"][-1] for t in traces]
    return {"choice_flat_index": idx, "choice": expected, "ever_oracle": any(ever),
            "ever_chosen": ever[idx], "endpoint_oracle": any(endpoint),
            "endpoint_chosen": endpoint[idx], "ever": ever, "endpoint": endpoint}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    p = json.loads(prior.OUT.read_text())
    saved = json.loads(prior.run.OUT.read_text())
    if p["source_sha256"] != file_hash(prior.run.OUT) or p["runner_sha256"] != file_hash(prior.__file__):
        raise ValueError("sealed physical/source mismatch")
    if len(p["rows"]) != len(saved["rows"]):
        raise ValueError("root count mismatch")
    input_path = prior.run.prior.source.block.prior.inputs.INPUT
    data_path = prior.run.prior.source.block.prior.common.DATA
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Replay missing parents; measure each arm's actual first scored-pool losses.", flush=True)
    try:
        with np.load(input_path, allow_pickle=False) as inputs, h5py.File(data_path, "r") as data:
            for row, original in zip(p["rows"], saved["rows"], strict=True):
                root, task = row["root"], row["row"]
                if (root, task) != (original["root"], original["row"]) or int(inputs["rows"][root]) != task:
                    raise ValueError("root mismatch")
                decisions = prior.common_first(original)
                parents, probes, tails = prior.run.search.timed.build(decisions["effect"]["candidate_seed"], None, 10)
                parent_traces = []
                for parent in parents:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][task])
                    actor._set_goal_state(data["proprio"][task + 25])
                    start_success = bool(float((actor.agent_position - actor.target_position).norm()) < 16.)
                    if start_success != row["initial_success"]:
                        raise ValueError("initial success mismatch")
                    trace = {"distance": [], "success": []}
                    for action in parent:
                        _, _, term, trunc, info = actor.step(action.copy())
                        if trunc:
                            raise ValueError("unexpected truncation")
                        trace["distance"].append(float(info["distance_to_target"]))
                        trace["success"].append(bool(term))
                    parent_traces.append(trace)
                arms = {}
                probe_bank = np.concatenate((np.zeros_like(probes[:1]), probes))
                for arm, d in decisions.items():
                    result = build_scored(d, parents, row["physical"], parent_traces)
                    choice = d["choice"]
                    idx = choice["index"]
                    if choice["source"] == "parent":
                        chain = parents[idx]
                    elif choice["source"] == "probe":
                        chain = np.concatenate((parents[idx // 3, :10], probe_bank[idx % 3]))
                    elif choice["source"] == "tail":
                        chain = np.concatenate((parents[d["retained"][idx // 2], :10], tails[idx % 2]))
                    else:
                        raise ValueError("unknown source")
                    if prior.run.prior.action_hash(chain) != d["actions_sha256"]:
                        raise ValueError("selected chain hash mismatch")
                    count = len(d["dispatched"])
                    np.testing.assert_array_equal(chain[:count], np.asarray(d["dispatched"], dtype=np.float32))
                    chosen_trace = (parent_traces if choice["source"] == "parent" else
                                    [row["physical"][idx // 3][idx % 3]] if choice["source"] == "probe" else
                                    [row["physical"][d["retained"][idx // 2]][3 + idx % 2]])
                    selected_trace = chosen_trace[idx] if choice["source"] == "parent" else chosen_trace[0]
                    recorded = original["arms"][arm]["trace"][:count]
                    np.testing.assert_allclose(selected_trace["distance"][:count],
                                               [x["distance"] for x in recorded], rtol=0, atol=0)
                    arms[arm] = result
                rows.append({"root": root, "row": task, "initial_success": row["initial_success"],
                             "parents_physical": parent_traces, "arms": arms})
                print(f"root={root} parents_replayed=3", flush=True)
    finally:
        env.close()
    eligible = [r for r in rows if not r["initial_success"]]
    summary = {}
    for arm in prior.run.search.ARMS:
        summary[arm] = {metric: sum(r["arms"][arm][metric] for r in eligible)
                        for metric in ("ever_oracle", "ever_chosen", "endpoint_oracle", "endpoint_chosen")}
    with OUT.open("x") as stream:
        json.dump({"scope": "post-hoc, exposed roots, first-decision open-loop counterfactual",
                   "source_sha256": file_hash(prior.OUT), "run_sha256": file_hash(prior.run.OUT),
                   "input_sha256": file_hash(input_path), "data_sha256": file_hash(data_path),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "eligible_roots": len(eligible), "summary": summary, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
