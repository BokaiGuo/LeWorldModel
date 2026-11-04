"""Post-outcome collision-pattern analysis for sealed exact successor plans."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as exp

PROTOCOL = exp.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_COLLISION_ATTRIBUTION_DIAG_PROTOCOL_20261103_CODEX.md"
RESPONSE = exp.PREFIX.with_name(exp.PREFIX.name + "_response_diagnostic.json")
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_collision_attribution_diagnostic.json")


def rollout(task, actions):
    actions = np.asarray(actions, dtype=np.float32)
    padded = np.pad(actions.astype(float), ((0, 0), (0, 16 - actions.shape[1]), (0, 0)))
    return exp.exact.wall.rollout(task["start"], padded * exp.exact.wall.HIGH)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plans, response = json.loads(exp.PLANS.read_text()), json.loads(RESPONSE.read_text())
    if (plans["dependencies"] != exp.dependencies() or response["plans_sha256"] != exp.sha(exp.PLANS)
            or response["audit_sha256"] != exp.sha(exp.AUDIT)):
        raise ValueError("source or receipt mismatch")
    response_by_context = {(row["root"], row["context"]): row for row in response["rows"]}
    if len(response_by_context) != 72:
        raise ValueError("response panel mismatch")
    contexts, selected = [], []
    for row in plans["rows"]:
        task, result = row["task"], row["result"]
        chain = np.asarray(result["selected"]["chain"], dtype=np.float32)
        trace, blocked = rollout(task, chain[None])
        final_distance = float(np.linalg.norm(trace[0, 15, :2] - task["goal"]))
        np.testing.assert_allclose(final_distance, result["selected"]["terminal_score"], atol=1e-12, rtol=0)
        selected.append({"root": row["root"], "final_distance": final_distance,
                         "final_x": float(trace[0, 15, 0]), "blocked_steps":
                         np.flatnonzero(blocked[0, :15]).tolist()})
        pairs = [(np.zeros((0, 2)), result["generator"][0])]
        pairs += [(np.asarray(first["prefix"]), evidence[0])
                  for first, evidence in zip(result["first_bank"], result["successor_generator"], strict=True)]
        for idx, (prefix, evidence) in enumerate(pairs):
            nominal = np.asarray(evidence["nominal"])
            reference = np.asarray(evidence["jacobian"])
            base_actions = np.concatenate((prefix, nominal))[None]
            _, base_blocked = rollout(task, base_actions)
            scale_records = {}
            for epsilon in (.01, .05, .1):
                actions = []
                for direction in np.eye(10).reshape(10, 5, 2):
                    actions.extend((np.concatenate((prefix, nominal + epsilon * direction)),
                                    np.concatenate((prefix, nominal - epsilon * direction))))
                trace, events = rollout(task, np.asarray(actions))
                horizon = len(prefix) + 5
                matrix = ((trace[::2, horizon, :2] - trace[1::2, horizon, :2]) / (2 * epsilon)).T
                difference = float(np.linalg.norm(matrix - reference))
                expected = response_by_context[(row["root"], idx)]["differences"][str(epsilon)]
                np.testing.assert_allclose(difference, expected["absolute_frobenius"], rtol=0, atol=1e-12)
                differing = np.any(events[::2, :horizon] != events[1::2, :horizon], axis=1)
                scale_records[str(epsilon)] = {
                    "different_collision_pattern_pairs": int(differing.sum()),
                    "relative_change": expected["relative_frobenius"],
                    "absolute_change": difference,
                    "reference_norm": expected["reference_frobenius"],
                    "positive_negative_blocked": events[:, :horizon].astype(int).tolist()}
            contexts.append({"root": row["root"], "context": idx,
                             "nominal_blocked": base_blocked[0, :len(prefix) + 5].astype(int).tolist(),
                             "scales": scale_records})
    tables = {}
    for scale in ("0.01", "0.1"):
        table = {"pattern_change_matrix_change": 0, "pattern_change_matrix_stable": 0,
                 "same_pattern_matrix_change": 0, "same_pattern_matrix_stable": 0}
        for row in contexts:
            item = row["scales"][scale]
            key = ("pattern_change" if item["different_collision_pattern_pairs"] else "same_pattern")
            key += "_matrix_change" if item["relative_change"] > .1 else "_matrix_stable"
            table[key] += 1
        tables[scale] = table
    with OUTPUT.open("x") as stream:
        json.dump({"plans_sha256": exp.sha(exp.PLANS), "response_sha256": exp.sha(RESPONSE),
                   "protocol_sha256": exp.sha(PROTOCOL), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "selected": selected, "contexts": contexts, "tables": tables},
                  stream, indent=2, allow_nan=False)
    print(f"Audited {len(selected)} selected chains and {len(contexts)} response contexts.", flush=True)


if __name__ == "__main__":
    main()
