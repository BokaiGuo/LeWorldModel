"""Re-measure all sealed V25 chains at their unchanged float32 actions."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.pc_boundary_response_posthoc import probes, differences
from experiments.pc_v5_e0_prospective_interventions_dev_20261104_codex import exact

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_v25_narrow_prospective_same_chain_dev_20261104_codex/interventions.json"
OUT = ROOT / "results/pc_boundary_response_posthoc_20261104_codex"
NAMES = ("selected", "zero", "greedy", "lookahead", "random")


def emit(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main():
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    rows = source["rows"]
    if len(rows) != 16 or any(x["status"] != "completed" for x in rows):
        raise ValueError("expected all 16 completed sealed interventions")
    OUT.mkdir(exist_ok=False)
    with (OUT / "journal.jsonl").open("x") as journal:
        emit(journal, {"type": "manifest", "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "scope": "posthoc measurement sensitivity; not PC2"})
        for entry in rows:
            row = entry["data"]
            try:
                chain = np.asarray(row["selected_chain"], np.float32)
                tail = chain[5:].copy()
                goal = np.asarray(row["task"]["goal"])
                for name in NAMES:
                    prefix = np.asarray(row["controls"][name]["prefix"], np.float32)
                    for eps in (.01, .05):
                        tails, specs = probes(tail, eps)
                        actions = np.concatenate((np.broadcast_to(prefix, (len(tails), 5, 2)), tails), axis=1)
                        ident = {"root": row["root"], "control": name, "epsilon": eps}
                        emit(journal, {"type": "request", **ident, "actions": actions.tolist(),
                                       "specs": specs, "endpoint_chains": len(actions),
                                       "wall_chains": len(actions)})
                        endpoint = exact.endpoints(row["task"], actions)
                        trajectory, blocked = exact.wall.rollout(
                            row["task"]["start"],
                            np.pad(actions, ((0, 0), (0, 1), (0, 0))) * exact.wall.HIGH)
                        np.testing.assert_allclose(trajectory[:, 15, :2], endpoint, rtol=0, atol=1e-12)
                        blocked = np.asarray(blocked)[:, :15]
                        matrix, stable = differences(endpoint, blocked, specs)
                        delta = endpoint[0] - goal
                        distance = float(np.linalg.norm(delta))
                        gradient = None if distance <= 1e-10 else (matrix.T @ (delta / distance)).tolist()
                        old = row["response"]["controls"][name]["scales"][str(eps)]
                        old_matrix = np.asarray(old["matrix"])
                        old_gradient = old["distance_gradient"]
                        emit(journal, {"type": "response", **ident, "endpoint": endpoint.tolist(),
                                       "blocked": blocked.tolist(), "matrix": matrix.tolist(),
                                       "gradient": gradient, "coordinate_branch_stable": stable,
                                       "all_branch_stable": all(stable),
                                       "old_branch_stable": old["branch_stable"],
                                       "tail_clip_linf": row["response"]["tail_clip_linf"],
                                       "matrix_old_gap": float(np.linalg.norm(matrix - old_matrix)),
                                       "gradient_old_gap": None if gradient is None or old_gradient is None else
                                       float(np.linalg.norm(np.asarray(gradient) - old_gradient))})
                emit(journal, {"type": "root_complete", "root": row["root"]})
                print(f"root {row['root']}: measured all controls", flush=True)
            except Exception as error:
                emit(journal, {"type": "root_failure", "root": row["root"],
                               "exception": repr(error)})
                raise
        emit(journal, {"type": "complete", "roots": len(rows)})


if __name__ == "__main__":
    print("Posthoc sensitivity only: no gate reclassification.", flush=True)
    main()
