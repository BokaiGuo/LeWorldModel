"""Audit actual v25 candidates at every recorded search depth, without new calls."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as panel
from experiments.pc_v25_candidate_linearity_posthoc_20261104_codex import candidate_error, pairwise


def audit_layers(layers, calls, segment_length, action_dim, fd_step):
    """Reconstruct each local FD probe and score its generated candidate group."""
    axis = np.eye(segment_length * action_dim).reshape(-1, segment_length, action_dim)
    n = len(axis)
    output = []
    for depth, layer in enumerate(layers):
        if layer["depth"] != depth:
            raise ValueError("nonconsecutive layers")
        parents = ([{"chain": [], "path": []}] if depth == 0 else
                   [layers[depth - 1]["candidates"][i] for i in layers[depth - 1]["retained"]])
        groups = []
        for response in layer["responses"]:
            parent = parents[response["parent"]]
            group = [(index, item) for index, item in enumerate(layer["candidates"])
                     if item["path"][:-1] == parent["path"]
                     and item["path"][-1]["nominal"] == response["nominal"]
                     and np.array_equal(np.asarray(item["chain"][:-segment_length], np.float32).reshape(-1, action_dim),
                                        np.asarray(parent["chain"], np.float32).reshape(-1, action_dim))]
            base = [item for _, item in group if item["path"][-1]["mode"] is None]
            if len(base) != 1:
                raise ValueError("nominal candidate not unique")
            base = base[0]
            base_action = np.asarray(base["chain"], np.float32)
            probes = []
            for actions, features, scores in calls:
                if actions.shape == (1 + 2*n, len(base_action), action_dim) and np.array_equal(actions[0], base_action):
                    expected = np.broadcast_to(base_action, actions.shape).copy()
                    expected[1:n+1, -segment_length:] += fd_step * axis
                    expected[n+1:, -segment_length:] -= fd_step * axis
                    if np.array_equal(actions, expected):
                        augmented = np.column_stack((features, scores))
                        matrix = (augmented[1:n+1] - augmented[n+1:]) / (2 * fd_step)
                        if (np.allclose(matrix.T, response["matrix"], atol=1e-12, rtol=0)
                                and np.allclose(augmented[0], base["feature"], atol=1e-12, rtol=0)):
                            probes.append(matrix.T)
            if not probes:
                raise ValueError("no matching recorded nominal probe")
            records = []
            for index, candidate in group:
                if candidate["path"][-1]["mode"] is None:
                    continue
                record = candidate_error(base_action[-segment_length:], base["feature"], probes[0],
                                         candidate["chain"][-segment_length:], candidate["feature"])
                record.update(index=index, mode=candidate["path"][-1]["mode"],
                              requested_scale=candidate["path"][-1]["scale"])
                records.append(record)
            groups.append({"parent": response["parent"], "nominal": response["nominal"],
                           "candidates": records, "score_pairs": pairwise(records)})
        output.append({"depth": depth, "groups": groups})
    return output


def main():
    if len(sys.argv) != 2:
        raise SystemExit("provide a new output file")
    output = Path(sys.argv[1])
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    plans_path = panel.OUT / "plans.json"
    plans = json.loads(plans_path.read_text())
    audit = json.loads((panel.OUT / "plans_audit.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    if audit["status"] != "passed" or audit["plans_sha256"] != panel.seal.digest(plans_path):
        raise ValueError("source plans not sealed")
    rows = []
    for row in plans["rows"]:
        if row["status"] != "completed":
            rows.append({"root": row["root"], "status": row["status"]})
            continue
        calls = list(panel.fixed.read_calls(panel.OUT / f"plan_root{row['root']:02d}_calls"))
        config = row["result"]["child"]["config"]
        layers = audit_layers(row["result"]["child"]["layers"], calls,
                              config["segment_length"], config["action_dim"], config["fd_step"])
        rows.append({"root": row["root"], "status": "completed", "layers": layers})
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {
        "scope": "post-hoc exact dynamics, saved v25 probes and candidates; not PC1 certification",
        "plans_sha256": panel.seal.digest(plans_path),
        "analysis_sha256": panel.seal.digest(Path(__file__)), "rows": rows})
    print(f"Analyzed {len(rows)} saved roots across search depths without new calls: {output}")


if __name__ == "__main__":
    main()
