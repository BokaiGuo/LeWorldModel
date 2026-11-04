"""Retrospective mechanisms on sealed post-abstention multidirection chains."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_multidirection_successor_exact_dev_20261103_codex as exp
from experiments import pc_response_diverse_beam_same_chain_diag_20261103_codex as prior_diag

PROTOCOL = exp.previous.prior.ROOT.parent / "docs/PC_MULTIDIRECTION_SUCCESSOR_SAME_CHAIN_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_same_chain_diagnostic.json")


def dependencies():
    prior = exp.previous.previous
    paths = (Path(__file__), PROTOCOL, Path(prior_diag.__file__), exp.PLANS, exp.AUDIT, prior.PLANS, prior.AUDIT)
    return {**exp.dependencies(), **{str(p): exp.sha(p) for p in paths}}


def response(task, prefix):
    epsilon = .05
    directions = np.eye(10).reshape(10, 5, 2)
    tails = np.zeros((20, 10, 2))
    tails[0::2, :5] = epsilon * directions
    tails[1::2, :5] = -epsilon * directions
    actions = np.concatenate((np.broadcast_to(prefix, (20, 5, 2)), tails), axis=1)
    points = exp.previous.prior.exact.endpoints(task, actions.astype(np.float32))
    return ((points[0::2] - points[1::2]) / (2 * epsilon)).T


def analyze(row, previous):
    wrapped = {**row, "arm": "response_diverse"}
    result = prior_diag.analyze(wrapped, previous)
    matrices = {}
    for name, control in result["controls"].items():
        matrix = response(row["task"], np.asarray(control["prefix"]))
        matrices[name] = matrix
        control["successor_response"] = matrix.tolist()
        control["response_singular_values"] = np.linalg.svd(matrix, compute_uv=False).tolist()
    result["response_difference_frobenius"] = {
        name: float(np.linalg.norm(matrices["selected"] - matrices[name]))
        for name in prior_diag.CONTROL_NAMES}
    result["diagnostic_prediction_counts"] = {"common_chains": 30, "replanning_chains": 600,
                                               "response_chains": 100}
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    hashes = dependencies()
    data, old = json.loads(exp.PLANS.read_text()), json.loads(exp.previous.previous.PLANS.read_text())
    if data["dependencies"] != exp.dependencies() or json.loads(exp.AUDIT.read_text())["plans_sha256"] != exp.sha(exp.PLANS):
        raise ValueError("compiler seal mismatch")
    source = exp.previous.previous
    if old["dependencies"] != source.dependencies() or json.loads(source.AUDIT.read_text())["plans_sha256"] != exp.sha(source.PLANS):
        raise ValueError("control seal mismatch")
    previous = {(r["root"], r["arm"]): r for r in old["rows"]}
    print("Post-hoc final-chain common tails, matched replanning and successor responses.", flush=True)
    rows = [analyze(row, previous) for row in data["rows"]]
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} retrospective mechanism rows.", flush=True)


if __name__ == "__main__":
    main()
