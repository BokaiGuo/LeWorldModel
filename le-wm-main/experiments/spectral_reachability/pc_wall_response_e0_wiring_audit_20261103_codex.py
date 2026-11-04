"""Post-outcome action injection and multiscale FD audit, never a planning arm."""
import json
from pathlib import Path

import numpy as np

from . import pc_wall_response_joint_e0_dev_20261103_codex as experiment
from . import pc_wall_retention_coverage_audit_side_20261103 as witness

base, joint = experiment.base, experiment.joint
PROTOCOL = experiment.ROOT.parent / "docs/PC_WALL_RESPONSE_E0_WIRING_AUDIT_PROTOCOL_20261103_CODEX.md"
OUTPUT = experiment.ROOT / "results/pc_wall_response_e0_wiring_audit_20261103_codex.json"


def parameters(actions):
    a = np.asarray(actions)
    p = np.array([a[0, 1] * 2 / np.pi, a[2, 0] / .3,
                  a[6, 1] * 2 / np.pi, a[8, 0] / .3])
    if np.max(np.abs(p)) > 1 + 1e-12 or not np.allclose(joint.decode(p[None])[0], a, atol=1e-12, rtol=0):
        raise ValueError("witness is not representable in declared template")
    return p


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    saved, oracle = (json.loads(p.read_text()) for p in (experiment.OUTPUT, witness.OUTPUT))
    for record in (saved, oracle):
        if any(base.digest(Path(p)) != h for p, h in record["dependencies"].items()):
            raise ValueError("sealed input dependency drift")
    paths = (PROTOCOL, Path(__file__), experiment.OUTPUT, witness.OUTPUT, Path(experiment.__file__),
             Path(witness.__file__), Path(joint.__file__), Path(base.__file__), Path(base.physics.__file__))
    hashes = {str(p): base.digest(p) for p in paths}
    rows = []
    extra_rollouts = 0
    print("Audit only: fixed witness injection and FD scale, no oracle action supplied to planner.", flush=True)
    for savedrow, oracle_row in zip(saved["rows"], oracle["rows"], strict=True):
        if savedrow["index"] != oracle_row["index"]:
            raise ValueError("root mismatch")
        task, index = savedrow["task"], savedrow["index"]
        index0 = oracle_row["witness_indices"][0]
        reference = oracle_row["witnesses"][index0]
        p = parameters(reference["actions"])
        trace, blocked = base.rollout(task["start"], joint.decode(p[None]))
        extra_rollouts += 1
        score = float(np.linalg.norm(trace[0, -1, :2] - task["goal"]))
        if not np.isclose(score, reference["terminal_distance"], atol=1e-12, rtol=0):
            raise ValueError("witness scores differ")
        injected = []
        for cap in experiment.BUDGETS:
            plans = [p for p in savedrow["plans"] if p["rollout_calls"] == cap]
            distances = [p["distance"] for p in plans]
            injected.append({"cap": cap, "witness_distance": score, "planner_distances": distances,
                             "chosen_witness": int(np.argmin([score, *distances])) == 0})
        probes = []
        for plan in savedrow["plans"]:
            if plan["method"] not in ("goal", "singular", "ridge"):
                continue
            for i, evidence in enumerate(plan["response_evidence"]):
                nominal = np.asarray(evidence["nominal"])
                nominal_trace, nominal_blocked = base.rollout(task["start"], joint.decode(nominal[None]))
                extra_rollouts += 1
                matrices = {}
                for epsilon in (.001, .01, .05):
                    points = np.concatenate((nominal + epsilon * np.eye(4), nominal - epsilon * np.eye(4)))
                    end, _ = base.rollout(task["start"], joint.decode(points))
                    extra_rollouts += 8
                    matrices[str(epsilon)] = ((end[:4, -1, :2] - end[4:, -1, :2]) / (2 * epsilon)).T
                saved_matrix = np.asarray(evidence["jacobian"])
                if not np.allclose(matrices["0.01"], saved_matrix, atol=1e-10, rtol=0):
                    raise ValueError("saved finite difference mismatch")
                candidate = np.asarray(plan["candidates"][i])
                candidate_trace, _ = base.rollout(task["start"], joint.decode(candidate[None]))
                extra_rollouts += 1
                center = nominal_trace[0, -1, :2]
                if not np.allclose(center, evidence["endpoint"], atol=1e-12, rtol=0):
                    raise ValueError("nominal endpoint mismatch")
                norm = float(np.linalg.norm(saved_matrix))
                probes.append({"cap": plan["rollout_calls"], "method": plan["method"], "nominal_index": i,
                               "nominal": nominal.tolist(), "candidate": candidate.tolist(),
                               "nominal_distance": float(np.linalg.norm(center - task["goal"])),
                               "candidate_distance": float(np.linalg.norm(candidate_trace[0, -1, :2] - task["goal"])),
                               "final_distance": plan["distance"], "nominal_blocked_steps": int(nominal_blocked.sum()),
                               "singular_values": np.linalg.svd(saved_matrix, compute_uv=False).tolist(),
                               "jacobian_norm": norm,
                               "fd_scales": {e: {"matrix": m.tolist(), "absolute_difference": float(np.linalg.norm(m - saved_matrix)),
                                                  "relative_difference": float(np.linalg.norm(m - saved_matrix) / norm) if norm > 1e-12 else None}
                                             for e, m in matrices.items()}})
        rows.append({"index": index, "witness_index": index0, "witness_parameters": p.tolist(),
                     "witness_distance": score, "witness_blocked_steps": int(blocked.sum()),
                     "injection_comparisons": injected, "response_probes": probes})
        print(f"audited task {index}", flush=True)
    if any(base.digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("source changed")
    probes = [p for row in rows for p in row["response_probes"]]
    summary = {"witnesses_selected": sum(x["chosen_witness"] for row in rows for x in row["injection_comparisons"]),
               "witness_comparisons": sum(len(row["injection_comparisons"]) for row in rows),
               "responses": len(probes), "candidate_improves_nominal": sum(p["candidate_distance"] < p["nominal_distance"] - 1e-12 for p in probes),
               "candidate_worse_than_nominal": sum(p["candidate_distance"] > p["nominal_distance"] + 1e-12 for p in probes),
               "nominal_with_collision": sum(p["nominal_blocked_steps"] > 0 for p in probes),
               "zero_jacobian": sum(p["jacobian_norm"] <= 1e-12 for p in probes),
               "fd_0001_abs_median_max": [float(np.median([p["fd_scales"]["0.001"]["absolute_difference"] for p in probes])),
                                           float(max(p["fd_scales"]["0.001"]["absolute_difference"] for p in probes))],
               "fd_005_abs_median_max": [float(np.median([p["fd_scales"]["0.05"]["absolute_difference"] for p in probes])),
                                          float(max(p["fd_scales"]["0.05"]["absolute_difference"] for p in probes))]}
    with OUTPUT.open("x") as file:
        json.dump({"scope": "privileged post-outcome diagnostic, not method performance",
                   "dependencies": hashes, "extra_rollouts": extra_rollouts, "summary": summary,
                   "rows": rows, "PC2_pass": False, "PC3_pass": False}, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    audit()
