"""Post-selection response-scale diagnostic; never alters sealed plans."""
import json

import numpy as np

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as exp

OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_response_diagnostic.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(exp.PLANS.read_text())
    audit = json.loads(exp.AUDIT.read_text())
    intervention = json.loads(exp.INTERVENTION.read_text())
    if (data["dependencies"] != exp.dependencies() or audit["plans_sha256"] != exp.sha(exp.PLANS)
            or intervention["plans_sha256"] != exp.sha(exp.PLANS)):
        raise ValueError("source or receipt mismatch")
    rows = []
    for row in data["rows"]:
        task, result = row["task"], row["result"]
        contexts = [(np.zeros((0, 2)), result["generator"][0])]
        contexts += [(np.asarray(first["prefix"]), evidence[0])
                     for first, evidence in zip(result["first_bank"], result["successor_generator"], strict=True)]
        for index, (prefix, evidence) in enumerate(contexts):
            nominal = np.asarray(evidence["nominal"])
            reference = np.asarray(evidence["jacobian"])
            differences = {}
            for eps in (.01, .05, .1):
                columns = []
                for direction in np.eye(10).reshape(10, 5, 2):
                    actions = np.asarray([np.concatenate((prefix, nominal + eps * direction)),
                                          np.concatenate((prefix, nominal - eps * direction))], dtype=np.float32)
                    positions = exp.exact.endpoints(task, actions)
                    columns.append((positions[0] - positions[1]) / (2 * eps))
                matrix = np.stack(columns, axis=1)
                diff = np.linalg.norm(matrix - reference)
                differences[str(eps)] = {"absolute_frobenius": float(diff),
                                          "relative_frobenius": float(diff / max(np.linalg.norm(reference), 1e-12)),
                                          "reference_frobenius": float(np.linalg.norm(reference))}
            rows.append({"root": row["root"], "context": index, "differences": differences})
    with OUTPUT.open("x") as stream:
        json.dump({"plans_sha256": exp.sha(exp.PLANS), "audit_sha256": exp.sha(exp.AUDIT),
                   "intervention_sha256": exp.sha(exp.INTERVENTION), "rows": rows,
                   "scope": "post-hoc response scale diagnostic; no planner modification"},
                  stream, indent=2, allow_nan=False)
    print(f"Audited response scales at {len(rows)} recorded contexts.", flush=True)


if __name__ == "__main__":
    main()
