"""Post-seal audit of actually executed goal-mode feedback prefixes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_goal_mode_matched_feedback_20261103_codex as run

PHYSICAL = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")
OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_selected_audit.json")
PROTOCOL = run.ROOT.parent / "docs/PC_MATCHED_FEEDBACK_SELECTED_CHAIN_AUDIT_PROTOCOL_20261103_CODEX.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_goal_row(physical, decisions, execution):
    root = physical["root"]
    distances = np.asarray(physical["distances"], dtype=float)
    if (physical["arm"] != "goal" or distances.shape != (4,)
            or not np.isfinite(distances).all()
            or len(decisions) != 3 or len(execution["rows"]) != 3):
        raise ValueError("invalid physical record")
    records = []
    for step, (decision, executed) in enumerate(zip(decisions, execution["rows"], strict=True)):
        selected = decision["result"]["selected"]
        if (decision["root"] != root or decision["step"] != step or decision["arm"] != "goal"
                or executed["step"] != step or executed["actions"] != selected["prefix"]):
            raise ValueError("selected action binding mismatch")
        model_change = float(selected["prefix_score"] - decision["initial_score"])
        physical_change = float(distances[step + 1] - distances[step])
        records.append({"root": root, "step": step, "selected": selected["name"],
                        "same_bank_greedy": decision["result"]["same_bank_greedy_name"],
                        "abstained": decision["result"]["selection"]["abstained"],
                        "physical_before": float(distances[step]),
                        "physical_after": float(distances[step + 1]),
                        "physical_change": physical_change,
                        "physical_nonimprovement": physical_change >= 0.,
                        "model_prefix_score_change": model_change,
                        "prediction_counts": decision["token_counts"],
                        "compiler_counts": decision["result"]["counts"],
                        "compiler_seconds": decision["result"]["seconds"],
                        "encoding_calls_reported": decision["encoding_calls"],
                        "encoding_seconds": decision["encoding_seconds"],
                        "planning_wall_seconds": decision["planning_seconds"]})
    return records


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    physical = json.loads(PHYSICAL.read_text())
    if physical["backend"] != "2.1.2.14" or len(physical["rows"]) != 20:
        raise ValueError("invalid physical panel")
    goal_rows = [row for row in physical["rows"] if row["arm"] == "goal"]
    if sorted(row["root"] for row in goal_rows) != list(range(4)):
        raise ValueError("goal roots mismatch")
    records, bindings = [], []
    for row in goal_rows:
        decision_paths, execution_path, image_path = run.paths(row["root"], "goal")
        if (row["execution_sha256"] != sha(execution_path)
                or row["images_sha256"] != sha(image_path)
                or row["decision_sha256"] != [sha(path) for path in decision_paths]):
            raise ValueError("physical artifact binding mismatch")
        decisions = [json.loads(path.read_text()) for path in decision_paths]
        execution = json.loads(execution_path.read_text())
        records.extend(audit_goal_row(row, decisions, execution))
        bindings.append({"root": row["root"], "decision_sha256": row["decision_sha256"],
                         "execution_sha256": row["execution_sha256"]})
    result = {"scope": "post-outcome diagnostic, four reused roots",
              "nonimproving_prefixes": sum(row["physical_nonimprovement"] for row in records),
              "number_of_prefixes": len(records), "rows": records, "bindings": bindings,
              "unknown_costs": ["complete shared setup attribution", "FLOPs",
                                "GPU memory peak", "physical replay cost"],
              "full_cost_certified": False, "pc2_pass": False, "pc3_pass": False,
              "physical_sha256": sha(PHYSICAL), "protocol_sha256": sha(PROTOCOL),
              "runner_sha256": sha(Path(__file__))}
    with OUTPUT.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(f"audited {len(records)} actual prefixes; nonimproving="
          f"{result['nonimproving_prefixes']}; no PC gate promotion", flush=True)


if __name__ == "__main__":
    main()
