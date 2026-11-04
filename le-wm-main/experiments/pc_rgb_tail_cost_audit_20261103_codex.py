"""Audit recorded root and feedback costs without inventing missing metrics."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index(rows):
    result = {}
    for row in rows:
        key = (row["root"], row["budget"], row["method"])
        if key in result:
            raise ValueError("duplicate experimental binding")
        result[key] = row
    return result


def audit(root, feedback, execution, root_hash, feedback_hash, images_hash):
    if (feedback["root_plan_sha256"] != root_hash
            or execution["cp10_sha256"] != feedback_hash
            or execution["images_sha256"] != images_hash):
        raise ValueError("sealed artifact hash mismatch")
    roots, decisions, executed = (index(data["rows"]) for data in (root, feedback, execution))
    if roots.keys() != decisions.keys() or roots.keys() != executed.keys():
        raise ValueError("panel mismatch")
    rows = []
    for key, initial in roots.items():
        final, replay = decisions[key], executed[key]
        chain = initial["chosen"]["chain"]
        choice = final["decision"]
        if (chain[:10] != final["first10"] or chain[10:] != final["old_tail"]
                or final["first10"] != replay["first10"]
                or choice["tail"] != replay["suffix"]
                or len(chain) != 15 or len(replay["suffix"]) != 5):
            raise ValueError("action binding mismatch")
        a, b = initial["counts"], final["counts"]
        if (a["prediction_token_samples"] > a["token_cap"]
                or b["prediction_token_samples"] > b["token_cap"]):
            raise ValueError("prediction cap exceeded")
        rows.append({
            "root": key[0], "budget": key[1], "method": key[2],
            "root_planning": {"counts": a, "generation_counts": initial["generation_counts"],
                              "encoding_calls": initial["encoding_calls"],
                              "encoding_seconds": initial["encoding_seconds"],
                              "svd": initial["svd"],
                              "planning_wall_seconds": initial["planning_wall_seconds"]},
            "feedback_planning": {"counts": b, "encoding_calls": final["encoding_calls"],
                                  "encoding_seconds": final["encoding_seconds"],
                                  "planning_wall_seconds": final["planning_seconds"]},
            "known_prediction_token_samples_total": a["prediction_token_samples"] + b["prediction_token_samples"],
            "selected_execution_actions": len(replay["first10"]) + len(replay["suffix"]),
        })
    return {
        "scope": "sealed root and cp10 records; recorded counters only",
        "full_cost_certified": False,
        "unknown": ["complete derivative and linear algebra resource cost",
                    "complete physical and replay cost", "candidate filtering cost separately",
                    "shared setup allocation", "peak memory and FLOPs"],
        "shared_root_setup": {key: root.get(key) for key in
                              ("shared_model_load_seconds", "shared_goal_encoding_seconds")},
        "rows": rows,
    }


def main():
    root_dir = Path(__file__).resolve().parents[1] / "results"
    root_path = root_dir / "pc_budget_allocation_compare_20261103_codex_root.json"
    prefix = root_dir / "pc_rgb_feedback_tail_retention_20261103_codex_full"
    cp_path = prefix.with_name(prefix.name + "_cp10.json")
    exec_path = prefix.with_name(prefix.name + "_execution.json")
    images = prefix.with_name(prefix.name + "_images.npz")
    output = prefix.with_name(prefix.name + "_cost_audit.json")
    if output.exists():
        raise FileExistsError(output)
    inputs = [json.loads(path.read_text()) for path in (root_path, cp_path, exec_path)]
    report = audit(*inputs, sha(root_path), sha(cp_path), sha(images))
    report["input_sha256"] = {"root": sha(root_path), "feedback": sha(cp_path),
                              "execution": sha(exec_path), "images": sha(images),
                              "audit_runner": sha(Path(__file__))}
    with output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(f"audited {len(report['rows'])} recorded rows; full cost remains uncertified")


if __name__ == "__main__":
    main()
