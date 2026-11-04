"""CPU source preflight only; never loads or executes a neural model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as proposal

ROOT = Path(__file__).resolve().parents[2]
NAME = "reacher_pc_p1_common_tail_preflight_20261102_codexexclusive"
Q_PATH = ROOT / "results/reacher_pc_goal_q_selection_20261102_v1/report.json"
SOURCE = ROOT / "results/reacher_p1_successor_cem_development_20261102_v1/report.json"
BANK = ROOT / "results/reacher_pc_feasible_prefix_development_20261102_v1/report.json"
Q_SHA = "2b128d35b9f0bceb23247a8b8b16086f97d142ee5a0ce18aeeb9c7f9f6714fb8"
SOURCE_SHA = "647b93058349f8f58f4a0947d9440149d1b98ecdd6b22899483d7e341d33124a"
BANK_SHA = "949622e6a1e0dada510435dc194f7a5f4130b87f7b716422517ce2299dd1ad09"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_metadata(q, source, bank):
    require(q["source_report_sha256"] == bank["source_report_sha256"] == SOURCE_SHA and
            q["fixed_bank_sha256"] == BANK_SHA, "model-only report binding")
    for key in ("weights_sha256", "action_stats_sha256", "parameter_hash_before", "parameter_hash_after"):
        require(q[key] == bank[key], "model identity: " + key)
    require(q["parameter_hash_before"] == q["parameter_hash_after"], "frozen model receipt")
    fields = ("seed", "history_index", "root_rgb_sha256", "goal_rgb_sha256")
    require([row["seed"] for row in q["rows"]] == [20263160, 20263161] and
            len(source["rows"]) == len(bank["rows"]) == len(q["rows"]) == 2, "root roster")
    for a, b, c in zip(q["rows"], source["rows"], bank["rows"], strict=True):
        require(all(a[k] == b[k] == c[k] for k in fields), "root/goal/history receipt mismatch")
        m = a["goal_metric"]
        gram = np.asarray(m["gram"], dtype=np.float64)
        require(gram.shape == (192, 192) and np.isfinite(gram).all() and
                np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12), "goal Gram")
        require(m["physical_future_labels_read"] is False and m["input_vjp_rows"] == 192 and
                len(m["gradient_checks"]) == 6 and
                all(np.isfinite(c["relative_error"]) and np.isfinite(c["cosine"])
                    and c["relative_error"] <= .15 and c["cosine"] >= .98
                    for c in m["gradient_checks"]), "goal Q source and FD receipt")
    return [{k: row[k] for k in fields} for row in q["rows"]]


def material():
    for path, expected in ((Q_PATH, Q_SHA), (SOURCE, SOURCE_SHA), (BANK, BANK_SHA)):
        require(sha(path) == expected, "source hash: " + str(path))
    q, source, bank = (json.loads(p.read_text()) for p in (Q_PATH, SOURCE, BANK))
    roots = validate_metadata(q, source, bank)
    weights = Path(source["protocol"]["policy"]) / "weights.pt"
    stats = ROOT / source["protocol"]["action_stats"]
    require(sha(weights) == q["weights_sha256"] and sha(stats) == q["action_stats_sha256"],
            "checkpoint/action statistics hash")
    return {"schema": NAME + ".v1", "status": "cpu_source_receipts_checked_only",
            "roots": roots, "q_report_sha256": Q_SHA, "root_source_sha256": SOURCE_SHA,
            "bank_report_sha256": BANK_SHA, "weights_path": str(weights),
            "weights_sha256": q["weights_sha256"], "action_stats_path": str(stats),
            "action_stats_sha256": q["action_stats_sha256"],
            "model_parameter_hash_expected": q["parameter_hash_before"],
            "proposal_code_sha256": sha(Path(proposal.__file__)),
            "preflight_code_sha256": sha(Path(__file__)), "gpu_used": False, "physical_outcomes_read": False,
            "checkpoint_loaded": False, "root_reconstructed": False, "runtime_authorized": False,
            "old_new_environment_equivalence": "unknown", "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + "_") and path.suffix == ".json", "exclusive path")
    result = material()
    with path.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "output": str(path), "sha256": sha(path)}))


if __name__ == "__main__":
    main()
