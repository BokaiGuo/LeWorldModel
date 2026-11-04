"""Descriptive post-hoc rank comparison; no planner or candidate selection."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_joint_carry_all_roots_20261103_DEV01.json"
EXPECTED = "f5977bd71e636a316935ecfce1045968adbcf6c70b100de5c825bfc20f2fe849"


def analyze(report):
    rows = report["rows"]
    out = []
    for root in (8343, 8345, 8349, 8351):
        branches = [r for r in rows if r["root"] == root and r["mode"] == "carry_init"]
        if len(branches) not in (4, 5):
            raise ValueError("unexpected root roster")
        pairs = list(itertools.combinations(branches, 2))
        eligible = [(a, b) for a, b in pairs
                    if a["refined_score"] != b["refined_score"] and a["cp17_distance"] != b["cp17_distance"]]
        predicted = min(branches, key=lambda r: r["refined_score"])
        actual = min(branches, key=lambda r: r["cp17_distance"])
        out.append({"root": root, "branches": [{"role": b["role"], "predicted": b["refined_score"],
                                                 "physical_distance": b["cp17_distance"]} for b in branches],
                    "pairs": len(pairs), "comparable": len(eligible),
                    "reversed": sum((a["refined_score"] - b["refined_score"])
                                    * (a["cp17_distance"] - b["cp17_distance"]) < 0 for a, b in eligible),
                    "posthoc_predicted_best": predicted["role"], "posthoc_actual_best": actual["role"]})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(SOURCE) != EXPECTED:
        raise ValueError("source hash mismatch")
    print("Describe post-hoc rankings of existing cp7-conditioned outputs; no action selection.")
    output = {"schema": "pc_joint_carry_posthoc_rank.v1", "source_sha256": sha(SOURCE),
              "script_sha256": sha(Path(__file__)), "rows": analyze(json.loads(SOURCE.read_text())),
              "scope": "posthoc exposed dependent physical counterfactuals; not deployable selection"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
