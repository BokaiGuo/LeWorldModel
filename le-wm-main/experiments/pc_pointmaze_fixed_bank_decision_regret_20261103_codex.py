"""Post-hoc fixed-bank decision diagnostic; no action dispatch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_pointmaze_lookahead_same_action_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FIXED_BANK_DECISION_REGRET_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_pointmaze_fixed_bank_decision_regret_20261103_CODEX.json"
EXPECTED = "d9dde1cc4c341e4a8b2907436266d3b162a803cb75fc49f8b190d75d02d4dc46"
SCORES = ("open_cost", "refresh_cost", "actual_encoded_cost")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(row):
    branches = row["branches"]
    if len(branches) != 9 or sorted(b["index"] for b in branches) != list(range(9)):
        raise ValueError("expected exactly nine indexed branches")
    oracle = min(branches, key=lambda b: (b["physical_distance"], b["index"]))
    chosen = {}
    for key in SCORES:
        branch = min(branches, key=lambda b: (b[key], b["index"]))
        chosen[key] = {"index": branch["index"], "selector_cost": branch[key],
                       "physical_distance": branch["physical_distance"],
                       "physical_regret": branch["physical_distance"] - oracle["physical_distance"]}
    return {"root": row["root"], "oracle_index": oracle["index"],
            "oracle_physical_distance": oracle["physical_distance"], "selectors": chosen}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(SOURCE) != EXPECTED:
        raise ValueError("source hash mismatch")
    source = json.loads(SOURCE.read_text())
    if [row["root"] for row in source["rows"]] != [8343, 8345, 8349, 8351]:
        raise ValueError("root roster mismatch")
    print("Retrospective fixed-bank regret; not a planner or independent experiment.", flush=True)
    report = {"schema": "pc_pointmaze_fixed_bank_decision_regret.v1",
              "source_sha256": sha(SOURCE), "protocol_sha256": sha(PROTOCOL),
              "scope": "retrospective development diagnostic, no outcome used for planning",
              "rows": [analyze(row) for row in source["rows"]]}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
