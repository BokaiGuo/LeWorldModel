"""Matched-tail mechanism diagnostics on sealed goal-aligned final chains."""
import json
from pathlib import Path

from experiments import pc_goal_aligned_successor_dev_20261103_codex as new
from experiments import pc_multidirection_successor_same_chain_diag_20261103_codex as diagnostic

PROTOCOL = new.PROTOCOL.parent / "PC_GOAL_ALIGNED_SUCCESSOR_MECHANISM_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = new.PREFIX.with_name(new.PREFIX.name + "_mechanism_diag.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, new.PLANS, new.AUDIT, Path(diagnostic.__file__))
    return {**new.dependencies(), **diagnostic.dependencies(), **{str(p): new.sha(p) for p in paths}}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    hashes = dependencies()
    data = json.loads(new.PLANS.read_text())
    if data["dependencies"] != new.dependencies() or json.loads(new.AUDIT.read_text())["plans_sha256"] != new.sha(new.PLANS):
        raise ValueError("new plan seal mismatch")
    previous = diagnostic.exp.previous.previous
    old = json.loads(previous.PLANS.read_text())
    if old["dependencies"] != previous.dependencies() or json.loads(previous.AUDIT.read_text())["plans_sha256"] != new.sha(previous.PLANS):
        raise ValueError("historical control seal mismatch")
    controls = {(r["root"], "lookahead"): r for r in old["rows"] if r["arm"] == "lookahead"}
    controls.update({(r["root"], "random_bridge"): r for r in data["rows"] if r["arm"] == "random"})
    print("Matched-tail mechanism interventions on sealed goal-aligned selected chains.", flush=True)
    rows = [diagnostic.analyze(row, controls) for row in data["rows"] if row["arm"] == "goal"]
    if len(rows) != 24 or len({r["root"] for r in rows}) != 24 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print("Sealed 24 same-chain development diagnostics.", flush=True)


if __name__ == "__main__":
    main()
