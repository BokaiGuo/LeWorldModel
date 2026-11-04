"""Apply the existing matched-tail mechanism diagnostic to the new sealed policy."""
import json
from pathlib import Path

from experiments import pc_multidirection_multistart_budget_dev_20261103_codex as new
from experiments import pc_multidirection_successor_same_chain_diag_20261103_codex as diagnostic

PROTOCOL = new.PROTOCOL.parent / "PC_MULTIDIRECTION_MULTISTART_BUDGET_MECHANISM_DIAG_PROTOCOL_20261103_CODEX.md"
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
        raise ValueError("new method seal mismatch")
    previous = diagnostic.exp.previous.previous
    old = json.loads(previous.PLANS.read_text())
    if old["dependencies"] != previous.dependencies() or json.loads(previous.AUDIT.read_text())["plans_sha256"] != new.sha(previous.PLANS):
        raise ValueError("historical control seal mismatch")
    controls = {(r["root"], r["arm"]): r for r in old["rows"]}
    print("Matched-tail and response interventions on all sealed new-method decisions.", flush=True)
    rows = []
    for row in data["rows"]:
        if row["arm"] != "multistart":
            continue
        result = diagnostic.analyze(row, controls)
        result["cap"] = row["cap"]
        rows.append(result)
    if len(rows) != 48 or len({(r["root"], r["cap"]) for r in rows}) != 48 or hashes != dependencies():
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print("Sealed 48 same-chain development diagnostics.", flush=True)


if __name__ == "__main__":
    main()
