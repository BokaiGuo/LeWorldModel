"""Post-selection matched-tail interventions on the sealed nullspace plans."""
import json
import sys
from pathlib import Path

from experiments import pc_nullspace_successor_dev_20261103_codex as experiment
from experiments import pc_multidirection_successor_same_chain_diag_20261103_codex as diagnostic

PROTOCOL = experiment.PROTOCOL.parent / "PC_NULLSPACE_SUCCESSOR_MECHANISM_PROTOCOL_20261103_CODEX.md"
OUTPUT = experiment.PREFIX.with_name(experiment.PREFIX.name + "_mechanism.json")
AUDIT = experiment.PREFIX.with_name(experiment.PREFIX.name + "_mechanism_audit.json")


def dependencies():
    files = (Path(__file__), PROTOCOL, experiment.PLANS, experiment.AUDIT, Path(diagnostic.__file__))
    return {**experiment.dependencies(), **diagnostic.dependencies(),
            **{str(p): experiment.source.sha(p) for p in files}}


def calculate():
    data = json.loads(experiment.PLANS.read_text())
    if data["dependencies"] != experiment.dependencies() or json.loads(experiment.AUDIT.read_text())["plans_sha256"] != experiment.source.sha(experiment.PLANS):
        raise ValueError("planning source seal mismatch")
    previous = diagnostic.exp.previous.previous
    old = json.loads(previous.PLANS.read_text())
    if old["dependencies"] != previous.dependencies() or json.loads(previous.AUDIT.read_text())["plans_sha256"] != experiment.source.sha(previous.PLANS):
        raise ValueError("historical control seal mismatch")
    controls = {(r["root"], r["arm"]): r for r in old["rows"] if r["arm"] in ("lookahead", "random_bridge")}
    rows = [diagnostic.analyze(row, controls) for row in data["rows"] if row["arm"] == "null"]
    if len(rows) != 24 or len({row["root"] for row in rows}) != 24:
        raise ValueError("panel mismatch")
    return rows


def main(stage):
    target = OUTPUT if stage == "run" else AUDIT
    if target.exists() or (stage == "run" and AUDIT.exists()):
        raise FileExistsError(target)
    hashes = dependencies()
    if stage == "audit":
        saved = json.loads(OUTPUT.read_text())
        if saved["dependencies"] != hashes:
            raise ValueError("mechanism dependency mismatch")
    print(f"{stage}: full selected-chain interventions.", flush=True)
    rows = calculate()
    if hashes != dependencies():
        raise ValueError("dependency drift")
    if stage == "run":
        payload = {"dependencies": hashes, "rows": rows}
    else:
        if rows != saved["rows"]:
            raise ValueError("deterministic replay mismatch")
        payload = {"mechanism_sha256": experiment.source.sha(OUTPUT), "audit_sha256": experiment.source.sha(Path(__file__)),
                   "replayed_roots": len(rows), "status": "passed"}
    with target.open("x") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    print(f"{stage}: {len(rows)} roots sealed", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    main(sys.argv[1])
