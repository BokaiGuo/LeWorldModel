"""Equal-feedback optimizer on sealed multidirection mechanism prefixes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments import pc_fixed_prefix_equal_feedback_optimizer_20261103_codex as base
from experiments import pc_multidirection_successor_same_chain_diag_20261103_codex as source

PROTOCOL = base.ROOT.parent / "docs/PC_MULTIDIRECTION_EQUAL_FEEDBACK_MECHANISM_DIAG_PROTOCOL_20261103_CODEX.md"
PREFIX = base.ROOT / "results/pc_multidirection_equal_feedback_mechanism_diag_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")


def dependencies():
    paths = (Path(__file__), PROTOCOL, Path(base.__file__), source.OUTPUT)
    return {**source.dependencies(), **{str(p): base.sha(p) for p in paths}}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("sealed output exists")
    hashes = dependencies()
    data = json.loads(source.OUTPUT.read_text())
    if data["dependencies"] != source.dependencies():
        raise ValueError("source seal mismatch")
    print("Equal-feedback replanning of sealed multidirection prefixes; retrospective diagnostic.", flush=True)
    rows = [base.run_one(row, name) for row in data["rows"] for name in row["controls"]]
    if len(rows) != 120 or len({(r["root"], r["name"]) for r in rows}) != 120 or hashes != dependencies():
        raise ValueError("panel or source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": hashes, "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"Sealed {len(rows)} prefix feedback runs.", flush=True)


def audit():
    data, original = json.loads(PLANS.read_text()), json.loads(source.OUTPUT.read_text())
    hashes = dependencies()
    if data["dependencies"] != hashes or original["dependencies"] != source.dependencies():
        raise ValueError("source seal mismatch")
    old = base.PLANS, base.AUDIT, base.dependencies, base.source
    proxy = SimpleNamespace(OUTPUT=source.OUTPUT, dependencies=source.dependencies, feedback=base.source.feedback)
    try:
        base.PLANS, base.AUDIT, base.dependencies, base.source = PLANS, AUDIT, lambda: hashes, proxy
        base.audit()
    finally:
        base.PLANS, base.AUDIT, base.dependencies, base.source = old


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
