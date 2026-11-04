"""Same-bank diagnostic of V5 response-selected chains."""
import json
import sys
from pathlib import Path

from experiments import pc_v5_canonical_retention_e0_20261104_codex as v5
from experiments import pc_v3_same_bank_lookahead_20261104_codex as harness
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as method
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = v5.ROOT
PREFIX = ROOT / "results/pc_v5_canonical_response_same_bank_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V5_CANONICAL_RESPONSE_SAME_BANK_MECHANISM_PROTOCOL_20261104_CODEX.md"


class Seal:
    @staticmethod
    def capture(paths):
        return seal.capture([*paths, Path(__file__), Path(method.__file__), v5.MANIFEST])

    @staticmethod
    def verify(record):
        return seal.verify(record)


def source():
    data = json.loads(v5.PLANS.read_text())
    if json.loads(v5.AUDIT.read_text())["plans_sha256"] != harness.src.prior.base.sha(v5.PLANS):
        raise ValueError("v5 source audit mismatch")
    return [row for row in data["rows"] if row["strategy"] == "response"]


def configure():
    harness.PREFIX, harness.PROTOCOL, harness.source, harness.seal = PREFIX, PROTOCOL, source, Seal
    harness.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    harness.DATA = PREFIX.with_name(PREFIX.name + "_plans.json")
    harness.AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
    harness.src.PLANS, harness.src.AUDIT = v5.PLANS, v5.AUDIT


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    configure()
    {"plan": harness.plan, "audit": harness.audit}[sys.argv[1]]()
