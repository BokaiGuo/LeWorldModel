"""Run the two-start baseline through the shared sealed E0 feedback harness."""
import json
import sys
from pathlib import Path

from experiments import pc_v3_carry_aware_cem_e0_20261104_codex as harness
from experiments.spectral_reachability import pc_response_temporal_two_start_cem_v1 as method
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = harness.ROOT
PREFIX = ROOT / "results/pc_v3_two_start_cem_e0_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_V3_TWO_START_CEM_E0_PROTOCOL_20261104_CODEX.md"


class Seal:
    __file__ = seal.__file__

    @staticmethod
    def capture(paths):
        return seal.capture([*paths, Path(__file__), Path(method.__file__)])

    @staticmethod
    def verify(record):
        return seal.verify(record)


def configure():
    harness.PREFIX, harness.PROTOCOL = PREFIX, PROTOCOL
    harness.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    harness.PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
    harness.AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
    harness.warm, harness.seal = method, Seal


def audit():
    harness.audit()
    plans = json.loads(harness.PLANS.read_text())
    for row in plans["rows"]:
        for decision in row["decisions"][1:]:
            restarts = decision["result"]["restarts"]
            if ([r["name"] for r in restarts] != ["zero", "carry"]
                    or abs(restarts[0]["samples"] - restarts[1]["samples"]) > 1
                    or sum(r["samples"] for r in restarts) != sum(
                        sum(g["size"] for g in r["generations"]) for r in restarts)):
                raise ValueError("two-start allocation mismatch")
    output = PREFIX.with_name(PREFIX.name + "_allocation_audit.json")
    harness.old.prior.write(output, {"plans_sha256": harness.old.prior.sha(harness.PLANS),
                                    "audit_sha256": harness.old.prior.sha(harness.AUDIT),
                                    "verified_decisions": 48})
    print("Audited both restart allocations.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    configure()
    {"plan": harness.plan, "audit": audit}[sys.argv[1]]()
