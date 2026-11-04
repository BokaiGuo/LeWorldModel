"""Reuse the sealed same-bank suffix harness on v4's own chosen plans."""
import json
import sys
from pathlib import Path

from experiments import pc_target_reach_v4_e0_20261104_codex as v4
from experiments import pc_v3_same_bank_lookahead_20261104_codex as harness
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as method
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = v4.ROOT
PREFIX = ROOT / "results/pc_target_reach_v4_same_bank_mechanism_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_TARGET_REACH_V4_SAME_BANK_MECHANISM_PROTOCOL_20261104_CODEX.md"
V4_PLANS = v4.PREFIX.with_name(v4.PREFIX.name + "_plans.json")
V4_AUDIT = v4.PREFIX.with_name(v4.PREFIX.name + "_audit.json")


class Seal:
    @staticmethod
    def capture(paths):
        return seal.capture([*paths, Path(__file__), Path(method.__file__)])

    @staticmethod
    def verify(record):
        return seal.verify(record)


def source():
    data = json.loads(V4_PLANS.read_text())
    if json.loads(V4_AUDIT.read_text())["plans_sha256"] != harness.src.prior.base.sha(V4_PLANS):
        raise ValueError("v4 audit binding mismatch")
    return [row for row in data["rows"] if row["arm"] == "response"]


def configure():
    harness.PREFIX, harness.PROTOCOL, harness.source, harness.seal = PREFIX, PROTOCOL, source, Seal
    harness.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    harness.DATA = PREFIX.with_name(PREFIX.name + "_plans.json")
    harness.AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
    harness.src.PLANS, harness.src.AUDIT = V4_PLANS, V4_AUDIT


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    configure()
    {"plan": harness.plan, "audit": harness.audit}[sys.argv[1]]()
