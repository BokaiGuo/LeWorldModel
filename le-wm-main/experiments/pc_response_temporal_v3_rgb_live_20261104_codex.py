"""Bind the existing live RGB runner and its audit to the frozen v3 compiler."""
import json
import sys
from pathlib import Path

from experiments import pc_response_temporal_rgb_live_diag_20261104_codex as live
from experiments import pc_response_temporal_rgb_live_audit_20261104_codex as verifier
from experiments import pc_response_temporal_v3_fixed_terminal_e0_20261104_codex as e0
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as method

PROTOCOL = live.ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_RGB_LIVE_DIAG_PROTOCOL_20261104_CODEX.md"
PREFIX = live.ROOT / "results/pc_response_temporal_v3_rgb_live_diag_20261104_codex"
SCRIPT = Path(__file__).resolve()
original_dependencies = live.dependencies


def dependencies():
    result = original_dependencies()
    for path in (SCRIPT, Path(method.__file__), e0.PLANS, e0.AUDIT):
        result[str(path.resolve())] = live.old.prior.sha(path)
    if json.loads(e0.AUDIT.read_text())["plans_sha256"] != live.old.prior.sha(e0.PLANS):
        raise ValueError("v3 exact panel seal mismatch")
    return result


def configure():
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    live.feedback, live.dependencies = method, dependencies
    verifier.AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
    verifier.PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    configure()
    {"run": live.main, "audit": verifier.main}[sys.argv[1]]()
