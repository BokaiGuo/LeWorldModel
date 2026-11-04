"""Run the position-retention ablation with the sealed v4 E0 harness."""
import sys
from pathlib import Path

from experiments import pc_target_reach_v4_e0_20261104_codex as baseline
from experiments.spectral_reachability import pc_target_reach_v4_position_retention_ablation as method

ROOT = baseline.ROOT
PREFIX = ROOT / "results/pc_target_reach_v4_position_retention_e0_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_TARGET_REACH_V4_POSITION_RETENTION_ABLATION_PROTOCOL_20261104_CODEX.md"


def configure():
    baseline.configure()
    prior = baseline.source.prior
    prior.PROTOCOL = PROTOCOL
    prior.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    prior.PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
    prior.AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
    prior.INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
    prior.REBUILD = PREFIX.with_name(PREFIX.name + "_intervention_rebuild.json")
    prior.method = method
    original = prior.dependencies

    def dependencies():
        values = original()
        for path in (Path(__file__), PROTOCOL, Path(method.__file__)):
            values[str(path.resolve())] = prior.base.sha(path)
        return values

    prior.dependencies = dependencies


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    configure()
    {"plan": baseline.source.prior.plan, "audit": baseline.source.prior.audit}[sys.argv[1]]()
