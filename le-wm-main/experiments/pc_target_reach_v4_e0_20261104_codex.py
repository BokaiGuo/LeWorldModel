"""Development v4 on the sealed v3 tail-local exact-dynamics task grid."""
import sys
from pathlib import Path

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as source
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as method
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source.ROOT
PREFIX = ROOT / "results/pc_target_reach_v4_e0_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_TARGET_REACH_V4_E0_DEVELOPMENT_PROTOCOL_20261104_CODEX.md"


def configure():
    prior = source.prior
    prior.PROTOCOL = PROTOCOL
    prior.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    prior.PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
    prior.AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
    prior.INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
    prior.REBUILD = PREFIX.with_name(PREFIX.name + "_intervention_rebuild.json")
    prior.tasks, prior.ARMS, prior.method = source.tasks, ("response",), method
    original = source.original_dependencies

    def dependencies():
        values = original()
        for path in (Path(__file__), PROTOCOL, Path(method.__file__), Path(local.__file__),
                     Path(seal.__file__), source.PLANS, source.AUDIT):
            values[str(path.resolve())] = prior.base.sha(path)
        for path, digest in seal.capture()["files"].items():
            values[path] = digest
        return values

    prior.dependencies = dependencies


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    configure()
    {"plan": source.prior.plan, "audit": source.prior.audit}[sys.argv[1]]()
