"""Signature adapter for running unmodified V5 on a saved Reacher RGB context."""
import sys
from pathlib import Path

from experiments import pc_budget_allocation_20261103_codex as allocation
from experiments import pc_equal_resource_dev_20261103_codex as meter
from experiments import pc_v4_reacher_saved_context_20261104_codex as saved
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_canonical_v5 as canonical
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as response
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = saved.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_REACHER_SAVED_CONTEXT_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_reacher_saved_context_20261104_codex"
METHOD = canonical.METHOD


def decide(arm, root, step, target, predict, resource, carry=None):
    return canonical.decide("response", arm, root, step, target, predict, resource, carry)


class SealAdapter:
    @staticmethod
    def capture(files):
        return seal.capture([*files, Path(canonical.__file__), Path(response.__file__),
                             Path(core.__file__), Path(allocation.__file__), Path(meter.__file__)])

    verify = staticmethod(seal.verify)


def configure():
    saved.PROTOCOL, saved.PREFIX = PROTOCOL, PREFIX
    saved.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    saved.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    saved.method = sys.modules[__name__]
    saved.seal = SealAdapter


if __name__ == "__main__":
    configure()
    saved.main()
