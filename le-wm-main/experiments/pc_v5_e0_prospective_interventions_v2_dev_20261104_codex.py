"""Single-process adapter correcting selected candidate-index comparison."""
import sys
from pathlib import Path

from experiments import pc_v5_e0_prospective_interventions_dev_20261104_codex as original

ROOT = original.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_E0_PROSPECTIVE_INTERVENTION_V2_PROTOCOL_20261104_CODEX.md"
PREFIX = original.source.PREFIX.with_name(original.source.PREFIX.name + "_intervention_v2")
OLD_MANIFEST = original.MANIFEST


def main(mode):
    old_local, old_seal = original.local, original.seal
    old_paths = original.PREFIX, original.MANIFEST, original.RESULT, original.AUDIT

    class LocalAdapter:
        __file__ = old_local.__file__

        @staticmethod
        def build(row):
            result = old_local.build(row)
            result["controls"]["selected"]["candidate_index"] = -1
            return result

    class SealAdapter:
        @staticmethod
        def capture(paths):
            return old_seal.capture([*paths, Path(__file__), PROTOCOL, OLD_MANIFEST])

        @staticmethod
        def verify(record):
            return old_seal.verify(record)

    try:
        original.local, original.seal = LocalAdapter, SealAdapter
        original.PREFIX = PREFIX
        original.MANIFEST, original.RESULT, original.AUDIT = [
            PREFIX.with_name(PREFIX.name + part + ".json") for part in ("_manifest", "_result", "_audit")]
        original.main(mode)
    finally:
        original.local, original.seal = old_local, old_seal
        original.PREFIX, original.MANIFEST, original.RESULT, original.AUDIT = old_paths


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("intervene", "audit"):
        raise SystemExit("choose intervene or audit")
    main(sys.argv[1])
