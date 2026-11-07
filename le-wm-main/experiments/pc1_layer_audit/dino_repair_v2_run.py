"""Preserve interpolation-only failure; certify source QKV without new thresholds."""
from pathlib import Path

from experiments.pc1_layer_audit import dino_repair_run as base
from experiments.pc1_layer_audit import dino_source_compatible_v2 as repair

OUT = base.ROOT / "results/pc1_dino_source_compatible_v2_20261107"
PROTOCOL = base.ROOT.parent / "docs/PC1_DINO_SOURCE_QKV_REPAIR_ADDENDUM_20261107_CODEX.md"
original_sources = base.sources
original_protocol = base.PROTOCOL


def sources(paths):
    return original_sources([*paths, Path(__file__), original_protocol, PROTOCOL,
                             Path(base.__file__).with_name("dino_source_compatible.py")])


def main():
    base.OUT, base.PROTOCOL, base.repair, base.sources = OUT, PROTOCOL, repair, sources
    base.main()


if __name__ == "__main__":
    main()
