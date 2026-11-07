"""Run the unchanged finite audit with same-runtime synchronized render snapshots."""
import argparse
from pathlib import Path

from experiments.pc1_layer_audit import camera, camera_rendered_v2 as controlled
from experiments.pc_input_permissions import file_hash

OUT = camera.ROOT / "results/pc1_camera_layer_transfer_runtime_bound_20261107"
PROTOCOL = camera.ROOT.parent / "docs/PC1_CAMERA_RUNTIME_BOUND_PROTOCOL_20261107_CODEX.md"
original_sources = controlled.sources


def sources():
    return {**original_sources(), **{str(p): file_hash(p) for p in (
        Path(__file__), PROTOCOL,
        camera.ROOT.parent / "docs/PC1_CAMERA_LAYER_TRANSFER_RENDERED_V2_PROTOCOL_20261107_CODEX.md")}}


def main(stage):
    controlled.OUT, controlled.PROTOCOL = OUT, PROTOCOL
    controlled.sources = sources
    camera.WORKER = Path(__file__).with_name("camera_synced_worker.py")
    controlled.main(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "predict", "replay", "analyze", "all"))
    main(parser.parse_args().stage)
