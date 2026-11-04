"""Retry latent gap diagnosis with contiguous observed RGB arrays."""
from pathlib import Path

import numpy as np

from experiments import pc_v4_reacher_final_packet_latent_gap_20261104_codex as run


def main():
    original_prepare = run.live.p0._prepare
    original_capture = run.seal.capture

    def contiguous_prepare(processor, frames, root_frame, goal, history, device):
        return original_prepare(processor, frames, root_frame, np.ascontiguousarray(goal), history, device)

    def capture_with_wrapper(extra_paths=()):
        if extra_paths:
            extra_paths = [*extra_paths, Path(__file__)]
        return original_capture(extra_paths)

    run.live.p0._prepare = contiguous_prepare
    run.seal.capture = capture_with_wrapper
    run.PROTOCOL = run.ROOT.parent / "docs/PC_V4_REACHER_FINAL_PACKET_LATENT_GAP_V2_PROTOCOL_20261104_CODEX.md"
    run.PREFIX = run.ROOT / "results/pc_v4_reacher_final_packet_latent_gap_v2_20261104_codex"
    run.MANIFEST = run.PREFIX.with_name(run.PREFIX.name + "_manifest.json")
    run.RESULT = run.PREFIX.with_name(run.PREFIX.name + "_result.json")
    run.ARRAYS = run.PREFIX.with_name(run.PREFIX.name + "_arrays.npz")
    try:
        run.main()
    finally:
        run.live.p0._prepare = original_prepare
        run.seal.capture = original_capture


if __name__ == "__main__":
    main()
