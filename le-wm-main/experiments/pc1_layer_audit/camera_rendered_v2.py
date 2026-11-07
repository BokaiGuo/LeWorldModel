"""Controlled-render branch retaining failed exact-archive preflight as evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc1_layer_audit import camera
from experiments.pc_input_permissions import STATE, file_hash
from experiments.research_runtime.artifacts import atomic_write_json

FAILED = camera.OUT
BASE_PROTOCOL = camera.PROTOCOL
OUT = camera.ROOT / "results/pc1_camera_layer_transfer_rendered_v2_20261107"
PROTOCOL = camera.ROOT.parent / "docs/PC1_CAMERA_LAYER_TRANSFER_RENDERED_V2_PROTOCOL_20261107_CODEX.md"
original_sources = camera.source_hashes


def sources():
    return {**original_sources(), **{str(p): file_hash(p) for p in (
        Path(__file__), BASE_PROTOCOL,
        FAILED / "archive_parity_failure/result.json",
        FAILED / "archive_parity_failure/frames.npz",
        FAILED / "render_repeat/result.json", FAILED / "render_repeat/frames.npz")}}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "prepare.npz").exists():
        raise FileExistsError("prepare output already exists")
    source = camera.DATA / "states.pth"
    archive = torch.load(source, weights_only=True, map_location="cpu")
    states = np.stack([archive[r, [0, 5]].numpy() for r in camera.ROOTS])
    hashes = {str(source): file_hash(source), **sources()}
    for name in ("maze_model.py", "dynamic_mjc.py"):
        p = camera.ROOT / "external/jepa-wms/evals/simu_env_planning/envs/pointmaze_env" / name
        hashes[str(p)] = file_hash(p)
    rendered, seconds, xml_hash = camera.render("frames", states.reshape(-1, 4))
    frames = rendered["images"][:, 0, :, 0].reshape(32, 2, 2, 224, 224, 3)
    rows, refs = [], []
    for i, root in enumerate(camera.ROOTS):
        path = camera.DATA / "obses" / f"episode_{root:03d}.pth"
        ref = torch.load(path, weights_only=True, map_location="cpu").numpy()[[0, 5]]
        refs.append(ref)
        delta = np.abs(frames[i, :, 0].astype(float)-ref.astype(float))
        rows.append(dict(root=root, mae=delta.mean((1, 2, 3)).tolist(),
                         max_error=delta.max((1, 2, 3)).tolist()))
        hashes[str(path)] = file_hash(path)
    record = dict(track=STATE, observation_protocol="controlled_current_renderer",
                  archive_pixel_equivalence=False, scope="development; not official reproduction",
                  roots=camera.ROOTS, calibration_roots=camera.ROOTS[:16], evaluation_roots=camera.ROOTS[16:],
                  cameras=[[3., 3.], [3.5, 3.]], sources=hashes, archive_pixel_errors=rows,
                  xml_sha256=xml_hash, render_seconds=seconds,
                  arrays_sha256=camera.save_arrays("prepare", states=states, frames=frames, reference=np.stack(refs)))
    atomic_write_json(OUT / "prepare.json", record)
    print("controlled-render prepare complete; archive mismatch retained, all 32 roots.", flush=True)


def main(stage):
    camera.OUT, camera.PROTOCOL = OUT, PROTOCOL
    camera.source_hashes, camera.prepare = sources, prepare
    camera.main(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "predict", "replay", "analyze", "all"))
    main(parser.parse_args().stage)
