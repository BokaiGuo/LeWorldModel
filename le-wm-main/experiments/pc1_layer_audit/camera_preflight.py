"""Investigate failed archive parity without changing the frozen audit threshold."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.pc1_layer_audit import camera
from experiments.pc_pointmaze_camera_corrected_effect_20261107_codex import render_states
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json


def main():
    out = camera.OUT / "archive_parity_failure"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError(out)
    roots = camera.ROOTS + (1884,)
    archive = torch.load(camera.DATA / "states.pth", weights_only=True, map_location="cpu")
    states = np.stack([archive[r, [0, 5]].numpy() for r in roots])
    rendered, seconds, _ = camera.render("frames", states.reshape(-1, 4))
    frames = rendered["images"][:, 0, 0, 0].reshape(len(roots), 2, 224, 224, 3)
    isolated = render_states(states[[0, -1]].reshape(-1, 4)).reshape(2, 2, 224, 224, 3)
    rows, refs, sources = [], [], camera.source_hashes()
    for i, root in enumerate(roots):
        path = camera.DATA / "obses" / f"episode_{root:03d}.pth"
        refs.append(torch.load(path, weights_only=True, map_location="cpu")[[0, 5]].numpy())
        sources[str(path)] = file_hash(path)
        delta = np.abs(frames[i].astype(float) - refs[-1].astype(float))
        rows.append(dict(root=root, mae=delta.mean((1, 2, 3)).tolist(),
                         max_error=delta.max((1, 2, 3)).tolist(),
                         differing_fraction=(delta > 0).mean((1, 2, 3)).tolist()))
    references = np.stack(refs)
    arrays = out / "frames.npz"
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, roots=roots, frames=frames, reference=references, isolated=isolated)
    montage = np.concatenate([np.concatenate([references[i, 0], frames[i, 0],
        np.clip(np.abs(frames[i, 0].astype(float)-references[i, 0])*6, 0, 255).astype(np.uint8)], 1)
        for i in (0, -1)], 0)
    Image.fromarray(montage).save(out / "comparison.png")
    record = dict(scope="failed preflight; no model/physical outcome selection",
                  original_threshold="exact pixels", sources=sources, rows=rows,
                  isolated_vs_dual_camera_max_error=float(np.max(np.abs(isolated.astype(float)-frames[[0, -1]].astype(float)))),
                  frame_sha256=file_hash(arrays), render_seconds=seconds)
    atomic_write_json(out / "result.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
