"""Replay every saved PointMaze action and compare each RGB frame."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v5_pointmaze_rgb_live_v2_dev_20261104_codex as configured
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, run = configured.ROOT, configured.live
PROTOCOL = ROOT.parent / "docs/PC_V5_POINTMAZE_RGB_FRAME_REPLAY_PROTOCOL_20261104_CODEX.md"
OUT = configured.PREFIX.with_name(configured.PREFIX.name + "_rgb_frame_replay.json")


def main():
    configured.configure()
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.old.prior.sha
    manifest = json.loads(run.MANIFEST.read_text())
    if manifest["dependencies"] != run.dependencies():
        raise ValueError("manifest mismatch")
    decisions, images, execution = run.paths("response")
    receipt = json.loads(execution.read_text())
    if receipt["images_sha256"] != sha(images) or receipt["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("receipt mismatch")
    deps = seal.capture([Path(__file__), PROTOCOL, run.MANIFEST, images, execution, *decisions,
                         *[Path(p) for p in manifest["dependencies"]]])
    actions = np.asarray(run.old.WARMUP.tolist() + receipt["executed_actions"], dtype=np.float64)
    if actions.shape != (30, 2) or np.any(np.abs(actions) > 1):
        raise ValueError("invalid action sequence")
    with np.load(images, allow_pickle=False) as archive:
        expected = archive["rgb"]
        if expected.shape != (31, 224, 224, 3):
            raise ValueError("invalid image history")
        with run.bridge.runner.ShadowRGBBridge(np.asarray([*run.old.STARTS[0], 0., 0.])) as env:
            errors = [int(np.max(np.abs(env.initial_rgb.astype(np.int16)-expected[0].astype(np.int16))))]
            for index, action in enumerate(actions):
                image = env.step(action[None])
                errors.append(int(np.max(np.abs(image.astype(np.int16)-expected[index+1].astype(np.int16)))))
    seal.verify(deps)
    run.old.write_json(OUT, {"dependencies": deps, "execution_sha256": sha(execution),
                             "per_frame_max_abs": errors, "exact_frames": sum(v == 0 for v in errors),
                             "scope": "same shadow RGB backend, not independent physical state"})
    if any(errors):
        raise ValueError("RGB frame replay mismatch; failure record saved")
    print("Replayed and matched all 31 RGB frames.", flush=True)


if __name__ == "__main__":
    main()
