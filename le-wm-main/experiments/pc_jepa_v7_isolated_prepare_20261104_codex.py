"""Privileged preparation: export only predeclared planner inputs."""
import hashlib
import json
from pathlib import Path

import decord
import numpy as np

from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
BASE = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V7_ISOLATED_INPUT_DEV_PROTOCOL_20261104_CODEX.md"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if BASE.exists():
        raise FileExistsError(BASE)
    manifest = json.loads(MANIFEST.read_text())
    native.verify_manifest(ROOT, manifest)
    _, v5, _ = native.helper(ROOT).dependencies(ROOT)
    v4 = v5.import_v4(ROOT)
    arrays = v5.read_arrays(v4, ROOT)
    stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
    values, rgb, _ = v5.row_inputs(v4, ROOT, arrays, stats, native.EPISODE)
    saved = manifest["row"]
    if v4.array_sha(rgb) != saved["rgb_sha256"] or any(
        v4.array_sha(value) != saved["inputs"][key]["sha256"] for key, value in values.items()
    ):
        raise ValueError("sealed context mismatch")
    video = v4.data_root(ROOT) / "obses/episode_001.mp4"
    reader = decord.VideoReader(str(video), num_threads=1, ctx=decord.cpu(0))
    goal = reader.get_batch([25]).asnumpy()
    inputs = {
        "rgb_history": rgb[:3].copy(),
        "goal_rgb": goal.copy(),
        "normalized_proprio": values["normalized_proprio"][:3].copy(),
        "normalized_history_packets": values["packets"][:2].copy(),
        "action_mean": np.asarray(stats["ACTION_MEAN"], dtype=np.float32),
        "action_std": np.asarray(stats["ACTION_STD"], dtype=np.float32),
    }
    expected = {"rgb_history": (3, 224, 224, 3), "goal_rgb": (1, 224, 224, 3),
                "normalized_proprio": (3, 4), "normalized_history_packets": (2, 10),
                "action_mean": (2,), "action_std": (2,)}
    if any(inputs[key].shape != shape or not np.isfinite(inputs[key]).all() for key, shape in expected.items()):
        raise ValueError("input shape or finiteness failed")
    allowed_meta = ("official_weight_mapping", "expected_dynamics_schema", "expected_model_receipts")
    metadata = {key: manifest[key] for key in allowed_meta}
    metadata["seals"] = {path: digest for path, digest in manifest["seals"].items() if path.endswith(".py")}
    BASE.mkdir()
    (BASE / "input").mkdir()
    (BASE / "output").mkdir()
    with (BASE / "input/context.npz").open("xb") as stream:
        np.savez_compressed(stream, **inputs)
    with (BASE / "input/model_metadata.json").open("x") as stream:
        json.dump(metadata, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    record = {"scope": "preparation process has future-data access; output contains listed fields only",
        "protocol_sha256": sha(PROTOCOL), "source_manifest_sha256": sha(MANIFEST),
        "source_video_sha256": sha(video), "input_sha256": sha(BASE / "input/context.npz"),
        "model_metadata_sha256": sha(BASE / "input/model_metadata.json"),
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "array_sha256": v4.array_sha(value)} for key, value in inputs.items()},
        "no_future_actions_or_state_exported": True}
    with (BASE / "input/preparation.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
