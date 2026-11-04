"""Four prefix/five suffix physical factorial on exposed development roots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import ORIGINAL, ORIGINAL_SHA
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA
from experiments.a1_pointmaze_greedy_proprio_replan_20261103 import OUTPUT as GREEDY
from experiments.a1_pointmaze_proprio_spectral_cem_20261103 import OUTPUT as SPECTRAL
from experiments.a1_pointmaze_proprio_spectral_common_suffix_20261103 import OUTPUT as COMMON
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.a1_pointmaze_zero_prefix_proprio_replan_20261103 import OUTPUT as ZERO
from experiments.spectral_reachability import pointmaze_physical_evaluator as physical
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_DATASET, ROOT, sha256_file

SOURCES = (
    (GREEDY, "627316fe3fcaa10e224f22346087a221e46d1f0ac2f1cb77738553551fd85e3c"),
    (SPECTRAL, "b39bc04c3bceb2f54cc982b32d5517f5ae6be10d8f9ede598ac7496ab3ecda71"),
    (COMMON, "0f6ea3d1248eff0f79a433fff30002c5ec1df820e282cab5e8225e280a7515f5"),
    (ZERO, "13d69c90408483271f7db0d426a9411ab42035c71f60955a21a47a04eda35fef"),
    (ORIGINAL, ORIGINAL_SHA), (CAMERA, CAMERA_SHA),
)
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FOUR_PREFIX_COMMON_SUFFIX_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a1_pointmaze_four_prefix_common_suffix_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in SOURCES:
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    greedy, spectral, common, zero, original, camera = (json.loads(path.read_text()) for path, _ in SOURCES)
    for path, doc in ((GREEDY, greedy), (SPECTRAL, spectral), (COMMON, common), (ZERO, zero)):
        if sha256_file(path.with_suffix(".npz")) != doc["arrays_sha256"]:
            raise RuntimeError(f"source arrays changed: {path}")
    if sha256_file(DEFAULT_DATASET / "states.pth") != original["states_sha256"]:
        raise RuntimeError("states changed")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    xml = official_xml()
    if hashlib.sha256(xml.encode()).hexdigest() != camera["xml_sha256"]:
        raise RuntimeError("scene changed")
    for doc in (greedy, spectral, zero):
        if [r["root"] for r in doc["rows"]] != list(range(1888, 1896)):
            raise RuntimeError("root order mismatch")
    starts, sequences, identities = [], [], []
    for i, spec_row in enumerate(spectral["rows"]):
        root, grow, zrow = spec_row["root"], greedy["rows"][i], zero["rows"][i]
        spec = {b["arm"]: b for b in spec_row["branches"]}
        prefix = {"greedy": grow["prefix"][0], "zero": [0., 0.],
                  "random": spec["random"]["initial_plan"][0], "spectral": spec["spectral"]["initial_plan"][0]}
        suffix = {"zero_action": [0., 0.], "greedy": grow["tail"][0], "zero_replanned": zrow["tail"][0],
                  "random": spec["random"]["tail"][0], "spectral": spec["spectral"]["tail"][0]}
        for p in ("greedy", "zero", "random", "spectral"):
            for s in ("zero_action", "greedy", "zero_replanned", "random", "spectral"):
                seq = np.concatenate((np.repeat(np.asarray(prefix[p], np.float32)[None], 5, axis=0),
                                      np.repeat(np.asarray(suffix[s], np.float32)[None], 5, axis=0)))
                if not np.isfinite(seq).all() or np.any(np.abs(seq) > 1):
                    raise ValueError("invalid actions")
                starts.append(states[root, 0].astype(np.float64))
                sequences.append(seq)
                identities.append((root, p, s))
    starts, sequences = np.asarray(starts), np.asarray(sequences)
    if sequences.shape != (160, 10, 2):
        raise RuntimeError("invalid design size")
    started = time.monotonic()
    print("Executing 160 fixed four-prefix, five-suffix physical combinations.", flush=True)
    rgb, legacy = replay_legacy(starts, sequences, xml)
    cp5 = physical.replay_position_endpoints(starts, sequences[:, :5])
    cp10 = physical.replay_position_endpoints(starts, sequences)
    if not np.array_equal(legacy[:, :2], cp10[:, :2]):
        raise RuntimeError("terminal position mismatch")
    prior_index = {(r["root"], r["prefix"], r["suffix"]): i for i, r in enumerate(common["rows"])}
    identity_index = {identity: i for i, identity in enumerate(identities)}
    with np.load(COMMON.with_suffix(".npz"), allow_pickle=False) as c, \
            np.load(GREEDY.with_suffix(".npz"), allow_pickle=False) as g, \
            np.load(ZERO.with_suffix(".npz"), allow_pickle=False) as z:
        for i, (root, prefix, suffix) in enumerate(identities):
            old_suffix = "zero" if suffix == "zero_action" else suffix
            if prefix in ("zero", "random", "spectral") and old_suffix in ("zero", "random", "spectral"):
                k = prior_index[root, prefix, old_suffix]
                np.testing.assert_array_equal(sequences[i], c["sequences"][k])
                np.testing.assert_array_equal(rgb[i], c["rgb10"][k])
                np.testing.assert_array_equal(legacy[i], c["legacy"][k])
                np.testing.assert_array_equal(cp5[i], c["cp5"][k])
            if (prefix, suffix) == ("greedy", "greedy"):
                np.testing.assert_array_equal(rgb[i], g[f"r{root}_rgb10"])
                np.testing.assert_array_equal(legacy[i], g["endpoints_eval_only"][root-1888])
            if (prefix, suffix) == ("zero", "zero_replanned"):
                np.testing.assert_array_equal(rgb[i], z[f"r{root}_rgb10"])
                np.testing.assert_array_equal(legacy[i], z["endpoints_eval_only"][root-1888])
    rows = []
    for i, (root, prefix, suffix) in enumerate(identities):
        goal = states[root, 10, :2].astype(np.float64)
        control = cp10[identity_index[root, prefix, "zero_action"], :2]
        distance = float(np.linalg.norm(cp10[i, :2]-goal))
        rows.append({"root": root, "prefix": prefix, "suffix": suffix,
                     "goal_position_eval_only": goal.tolist(), "initial_position": starts[i, :2].tolist(),
                     "cp5_state": cp5[i].tolist(), "cp10_state": cp10[i].tolist(),
                     "initial_distance": float(np.linalg.norm(starts[i, :2]-goal)),
                     "cp5_distance": float(np.linalg.norm(cp5[i, :2]-goal)),
                     "cp10_distance": distance, "strict_endpoint_success_eval_only": bool(distance < .5),
                     "suffix_displacement_minus_zero": (cp10[i, :2]-control).tolist(),
                     "suffix_distance_change_minus_zero": distance-float(np.linalg.norm(control-goal))})
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, starts=starts, sequences=sequences, rgb10=rgb,
                            legacy=legacy, cp5=cp5, cp10=cp10)
    report = {"schema": "a1_pointmaze_four_prefix_common_suffix.v1",
              "source_hashes": {str(p.relative_to(ROOT)): h for p, h in SOURCES},
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "physical_sequences": len(sequences),
              "old_panel_exact": 72, "greedy_and_zero_original_exact": 16,
              "cp10_backend_position_exact": 160, "wall_seconds": time.monotonic()-started, "rows": rows,
              "scope": "exposed development roots; fixed common suffix; no P1"}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
