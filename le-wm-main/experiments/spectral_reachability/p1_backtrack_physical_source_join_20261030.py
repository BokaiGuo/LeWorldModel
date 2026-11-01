"""Join sealed P1 model actions to native source identities; no physical dispatch."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as physical
from experiments.spectral_reachability import p1_backtrack_cuda_replay_consume_20261030 as replay_consume

NAME = "p1_backtrack_physical_source_join_20261030"
replay = replay_consume.replay
ROOT = replay.ROOT
REPLAY_PATH = ROOT / "results/p1_backtrack_cuda_replay_20261030_MAIN01.json"
REPLAY_SHA = "83ea957ecb38025e6b5489a0bf843b112906b0954b8b6c62238c10461b31cfa4"
MANIFEST_PATH = ROOT / "results/p1_backtrack_cuda_replay_20261030_prepare_MAIN01.json"
MANIFEST_SHA = "f245d30ae0b28591a9c7308817951b2a38f7c89b81f24d6f6262434f6044310e"
ORDER = [(0, "off"), (0, "on"), (1, "off"), (1, "on")]


def assemble(action, replay_report, prepared, data):
    if len(action["rows"]) != 16 or len(replay_report["rows"]) != 16 or len(prepared["requests"]) != 4:
        raise ValueError("fixed source roster")
    metadata = physical.root_metadata(data)
    task_rows = []
    for slot, (root, condition) in enumerate(ORDER):
        req = prepared["requests"][slot]
        task = req["task"]
        if (req["slot"], req["root_slot"], req["condition"]) != (slot, root, condition):
            raise ValueError("task identity")
        if task["binding"]["blind_sha256"] != data["exports"][(root, condition)]["sha256"]:
            raise ValueError("blind authority mismatch")
        packet = data["roots"][root]["packets"][condition]
        raw = np.asarray(packet["raw_action_history"], np.float32)
        mean = np.tile(np.asarray(task["action_mean"], np.float32), 5)
        std = np.tile(np.asarray(task["action_std"], np.float32), 5)
        history = np.asarray(task["root_context"]["action_history"], np.float32)
        if (raw.shape != (10, 2) or mean.shape != (10,) or std.shape != (10,)
                or history.shape != (2, 10) or not np.isfinite(raw).all()
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("invalid warmup or normalizer")
        normalized = (raw.reshape(2, 10) - mean) / std
        if not np.array_equal(normalized, history):
            raise ValueError("warmup action history mismatch")
        task_rows.append(dict(slot=slot, root_slot=root, condition=condition,
                              task_id=task["binding"]["task_id"],
                              model_root_sha256=task["binding"]["root_sha256"],
                              native_root_sha256=metadata[root]["root_sha256"],
                              blind_sha256=task["binding"]["blind_sha256"],
                              goal_pixel_sha256=metadata[root]["goal_pixels"][condition],
                              native_goal=deepcopy(metadata[root]["goal"]),
                              native_root_state=deepcopy(metadata[root]["root_state"]),
                              renderer=deepcopy(metadata[root]["renderer"]),
                              warmup_raw_float32_hex=raw.astype("<f4").tobytes().hex(),
                              warmup_model_history_exact_numpy=True))
    rows = []
    for i, (entry, observed) in enumerate(zip(action["rows"], replay_report["rows"])):
        slot = i // 4
        if (entry["slot"] != slot or observed["slot"] != slot
                or any(entry[k] != observed[k] for k in ("arm", "role", "candidate_index"))
                or not all(observed["exact"].values())):
            raise ValueError("action and model replay identity mismatch")
        raw15 = np.asarray(entry["raw15"], dtype="<f4")
        if (raw15.shape != (15, 2) or not np.isfinite(raw15).all()
                or np.max(np.abs(raw15)) > 1 or raw15.tobytes().hex() != entry["raw15_le_float32_hex"]):
            raise ValueError("raw action mismatch")
        rows.append(dict(slot=slot, arm=entry["arm"], role=entry["role"],
                         candidate_index=entry["candidate_index"],
                         raw15_float32_hex=entry["raw15_le_float32_hex"],
                         physical_root_sha256=task_rows[slot]["native_root_sha256"],
                         task_id=task_rows[slot]["task_id"]))
    return dict(schema=NAME, fixed_slots=4, fixed_roles=16, known_roots=2,
                task_rows=task_rows, rows=rows, source_replay_sha256=REPLAY_SHA,
                source_plan=deepcopy(data["source_plan"]),
                source_slots=[deepcopy(root["slot"]) for root in data["roots"]],
                warm_traces=[deepcopy(root["warm_trace"]) for root in data["roots"]],
                source_bindings=deepcopy(data["bindings"]),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numpy_warm_history_verified=True, new_model_calls=0, new_physics_calls=0,
                physical_authorized=False, P1_admission=False, freshness=False,
                limitation="Saved-source join only; different root hash schemes not equated; no live reset, render, dispatch, or GPU normalization attested.")


def build():
    result = replay.one.saved.read_verified(REPLAY_PATH, REPLAY_SHA)
    manifest = replay.one.saved.read_verified(MANIFEST_PATH, MANIFEST_SHA)
    if result["manifest"] != str(MANIFEST_PATH) or result["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("replay manifest mismatch")
    action, original, report, prepared = replay.sources()
    checked = replay_consume.verify(result, manifest, action, original, report, prepared)
    if checked["all_exact"] is not True:
        raise ValueError("model replay incomplete")
    data = physical.source_material()
    joined = assemble(action, result, prepared, data)
    for path, expected in data["bindings"].items():
        physical.require(physical.sha(path) == expected, "source drift")
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if (not path.is_absolute() or path.parent != ROOT / "results" or path.resolve() != path
            or not path.name.startswith(NAME + "_") or path.exists()):
        raise ValueError("exclusive output path")
    joined = build()
    print(json.dumps(dict(path=str(path), sha256=replay.one.saved.write(path, joined),
                          fixed_slots=4, fixed_roles=16, physical_authorized=False)))


if __name__ == "__main__":
    main()
