"""Frozen cp12 single/three-frame same-action prediction diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

ROOT = h.ROOT
SOURCE = ROOT / "results/pc_causal_mixed_rejected_physical_20261103_DEV01.json"
REFRESH = ROOT / "results/pc_causal_mixed_cp12_refresh_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP12_THREE_FRAME_SAME_ACTION_PROTOCOL_20261103_CODEX.md"
OUT = ROOT / "results/pc_pointmaze_cp12_three_frame_same_action_20261103_CODEX.json"
NPZ = OUT.with_suffix(".npz")
EXPECTED = {
    SOURCE: "07ada31f9a58c3472fd84778f749c2df6cfb14db6442531954e5e7003ab587f8",
    REFRESH: "a2a967567427e2cf0d94a222547697505d7e595adea7a932b0f7ce9ad5bea1a3",
    h.CAMERA: h.CAMERA_SHA,
    h.OLD: h.OLD_SHA,
    h.DEFAULT_CONFIG: "c666b4251f72f56c08c69ab223cfe645625f39705f1c84285320edc98c205a66",
}


def packets(actions, mean, std):
    """Use exactly outgoing command slices for observation times 2, 7 and 12."""
    actions = np.asarray(actions)
    if actions.shape != (17, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("invalid raw action trajectory")
    if mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid normalizer")
    x = torch.as_tensor(actions[2:17], dtype=mean.dtype, device=mean.device)
    return ((x - mean) / std).reshape(3, 1, 10)


def mse(a, b):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid prediction or target")
    return float((a - b).square().mean().item())


def main():
    if OUT.exists() or NPZ.exists():
        raise FileExistsError("refusing overwrite")
    for path, digest in EXPECTED.items():
        if h.sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    source, refresh, camera, old = (json.loads(p.read_text()) for p in (SOURCE, REFRESH, h.CAMERA, h.OLD))
    for path, digest in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                         (REFRESH.with_suffix(".npz"), refresh["arrays_sha256"]),
                         (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    if [r["root"] for r in source["rows"]] != [8343, 8345, 8349, 8351] or len(refresh["rows"]) != 4:
        raise ValueError("unexpected roster")
    archive = ROOT / "results/a0_pointmaze_archive_bank_alignment_20261102_DEV01.json"
    if h.sha256_file(archive) != "93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291":
        raise ValueError("archive receipt changed")
    provenance = json.loads(archive.read_text())["provenance"]
    for filename, key in (("actions.pth", "actions_sha256"), ("states.pth", "states_sha256"),
                          ("seq_lengths.pth", "sequence_lengths_sha256")):
        if h.sha256_file(h.DEFAULT_DATASET / filename) != provenance[key]:
            raise ValueError(f"archive changed: {filename}")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    rows, arrays, parity = [], {}, 0.0
    print("Same executed actions: cp12 one RGB frame versus cp2/cp7/cp12 history; target cp17.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data, \
         np.load(REFRESH.with_suffix(".npz"), allow_pickle=False) as saved, \
         h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"] or before != refresh["model_before"]:
            raise ValueError("model parameters changed")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if mean.cpu().tolist() != provenance["action_mean"] or std.cpu().tolist() != provenance["action_std"]:
            raise ValueError("action statistics differ from archive receipt")
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        for row, oldrow in zip(source["rows"], refresh["rows"], strict=True):
            root = row["root"]
            if oldrow["root"] != root or len(row["branches"]) != 8:
                raise ValueError("roster mismatch")
            for branch, prior in zip(row["branches"], oldrow["branches"], strict=True):
                if (branch["arm"], branch["candidate"]) != (prior["arm"], prior["candidate"]):
                    raise ValueError("branch mismatch")
                name = f"r{root}_a{('raw', 'spectral').index(branch['arm'])}_p{branch['candidate']}"
                frames, actions = data[name + "_history"], data[name + "_actions"]
                if frames.shape != (18, 224, 224, 3) or frames.dtype != np.uint8:
                    raise ValueError("invalid RGB history")
                acts = packets(actions, mean, std)
                states = np.stack([h.causal_state(frames[t-1], frames[t], metadata, offset, .05)
                                   for t in (2, 7, 12, 17)])
                with torch.no_grad():
                    enc = [h.encode_with_state(model, frames[t], s, device)
                           for t, s in zip((2, 7, 12, 17), states, strict=True)]
                    context = {k: torch.cat([v[k] for v in enc[:3]], dim=1) for k in ("visual", "proprio")}
                    one = unroll_with_action_history(model, enc[2], acts[:0], acts[2:], 1)
                    three = unroll_with_action_history(model, context, acts[:2], acts[2:], 3)
                    entry = {"root": root, "arm": branch["arm"], "candidate": branch["candidate"]}
                    for k in ("visual", "proprio"):
                        a, b, target = one[k][-1], three[k][-1], enc[3][k][0]
                        reference = torch.as_tensor(saved[f"{name}_refresh_{k}"], device=device)
                        torch.testing.assert_close(a, reference, rtol=3e-5, atol=1e-6)
                        parity = max(parity, float((a-reference).abs().max().item()))
                        entry[f"one_{k}_mse"] = mse(a, target)
                        entry[f"three_{k}_mse"] = mse(b, target)
                        for label, value in (("one", a), ("three", b), ("target", target)):
                            arrays[f"{name}_{label}_{k}"] = value.cpu().numpy()
                arrays[name + "_causal_states"] = states
                arrays[name + "_raw_actions"] = actions
                arrays[name + "_normalized_packets"] = acts.cpu().numpy()
                rows.append(entry)
            print(f"root {root}: 8 paired branches complete", flush=True)
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if after != before:
            raise RuntimeError("model changed")
    with NPZ.open("xb") as out:
        np.savez_compressed(out, **arrays)
    dependencies = [Path(__file__), PROTOCOL, SOURCE, SOURCE.with_suffix(".npz"), REFRESH,
                    REFRESH.with_suffix(".npz"), h.CAMERA, h.OLD, archive, h.DEFAULT_CONFIG,
                    h.DEFAULT_CHECKPOINT, h.DEFAULT_DATASET / "actions.pth",
                    h.DEFAULT_DATASET / "states.pth", h.DEFAULT_DATASET / "seq_lengths.pth",
                    ROOT / "experiments/pointmaze_action_history_unroll_20261103.py",
                    ROOT / "external/jepa-wms/app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py"]
    report = {"schema": "cp12_three_frame_same_action.v1", "rows": rows,
              "clock": {"observations": [2, 7, 12], "action_ranges": [[2, 7], [7, 12], [12, 17]],
                        "prediction_target": 17, "velocity_dt_seconds": .05},
              "counts": {"branches": len(rows), "one_predictor_samples": len(rows),
                         "three_predictor_samples": len(rows), "observation_encodes": 4 * len(rows),
                         "physical_steps": 0},
              "one_saved_max_abs": parity, "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "action_mean": mean.cpu().tolist(),
              "action_std": std.cpu().tolist(), "arrays_sha256": h.sha256_file(NPZ),
              "files_sha256": {str(p): h.sha256_file(p) for p in dependencies},
              "scope": "exposed dependent offline fixed actions; not planner performance or independent confirmation"}
    with OUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(f"Saved {OUT} sha256={h.sha256_file(OUT)}", flush=True)


if __name__ == "__main__":
    main()
