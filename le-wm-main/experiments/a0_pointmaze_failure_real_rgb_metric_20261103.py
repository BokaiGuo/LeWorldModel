"""Actual successor RGB versus predicted and physical costs on fixed banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.a0_pointmaze_extra_cem_failure_oracle_20261103 import (
    ORIGINAL, ORIGINAL_SHA, OUTPUT as ORACLE, SOURCE as PLANS, SOURCE_SHA as PLANS_SHA,
)
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.a0_pointmaze_proprio_action_response_20261103 import encode_with_state
from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import pairwise_ranking
from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import replay_legacy
from experiments.spectral_reachability import codex_p3_queued_cost_lock_20261030_v1 as lock
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import (
    DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_DATASET, ROOT, load_model_and_dataset,
    parameter_hash, require_cuda, sha256_file,
)

ORACLE_SHA = "dc78522fb837507e37fcf255c275e885210fb3e2fbb8dec1e2f804700f2ae8ef"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FAILURE_REAL_RGB_METRIC_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_failure_real_rgb_metric_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((ORACLE, ORACLE_SHA), (PLANS, PLANS_SHA), (ORIGINAL, ORIGINAL_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    oracle, plans, original = (json.loads(p.read_text()) for p in (ORACLE, PLANS, ORIGINAL))
    for path, expected in ((ORACLE.with_suffix(".npz"), oracle["arrays_sha256"]),
                           (PLANS.with_suffix(".npz"), plans["arrays_sha256"]),
                           (DEFAULT_DATASET / "states.pth", original["states_sha256"]),
                           (DEFAULT_DATASET / "actions.pth", original["actions_sha256"]),
                           (DEFAULT_CHECKPOINT, original["checkpoint_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source input changed: {path}")
    states = torch.load(DEFAULT_DATASET / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    branches = {(row["root"], branch["method"]): branch for row in plans["rows"] for branch in row["branches"]}
    starts, sequences, expected, started = [], [], [], time.monotonic()
    with np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as bank:
        for row in oracle["rows"]:
            root, method, phase, n = row["root"], row["method"], row["phase"], row["n"]
            key = f"r{root}_{method}_{phase}"
            candidates = bank[f"{key}_candidates"]
            if len(candidates) != n:
                raise RuntimeError("candidate count mismatch")
            prefix = np.asarray(branches[root, method]["initial_plan"], np.float32)[:1]
            for candidate in candidates:
                plan = candidate if phase == "initial" else np.concatenate((prefix, candidate))
                sequences.append(np.repeat(plan, 5, axis=0))
                starts.append(states[root, 0])
            expected.extend(bank[f"{key}_endpoints"])
    if len(starts) != 624:
        raise RuntimeError("unexpected physical input count")
    print("Replaying 624 fixed candidates for actual terminal RGB scores.", flush=True)
    xml = official_xml()
    images, endpoints = replay_legacy(starts, sequences, xml)
    if images.shape != (624, 224, 224, 3) or images.dtype != np.uint8:
        raise RuntimeError("invalid RGB payload")
    if not np.array_equal(endpoints[:, :2], np.asarray(expected)[:, :2]):
        raise RuntimeError("legacy position mismatch with sealed physical oracle")
    args = argparse.Namespace(dataset_root=DEFAULT_DATASET, checkpoint=DEFAULT_CHECKPOINT,
                              config=DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
    arrays, rows, offset = {"rgb10": images, "legacy_endpoints": endpoints}, [], 0
    with np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(PLANS.with_suffix(".npz"), allow_pickle=False) as inputs, \
            lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = require_cuda()
        model, dataset = load_model_and_dataset(args, device)
        before = parameter_hash(model)
        if before != plans["model_before"]:
            raise RuntimeError("model changed")
        for row in oracle["rows"]:
            root, method, phase, n = row["root"], row["method"], row["phase"], row["n"]
            key = f"r{root}_{method}_{phase}"
            mask = torch.as_tensor(inputs[f"r{root}_mask"], dtype=torch.bool, device=device)
            goal = encode_with_state(model, inputs[f"r{root}_goal_rgb"], np.zeros(4), device)["visual"][0, 0]
            values = []
            with torch.no_grad():
                for batch in np.array_split(images[offset:offset+n], (n+15)//16):
                    visual = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).float()[:, None].to(device)
                    z = model.encode({"visual": visual, "proprio": torch.zeros(len(batch), 1, 4, device=device)})["visual"][:, 0]
                    error = ((z - goal) ** 2).mean(dim=(1, 4))
                    values.extend(error[:, mask].mean(dim=1).cpu().tolist())
            offset += n
            true_cost = np.asarray(values, np.float64)
            model_score, distance = bank[f"{key}_scores"], bank[f"{key}_distances"]
            if true_cost.shape != (n,) or not np.isfinite(true_cost).all():
                raise RuntimeError("invalid actual RGB costs")
            arrays[f"{key}_real_cost"] = true_cost
            real_choice = int(np.argmin(true_cost))
            rows.append({"root": root, "method": method, "phase": phase, "n": n,
                         "model_vs_real_rgb": pairwise_ranking(model_score, true_cost),
                         "real_rgb_vs_physical": pairwise_ranking(true_cost, distance),
                         "model_vs_physical": pairwise_ranking(model_score, distance),
                         "model_choice": int(np.argmin(model_score)), "real_rgb_oracle_choice": real_choice,
                         "real_rgb_oracle_distance": float(distance[real_choice]),
                         "model_choice_distance": float(distance[np.argmin(model_score)]),
                         "physical_oracle_distance": float(np.min(distance))})
            print(f"{key}: real RGB encoded", flush=True)
        after = parameter_hash(model)
        if before != after:
            raise RuntimeError("model parameters changed")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_failure_real_rgb_metric.v1", "oracle_sha256": ORACLE_SHA,
              "plan_sha256": PLANS_SHA, "protocol_sha256": sha256_file(PROTOCOL),
              "script_sha256": sha256_file(Path(__file__)), "arrays_sha256": sha256_file(ARRAYS),
              "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(), "model_before": before, "model_after": after,
              "device": torch.cuda.get_device_name(device), "actual_rgb_encodes": 624, "goal_encodes": 12,
              "physical_sequences": 624, "position_exact_count": 624,
              "wall_seconds": time.monotonic()-started,
              "scope": "outcome-selected exposed roots; real future is oracle; no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
