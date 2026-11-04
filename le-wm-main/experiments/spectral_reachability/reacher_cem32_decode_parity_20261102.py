"""Audit CUDA/CPU decoding of all six saved CEM32 development tails."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import reacher_cem32_same_chain_rgb_20261102 as rgb

host, setup = rgb.host, rgb.setup
ROOT = rgb.ROOT
PROTOCOL = ROOT.parent / "docs/PC_REACHER_CEM32_DECODE_PARITY_PROTOCOL_20261102.md"
AUTH = "REACHER_CEM32_DECODE_PARITY_DEVELOPMENT"


def compare(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    host.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "array validity")
    return {"bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
            "unequal_elements": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
            "max_abs": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))}


def prepare():
    rgb.prepare()  # validates the physical plan, source chain and stored replay.
    report = rgb.exp.live.SELECTION_REPORT
    old = json.loads(rgb.exp.run.physical.OLD_PLAN.read_text())
    files = [Path(__file__), PROTOCOL, report, rgb.PLAN, rgb.JOURNAL, rgb.RECEIPT, Path(old["stats"]),
             Path(setup.p0.__file__)]
    return {"schema": "reacher_cem32_decode_parity_v1",
            "files": {str(p): setup._sha256_file(p) for p in files},
            "roots": [20263160, 20263161], "arms": 6,
            "physical_steps": 0, "neural_calls": 0, "P1_admission": False}


def run(plan_path, digest, output, authority):
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == digest and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_cem32_decode_parity_RUN_") and not os.path.lexists(output),
                 "exclusive output")
    old = json.loads(rgb.exp.run.physical.OLD_PLAN.read_text())
    physical = json.loads(rgb.PLAN.read_text())
    selected = json.loads(rgb.exp.live.SELECTION_REPORT.read_text())
    mean, std = setup._load_action_stats(Path(old["stats"]), "swm/ReacherDMControl-v0")
    lo, hi = setup._normalized_action_bounds(mean, std, np.array([-1., -1.], np.float32),
                                              np.array([1., 1.], np.float32))
    raw_lo, raw_hi = np.array([-1., -1.], np.float32), np.array([1., 1.], np.float32)
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        rows = []
        for root, score in zip(physical["roots"], selected["rows"], strict=True):
            host.require(root["seed"] == score["seed"], "root order")
            for branch in root["branches"]:
                candidate = score["selection"]["proposal_rows"][branch["prefix_index"]]
                host.require(candidate["proposal_index"] == branch["prefix_index"], "candidate identity")
                params = np.asarray(candidate["suffix_parameters"], np.float32)
                host.require(params.shape == (2, 10), "suffix shape")
                cpu = torch.as_tensor(params).unsqueeze(0)
                gpu = cpu.to("cuda")
                cpu_raw, cpu_normal = setup.p0._to_raw_actions(
                    cpu, mean, std, torch.as_tensor(lo), torch.as_tensor(hi), raw_lo, raw_hi)
                gpu_raw, gpu_normal = setup.p0._to_raw_actions(
                    gpu, mean, std, torch.as_tensor(lo, device="cuda"), torch.as_tensor(hi, device="cuda"),
                    raw_lo, raw_hi)
                executed = np.asarray(branch["raw_actions"], np.float32)[5:]
                rows.append({"seed": root["seed"], "role": branch["role"], "prefix": branch["prefix_index"],
                             "cpu_raw": cpu_raw.tolist(), "cuda_raw": gpu_raw.tolist(),
                             "cpu_normalized": cpu_normal.tolist(), "cuda_normalized": gpu_normal.tolist(),
                             "cpu_vs_cuda_raw": compare(cpu_raw, gpu_raw),
                             "cpu_vs_cuda_normalized": compare(cpu_normal, gpu_normal),
                             "cpu_vs_executed": compare(cpu_raw, executed),
                             "cuda_vs_executed": compare(gpu_raw, executed)})
        host.require(len(rows) == 6 and json.loads(plan_path.read_text()) == prepare(), "postcheck")
        result = {"schema": "reacher_cem32_decode_parity_result_v1", "plan_sha256": digest,
                  "cuda_allocation": allocation, "rows": rows, "P1_admission": False,
                  "limitation": "CUDA decoding reconstructed, not saved in original scoring run"}
        host.write_new(output, result)
        print(json.dumps({"sha256": setup._sha256_file(output),
                          "rows": [{k: v for k, v in row.items() if k not in
                                    ("cpu_raw", "cuda_raw", "cpu_normalized", "cuda_normalized")}
                                   for row in rows]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
