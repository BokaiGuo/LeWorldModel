"""CUDA goal-only intervention; no action changes, future execution or P1 pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.audit_pusht_reachability_horizon import collect_nonterminal_boundary_contact_roots
from experiments.pusht_protocol_contracts import root_fingerprint
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
from experiments.spectral_reachability import run_pusht_d2p_p1_physical_calibration as physical
from experiments.spectral_reachability.analyze_pusht_d2p_full_metric_bank import MODES, summarize, verify_stages
from experiments.spectral_reachability.audit_pusht_d2p_full_metric_bank import OFFICIAL_SHA
from experiments.spectral_reachability.analyze_pusht_d2p_physical_calibration import digest

OFFSETS = ((0., 0.), (-32., 0.), (32., 0.), (0., -32.), (0., 32.))


def costs(vectors, goal):
    values, target = np.asarray(vectors, float), np.asarray(goal, float)
    if values.ndim != 2 or values.shape[1] != 192 or target.shape != (192,) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("finite full-coordinate model vectors required")
    return np.mean((values-target)**2, axis=1).tolist()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable intervention output required")
    args.output.mkdir()
    bank = json.loads(args.bank_report.read_text())
    summarize(bank)
    verify_stages(args.bank_report, bank)
    upstream_path = Path(bank["upstream_path"])
    upstream = physical._load_preflight(upstream_path)
    if digest(args.policy/"weights.pt") != OFFICIAL_SHA:
        raise ValueError("official frozen checkpoint required")
    sources = {"auditor": Path(__file__), "context": Path(p0.__file__),
               "physical": Path(physical.__file__), "loader": Path(load_compatible_pretrained.__code__.co_filename),
               "root_collector": Path(collect_nonterminal_boundary_contact_roots.__code__.co_filename),
               "bank_analyzer": Path(summarize.__code__.co_filename)}
    archive = args.output/"implementation"
    archive.mkdir()
    hashes = {}
    for name, source in sources.items():
        hashes[name] = digest(source)
        shutil.copyfile(source, archive/(name+".py"))
    protocol = {"schema": "pusht_d2p_goal_agent_intervention_protocol.v1",
        "bank_path": str(args.bank_report.resolve()), "bank_sha256": digest(args.bank_report),
        "upstream_sha256": digest(upstream_path), "weights_sha256": OFFICIAL_SHA,
        "config_sha256": digest(args.policy/"config.json"), "implementation_sha256": hashes,
        "root_ids": bank["protocol"]["roots"], "goal_agent_xy_offsets": OFFSETS,
        "task_object_goal_unchanged": True, "all16_proposals_all3_solvers_retained": True,
        "model_prediction_samples": 0, "expected_goal_encoder_forwards": 20,
        "new_physical_future_branches": 0, "P1_admission": False,
        "scope": "Known calibration goal-image nuisance intervention. Simulator state only renders specified goal images, never enters action scoring. Not an implementable RGB-only nuisance remover, a policy change or matched efficacy comparison."}
    write_new(args.output/"protocol.json", protocol)
    runtime = require_cuda_allocation("cuda")
    device = torch.device("cuda")
    model = load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    before = p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    transform = p0.image_transform()
    rows, forwards = [], 0
    try:
        roots = collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        for index, root in enumerate(roots):
            if root_fingerprint(root) != protocol["root_ids"][index]:
                raise ValueError("known root identity changed")
            group = sorted([r for r in bank["rows"] if r["root_id"] == index], key=lambda r: r["proposal_index"])
            variants = []
            for dx, dy in OFFSETS:
                goal = np.asarray(root["goal"], float).copy()
                goal[:2] += [dx, dy]
                if not np.array_equal(goal[2:], np.asarray(root["goal"])[2:]):
                    raise ValueError("intervention changed physical object goal")
                _, info = env.reset(seed=0, options={"state": np.asarray(root["state"]).copy(), "goal_state": goal})
                rgb = np.asarray(info["goal"], np.uint8).copy()
                if device.type != "cuda":
                    raise RuntimeError("CPU neural fallback forbidden")
                vector = p0._goal_latent(model, rgb, transform=transform, device=device).cpu().numpy()
                forwards += 1
                if (dx, dy) == (0., 0.):
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != upstream["rows"][index]["goal_rgb_sha256"]:
                        raise ValueError("original goal render changed")
                    np.testing.assert_allclose(vector, group[0]["goal_vector"], rtol=1e-6, atol=1e-6)
                predictions = {}
                for mode in MODES:
                    values = costs([r["predicted"]["suffixes"][mode]["predicted_terminal_vector"] for r in group], vector)
                    predictions[mode] = {"terminal_costs": values,
                        "model_terminal_cost_argmin": int(np.argmin(values)),
                        "note": "diagnostic fixed-bank terminal argmin, not original reconfiguration selector"}
                variants.append({"offset": [dx, dy], "goal_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "goal_vector": vector.tolist(), "model_fixed_actions": predictions})
            rows.append({"root_id": index, "root_sha256": protocol["root_ids"][index], "variants": variants})
    finally:
        env.close()
    gates = {"cuda_only": device.type == "cuda", "all20_goal_forwards": forwards == 20,
        "parameters_unchanged": before == p0._parameter_hash(model),
        "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "all_modules_eval": all(not m.training for m in model.modules()),
        "sources_unchanged": all(digest(p) == hashes[k] for k, p in sources.items()),
        "weights_unchanged": digest(args.policy/"weights.pt") == OFFICIAL_SHA,
        "config_unchanged": digest(args.policy/"config.json") == protocol["config_sha256"],
        "bank_unchanged": digest(args.bank_report) == protocol["bank_sha256"]}
    result = {"schema": "pusht_d2p_goal_agent_intervention.v1", "status": "completed" if all(gates.values()) else "integrity_fail",
        "protocol": protocol, "runtime": runtime, "gates": gates, "rows": rows,
        "goal_encoder_forwards": forwards, "new_predict_samples": 0, "new_physical_future_branches": 0,
        "P1_admission": False,
        "argmin_changes_from_original": {mode: sum(v["model_fixed_actions"][mode]["model_terminal_cost_argmin"] != r["variants"][0]["model_fixed_actions"][mode]["model_terminal_cost_argmin"]
            for r in rows for v in r["variants"][1:]) for mode in MODES}}
    write_new(args.output/"report.json", result)
    print(json.dumps({"gates": gates, "argmin_changes_from_original": result["argmin_changes_from_original"]}))
    if not all(gates.values()):
        raise RuntimeError("intervention integrity failed")


if __name__ == "__main__":
    main()
