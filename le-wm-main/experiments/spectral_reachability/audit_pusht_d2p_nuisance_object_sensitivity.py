"""CUDA object/held-out-agent goal sensitivity, not physical efficacy."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
from experiments.spectral_reachability import pusht_d2p_goal_nuisance_metric as metric
from experiments.spectral_reachability.analyze_pusht_d2p_goal_agent_intervention import analyze

PERTURBATIONS = (("object_x_minus", 2, -16.), ("object_x_plus", 2, 16.),
    ("object_y_minus", 3, -16.), ("object_y_plus", 3, 16.),
    ("object_angle_minus", 4, -.15), ("object_angle_plus", 4, .15),
    ("agent_diagonal_mm", 0, (-32., -32.)), ("agent_diagonal_mp", 0, (-32., 32.)),
    ("agent_diagonal_pm", 0, (32., -32.)), ("agent_diagonal_pp", 0, (32., 32.)))


def retention(delta, projection):
    delta = np.asarray(delta, float)
    if not np.isfinite(delta).all() or delta.shape != (192,):
        raise ValueError("finite192-vector required")
    energy = float(delta@delta)
    projected = projection@delta
    return {"original_energy": energy, "retained_energy": float(projected@projected),
            "retained_fraction": float(projected@projected)/energy if energy > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("immutable sensitivity run required")
    analyze(args.intervention)
    source = json.loads(args.intervention.read_text())
    bank = json.loads(Path(source["protocol"]["bank_path"]).read_text())
    upstream = base.physical._load_preflight(Path(bank["upstream_path"]))
    if base.digest(args.policy/"weights.pt") != base.OFFICIAL_SHA:
        raise ValueError("original official frozen weights required")
    args.output.mkdir()
    archive = args.output/"implementation"
    archive.mkdir()
    paths = {"runner": Path(__file__), "metric": Path(metric.__file__),
             "goal_context": Path(base.p0.__file__), "base": Path(base.__file__),
             "loader": Path(base.load_compatible_pretrained.__code__.co_filename),
             "collector": Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)}
    hashes = {k: base.digest(p) for k, p in paths.items()}
    for k, p in paths.items(): shutil.copyfile(p, archive/(k+".py"))
    protocol = {"schema": "pusht_d2p_nuisance_sensitivity_protocol.v1", "perturbations": PERTURBATIONS,
        "intervention_path": str(args.intervention.resolve()), "intervention_sha256": base.digest(args.intervention),
        "implementation_sha256": hashes, "expected_encoder_forwards": 44,
        "weights_sha256": base.OFFICIAL_SHA, "config_sha256": base.digest(args.policy/"config.json"),
        "rank_uses_only_original5_goal_vectors": True, "no_performance_threshold_or_selection": True,
        "new_physical_future_branches": 0, "P1_admission": False}
    base.write_new(args.output/"protocol.json", protocol)
    host = base.require_cuda_allocation("cuda")
    model = base.load_compatible_pretrained(args.policy, device=torch.device("cuda")).eval()
    model.requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    transform = base.p0.image_transform()
    rows, forwards = [], 0
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        for index, root in enumerate(roots):
            old = source["rows"][index]
            if base.root_fingerprint(root) != old["root_sha256"]:
                raise ValueError("known root changed")
            goals = np.asarray([v["goal_vector"] for v in old["variants"]], float)
            projection, receipt = metric.compile_metric(goals)
            probes = []
            for name, column, change in (("original", 0, 0.), *PERTURBATIONS):
                goal = np.asarray(root["goal"], float).copy()
                if isinstance(change, tuple): goal[:2] += change
                else: goal[column] += change
                _, info = env.reset(seed=0, options={"state": np.asarray(root["state"]).copy(), "goal_state": goal})
                rgb = np.asarray(info["goal"], np.uint8).copy()
                vector = base.p0._goal_latent(model, rgb, transform=transform, device=torch.device("cuda"))
                if vector.device.type != "cuda": raise RuntimeError("CPU neural fallback forbidden")
                vector = vector.cpu().numpy()
                forwards += 1
                if name == "original":
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != old["variants"][0]["goal_rgb_sha256"]:
                        raise ValueError("original goal render changed")
                    np.testing.assert_allclose(vector, goals[0], rtol=1e-6, atol=1e-6)
                probes.append({"name": name, "goal_vector": vector.tolist(),
                    "goal_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    **retention(vector-goals[0], projection)})
            rows.append({"root_id": index, "root_sha256": old["root_sha256"], "metric_receipt": receipt,
                "projection": projection.tolist(), "probes": probes})
    finally: env.close()
    gates = {"cuda_only": True, "all44_forwards": forwards == 44,
        "parameters_unchanged": before == base.p0._parameter_hash(model),
        "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "all_eval": all(not m.training for m in model.modules()),
        "source_unchanged": all(base.digest(p) == hashes[k] for k, p in paths.items()),
        "weights_unchanged": base.digest(args.policy/"weights.pt") == base.OFFICIAL_SHA,
        "config_unchanged": base.digest(args.policy/"config.json") == protocol["config_sha256"],
        "intervention_unchanged": base.digest(args.intervention) == protocol["intervention_sha256"]}
    summary = {kind: {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}
        for kind in ("object", "agent") for values in [[p["retained_fraction"] for r in rows for p in r["probes"]
            if p["name"].startswith(kind) and p["retained_fraction"] is not None]]}
    base.write_new(args.output/"report.json", {"schema": "pusht_d2p_nuisance_sensitivity.v1",
        "status": "completed" if all(gates.values()) else "integrity_fail", "protocol": protocol,
        "host": host, "gates": gates, "rows": rows, "summary": summary, "encoder_forwards": forwards,
        "new_predict_samples": 0, "new_physical_future_branches": 0, "P1_admission": False,
        "scope": "Goal-render perturbation sensitivity only. Not control accuracy, global invariance, new-root efficacy or P1."})
    print(json.dumps({"gates": gates, "summary": summary}))
    if not all(gates.values()): raise RuntimeError("sensitivity integrity failed")


if __name__ == "__main__": main()
