"""Known full-bank CUDA suffix intervention, never a formal P1 admission."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
from experiments.spectral_reachability import pusht_d2p_consistent_metric_suffix as solver
from experiments.spectral_reachability import pusht_d2p_goal_nuisance_metric as metric
from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability.analyze_pusht_d2p_nuisance_object_sensitivity import analyze

MODES = ("identity_gn4", "nuisance_gn4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable suffix experiment required")
    analyze(args.sensitivity_report)
    sensitivity = json.loads(args.sensitivity_report.read_text())
    intervention_path = Path(sensitivity["protocol"]["intervention_path"])
    intervention = json.loads(intervention_path.read_text())
    bank_path = Path(intervention["protocol"]["bank_path"])
    bank = json.loads(bank_path.read_text())
    base.summarize(bank)
    base.verify_stages(bank_path, bank)
    upstream = base.physical._load_preflight(Path(bank["upstream_path"]))
    if base.digest(args.policy/"weights.pt") != base.OFFICIAL_SHA:
        raise ValueError("official original frozen checkpoint required")
    args.output.mkdir()
    stages, archive = args.output/"stages", args.output/"implementation"
    stages.mkdir()
    archive.mkdir()
    paths = {"runner": Path(__file__), "solver": Path(solver.__file__), "metric": Path(metric.__file__),
        "GN": Path(solver.solver.__file__), "dynamics": Path(p1.__file__), "context": Path(base.p0.__file__),
        "physical": Path(base.physical.__file__), "base": Path(base.__file__),
        "loader": Path(base.load_compatible_pretrained.__code__.co_filename),
        "collector": Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename)}
    hashes = {k: base.digest(p) for k,p in paths.items()}
    for k,p in paths.items(): shutil.copyfile(p, archive/(k+".py"))
    bindings = {str(p.resolve()): base.digest(p) for p in (args.sensitivity_report, intervention_path, bank_path, Path(bank["upstream_path"]))}
    protocol = {"schema": "pusht_d2p_nuisance_suffix_bank_protocol.v1", "bindings": bindings,
        "bank_path": str(bank_path.resolve()), "modes": MODES, "root_ids": bank["protocol"]["roots"],
        "proposal_indices": list(range(16)), "implementation_sha256": hashes,
        "weights_sha256": base.OFFICIAL_SHA, "config_sha256": base.digest(args.policy/"config.json"),
        "expected_predict_samples": 3968, "expected_real_branches": 128,
        "iterations": 4, "damping": .01, "all_prefixes_fixed_unchanged": True,
        "goal_set_only_metric_construction": True, "no_outcome_selection": True, "P1_admission": False,
        "scope": "Known4 calibration roots all16 prefixes, observed-RGB suffix recompilation only. Goal-set input, not single-RGB-only. Not full prefix/stage/compiler repair, strong-baseline efficacy or independent P1."}
    base.write_new(args.output/"protocol.json", protocol)
    host = base.require_cuda_allocation("cuda")
    device = torch.device("cuda")
    model = base.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    before = base.p0._parameter_hash(model)
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    original, samples = model.predict, 0
    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda": raise RuntimeError("CPU neural fallback forbidden")
        samples += len(emb)
        return original(emb, actions)
    model.predict = counted
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    transform = base.p0.image_transform()
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        for index, root in enumerate(roots):
            if base.root_fingerprint(root) != protocol["root_ids"][index]: raise ValueError("known root changed")
            group = sorted([r for r in bank["rows"] if r["root_id"] == index], key=lambda r:r["proposal_index"])
            goals = np.asarray([v["goal_vector"] for v in intervention["rows"][index]["variants"]], float)
            projection, receipt = metric.compile_metric(goals)
            matrices = {MODES[0]: torch.eye(192, device=device), MODES[1]: torch.as_tensor(projection, dtype=torch.float32, device=device)}
            goal = torch.as_tensor(goals[0], dtype=torch.float32, device=device)
            root_plan = {"root_id": index, "root_sha256": protocol["root_ids"][index], "goal_vector": goals[0].tolist(),
                "projection": projection.tolist(), "metric_receipt": receipt,
                "prefixes": [r["prefix_action"] for r in group], "physical_future_labels_used": False}
            base.write_new(stages/f"root_{index}.plan.json", root_plan)
            for old in group:
                candidate, branches = old["proposal_index"], {}
                prefix = np.asarray(old["prefix_action"], np.float32)
                for mode in MODES:
                    frames, history, rgbgoal = base.physical._reset_and_warmup(env, root, history_len=3)
                    if base.physical._sha256(rgbgoal) != upstream["rows"][index]["goal_rgb_sha256"]:
                        raise ValueError("goal render differs from original")
                    base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(prefix))
                    rgb = np.asarray(env.render(), np.uint8).copy()
                    state = base.p0.packets.state_of(env)
                    if base.physical._sha256(rgb) != old["branches"]["full192_gn4"]["successor_rgb_sha256"]:
                        raise ValueError("same-prefix RGB replay drift")
                    emb, actions = p1._encode_context(model, frames[1:]+[rgb], np.concatenate((history[1:], [prefix])), transform, device)
                    action, forecast = solver.solve(model, emb, actions, goal, matrices[mode])
                    stage = {"root_id": index, "proposal_index": candidate, "mode": mode,
                        "root_plan_sha256": base.digest(stages/f"root_{index}.plan.json"),
                        "observed_vector": emb[0,-1].cpu().tolist(), "suffix_action": action.tolist(),
                        "forecast": forecast, "future_suffix_observations_used": False}
                    stage_path = stages/f"root_{index}.candidate_{candidate}.{mode}.json"
                    base.write_new(stage_path, stage)
                    base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(action[0]))
                    final_rgb = np.asarray(env.render(), np.uint8).copy()
                    with torch.no_grad():
                        final = model.encode({"pixels": base.p0._image_tensor(final_rgb, transform=transform, device=device)})["emb"][0,0]
                    if mode == MODES[0]:
                        baseline = old["branches"]["full192_gn4"]
                        np.testing.assert_array_equal(action, baseline["suffix_action"])
                        if base.physical._sha256(final_rgb) != baseline["final_rgb_sha256"]:
                            raise ValueError("identity baseline final replay drift")
                    branches[mode] = {"stage_path": str(stage_path.resolve()), "stage_sha256": base.digest(stage_path),
                        "successor_state": state.tolist(), "prefix_pose": base.physical._pose_label(state, root["goal"]),
                        "final_state": base.p0.packets.state_of(env).tolist(),
                        "final_pose": base.physical._pose_label(base.p0.packets.state_of(env), root["goal"]),
                        "final_vector": final.cpu().tolist(), "final_rgb_sha256": base.physical._sha256(final_rgb),
                        "suffix_action": action.tolist(), "forecast": forecast, "observed_vector": stage["observed_vector"]}
                rows.append({"root_id": index, "proposal_index": candidate, "branches": branches})
            print(json.dumps({"root": index, "real_branches": len(rows)*2, "predict_samples": samples}), flush=True)
    finally:
        env.close()
        model.predict = original
    gates = {"cuda_only": True, "all64_pairs": len(rows) == 64, "all128_branches": sum(len(r["branches"]) for r in rows) == 128,
        "predict_accounting": samples == 3968, "parameters_unchanged": before == base.p0._parameter_hash(model),
        "buffers_unchanged": all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_eval": all(not m.training for m in model.modules()),
        "source_unchanged": all(base.digest(p) == hashes[k] for k,p in paths.items()),
        "inputs_unchanged": all(base.digest(Path(p)) == sha for p,sha in bindings.items()),
        "weights_unchanged": base.digest(args.policy/"weights.pt") == base.OFFICIAL_SHA,
        "config_unchanged": base.digest(args.policy/"config.json") == protocol["config_sha256"]}
    base.write_new(args.output/"report.json", {"schema": "pusht_d2p_nuisance_suffix_bank.v1", "status": "completed" if all(gates.values()) else "integrity_fail",
        "protocol": protocol, "host": host, "gates": gates, "actual_predict_samples": samples,
        "physical_future_branches": 128, "rows": rows, "P1_admission": False})
    print(json.dumps({"gates": gates, "predict_samples": samples}), flush=True)
    if not all(gates.values()): raise RuntimeError("suffix-bank integrity failed")


if __name__ == "__main__": main()
