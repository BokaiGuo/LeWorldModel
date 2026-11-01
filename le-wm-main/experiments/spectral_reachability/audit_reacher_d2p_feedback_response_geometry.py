"""All-eight-root post-prefix physical response, evaluator only and CUDA only.

Replay every already sealed prefix (four families, replay and tracking), then
measure both signs of all twenty future command coordinates at two fixed
scales. No measured future, qpos or physical response enters a planner.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import reacher_d2p_box_action_suffix as box
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import analyze_reacher_d2p_waypoint_tracking as upstream
from experiments.spectral_reachability.bounded_response_qp import certified_response_gain
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_post_prefix_geometry import angular_difference
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

SCALES = (.1, .25)
METHODS = tuple(f + "_" + m for f in upstream.FAMILIES for m in upstream.MODES)


def probes(scale):
    if scale not in SCALES:
        raise ValueError("predeclared command scale required")
    result = []
    for i in range(20):
        value = np.zeros(20, np.float32)
        value[i] = scale
        result.append((i, value, -value))
    return result


def validate_endpoints(endpoints, scale):
    if len(endpoints) != 20 or [e["coordinate"] for e in endpoints] != list(range(20)):
        raise ValueError("twenty ordered full-coordinate probes required")
    matrices = {"model_fd": [], "physical_rgb_fd": [], "physical_qpos_fd": []}
    for e in endpoints:
        step = float(np.float32(scale))
        for sign in ("plus", "minus"):
            v = e[sign]
            for field, shape in (("model", (192,)), ("observed", (192,)), ("qpos", (2,)), ("raw_actions", (10, 2))):
                a = np.asarray(v[field], np.float64)
                if a.shape != shape or not np.isfinite(a).all():
                    raise ValueError("finite complete probe endpoints required")
            if np.max(np.abs(v["raw_actions"])) > 1 + 1e-6:
                raise ValueError("physical probe outside action box")
        for output, source in (("model_fd", "model"), ("physical_rgb_fd", "observed")):
            matrices[output].append((np.asarray(e["plus"][source]) - np.asarray(e["minus"][source])) / (2 * step))
        matrices["physical_qpos_fd"].append(angular_difference(e["plus"]["qpos"], e["minus"]["qpos"]) / (2 * step))
    return {k: np.stack(v, axis=1) for k, v in matrices.items()}


def geometry(response, residual, *, physical):
    result = certified_response_gain(response, residual, damping=.01)
    # The shared numerical solver cannot know whether its input is measured.
    # Override its planner-oriented label for evaluator-only physical inputs.
    return {**result, "physical_labels_used": physical, "deployed": False}


def snapshot_packets(replays, branch):
    if len(replays) != 2 or [r["repeat"] for r in replays] != [0, 1]:
        raise ValueError("both independently verified snapshot replays required")
    first, second = (r["packets"] for r in replays)
    if len(first) != 4 or len(second) != 4:
        raise ValueError("all four observed prefix snapshots required")
    for i, (left, right) in enumerate(zip(first, second)):
        old = branch["packets"][i]
        if (left["packet"] != i + 1 or right["packet"] != i + 1 or left["snapshot"] != right["snapshot"]
                or any(p[k] is not True for p in (left, right) for k in ("root_matches_history", "frames_match_history", "actions_match_history", "vector_matches_history"))
                or left["root_rgb_sha256"] != old["root_rgb_sha256"] or left["history_rgb_sha256"] != old["history_rgb_sha256"]
                or left["snapshot"]["observed_root_rgb"]["sha256"] != old["root_rgb_sha256"]):
            raise ValueError("snapshot/observation must exactly match both replays and historical receipt")
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--prefix-snapshot-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = upstream.load(args.reference)
    report = json.loads(args.reference.read_text())
    replay_report = json.loads(args.prefix_snapshot_reference.read_text())
    replay_gates = {"cuda_only", "all8_roots_and512_boundaries", "parameters_unchanged", "buffers_unchanged", "implementation_unchanged", "reference_unchanged"}
    if (replay_report.get("schema") != "reacher_d2p_prefix_replay_stability.v1" or replay_report.get("status") != "completed"
            or not replay_gates.issubset(replay_report.get("gates", {})) or not all(v is True for v in replay_report["gates"].values())
            or replay_report["historical_mismatches"] or replay_report["repeat_differences"]
            or replay_report["sealed_protocol"]["reference_sha256"] != digest(args.reference)
            or replay_report["sealed_protocol"] != json.loads(args.prefix_snapshot_reference.with_suffix(".protocol.json").read_text())):
        raise ValueError("complete strict repeated prefix snapshot audit required")
    replay_rows = {r["seed"]: r for r in replay_report["rows"]}
    if len(replay_rows) != 8 or len(replay_report["rows"]) != 8 or set(replay_rows) != set(prior["root_seeds"]):
        raise ValueError("all eight unique repeated snapshot roots required")
    initial_path = args.reference.with_suffix(".selection.json")
    initial = json.loads(initial_path.read_text())
    ref = report["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]
            or source_sha != ref["source_sha256"]):
        raise ValueError("original frozen checkpoint/source/action receipt required")
    paths = {k: args.output.with_suffix(ext) for k, ext in (("protocol", ".protocol.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable geometry outputs required")
    modules = {"auditor": Path(__file__), "upstream": Path(upstream.__file__), "context": Path(execution.__file__),
        "p0": Path(p0.__file__), "goal_contract": Path(goals.__file__), "box": Path(box.__file__),
        "qp": Path(certified_response_gain.__code__.co_filename), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    for key in ("context", "p0", "goal_contract", "box", "qp", "source_context", "loader"):
        if digest(modules[key]) != ref["implementation_sha256"][key]:
            raise ValueError("upstream implementation changed: " + key)
    protocol = {"root_seeds": prior["root_seeds"], "reference_sha256": digest(args.reference), "reference_path": str(args.reference.resolve()),
        "initial_selection_sha256": digest(initial_path), "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"],
        "prefix_snapshot_reference_path": str(args.prefix_snapshot_reference.resolve()), "prefix_snapshot_reference_sha256": digest(args.prefix_snapshot_reference),
        "prefix_observations": "Actual RGB and exact integration/scene snapshots from two fully bit-exact replay passes; never substitute re-rendered pixels.",
        "source_sha256_receipt": source_sha, "target_delta": ref["target_delta"], "goal_contract": "joint-feasible-v1",
        "methods": list(METHODS), "prefix_packets": 4, "future_packets": 2, "command_coordinates": 20,
        "command_scales": list(SCALES), "signs": [-1, 1], "common_task_reference": "zero_replay center-command future",
        "response_gain_damping": .01, "physical_labels_used_for_planning": False,
        "source_verification": "Existing full-source SHA receipt reused; all actual root identities rechecked, no new full-file hash claimed.",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "Known-root evaluator-only finite-scale mechanism audit. Not deployed, not a controllability theorem, calibration, P1 admission or compute-matched efficacy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    for name in ("roots", "implementation"):
        paths[name].mkdir()
    for k, p in modules.items():
        shutil.copyfile(p, paths["implementation"] / (k + ".py"))
    print("HOST CUDA: all8 roots x all8 sealed prefixes, two scales x20 future coordinates x both signs. Measure actual RGB and joint response, not planner oracle inputs or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters = shared._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    original_predict, samples = model.predict, 0

    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return original_predict(emb, actions)

    model.predict = counted
    env = goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = (torch.as_tensor(v, device="cuda") for v in (lo, hi))

    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))

    def raw_commands(flat):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * flat.reshape(1, 2, 10))[0].detach().cpu().numpy()
        return p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)

    plans = {r["seed"]: r for r in initial["rows"]}
    rows = []
    try:
        for old in sorted(report["rows"], key=lambda r: r["seed"]):
            seed, cells = old["seed"], {}
            ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(ctx[k] != plans[seed][k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")):
                raise RuntimeError("source/root/goal identities changed")
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            target = torch.tensor(plans[seed]["goal_vector"], device="cuda")
            for name in METHODS:
                frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                p0._verify_execution_identity(frame, goal, ctx["root_frame"], ctx["goal"])
                current, diagnostics = ctx, []
                raw = np.asarray(old["methods"][name]["action_witness_raw"], np.float32)[:20]
                saved_packets = snapshot_packets(replay_rows[seed]["methods"][name], old["methods"][name])
                for i in range(4):
                    current = execution.advance_executed_context(env, context=current, raw_action_block=raw[i * 5:(i + 1) * 5], action_block=5, verified_root_frame=current["root_frame"])
                    current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
                    packet = old["methods"][name]["packets"][i]
                    execution.assert_exact_execution_state(saved_packets[i]["snapshot"], current["execution_snapshot"])
                    if current["advanced_action_sha256"] != packet["action_sha256"]:
                        raise RuntimeError("sealed prefix action changed")
                    true_frame = execution.unpack_rgb_observation(saved_packets[i]["snapshot"]["observed_root_rgb"])
                    difference = current["root_frame"].astype(np.int16) - true_frame.astype(np.int16)
                    diagnostics.append({"packet": i + 1, "physical_integration_and_scene_exact": True,
                        "rerendered_rgb_exact": current["root_rgb_sha256"] == packet["root_rgb_sha256"],
                        "max_pixel_difference": int(np.abs(difference).max()), "changed_components": int(np.count_nonzero(difference))})
                # Reuse only genuine, independently verified observations at
                # this exact state; no approximate image acceptance or oracle.
                current = {**current, "frames": [execution.unpack_rgb_observation(p["snapshot"]["observed_root_rgb"]) for p in saved_packets[:3]],
                    "root_frame": execution.unpack_rgb_observation(saved_packets[3]["snapshot"]["observed_root_rgb"]),
                    "history_action_blocks": [block.copy() for block in raw[5:20].reshape(3, 10)],
                    "execution_snapshot": saved_packets[3]["snapshot"], "root_rgb_sha256": saved_packets[3]["root_rgb_sha256"],
                    "history_rgb_sha256": saved_packets[3]["history_rgb_sha256"]}
                execution.restore_execution_snapshot(env, seed=seed, snapshot=current["execution_snapshot"], root_identity="sealed_observation_exact_state")
                pix, _, hist = prepare(current)
                with torch.no_grad():
                    observed = p0._latent_context(model, pix, hist)[0, -1]
                if observed.tolist() != old["methods"][name]["packets"][3]["observed_vector"]:
                    raise RuntimeError("complete sealed actual-history embedding changed")
                emb = p0._latent_context(model, pix, hist).detach()
                snapshot = current["execution_snapshot"]

                def predict(flat):
                    return box.terminal_commands(model, emb, hist, flat.reshape(1, 2, 10), action_lower=lower, action_upper=upper)[0]

                def observe(flat):
                    actions = raw_commands(flat)
                    restored, goal_image = execution.restore_execution_snapshot(env, seed=seed, snapshot=snapshot, root_identity="sealed_observation_exact_state")
                    p0._verify_execution_identity(restored, goal_image, current["root_frame"], current["goal"])
                    advanced = current
                    for j in range(2):
                        advanced = execution.advance_executed_context(env, context=advanced, raw_action_block=actions[j * 5:(j + 1) * 5], action_block=5, verified_root_frame=advanced["root_frame"])
                    pixels, _, history = prepare(advanced)
                    with torch.no_grad():
                        actual = p0._latent_context(model, pixels, history)[0, -1]
                    return {"observed": actual.tolist(), "qpos": np.asarray(env.unwrapped.env.physics.data.qpos, np.float64)[:2].tolist(),
                        "raw_actions": actions.tolist(), "root_rgb_sha256": advanced["root_rgb_sha256"]}

                center = torch.zeros(20, device="cuda")
                with torch.no_grad():
                    base = {**observe(center), "model": predict(center).tolist()}
                jacobian = torch.autograd.functional.jacobian(predict, center.requires_grad_(True), vectorize=True).detach()
                if jacobian.shape != (192, 20) or not torch.isfinite(jacobian).all():
                    raise RuntimeError("finite full-coordinate model response required")
                measured = []
                for scale in SCALES:
                    endpoints = []
                    for index, plus, minus in probes(scale):
                        record = {"coordinate": index}
                        for sign, value in (("plus", plus), ("minus", minus)):
                            command = torch.as_tensor(value, device="cuda")
                            with torch.no_grad():
                                record[sign] = {**observe(command), "model": predict(command).tolist()}
                        endpoints.append(record)
                    matrices = validate_endpoints(endpoints, scale)
                    measured.append({"scale": scale, "endpoints": endpoints, **{k: v.tolist() for k, v in matrices.items()}})
                cells[name] = {"prefix_observation_exact": True, "prefix_raw_actions": raw.tolist(), "prefix_observed_vector": observed.tolist(),
                    "reconstruction_diagnostics": diagnostics,
                    "post_prefix_qpos": snapshot["qpos"], "target_qpos": snapshot["target_qpos"], "center": base,
                    "model_analytic_response": jacobian.tolist(), "probes": measured}
                print(json.dumps({"seed": seed, "method": name, "prefix_exact": True, "physical_futures": 81}), flush=True)
            common_rgb = target.to(torch.float64) - torch.tensor(cells["zero_replay"]["center"]["observed"], device="cuda", dtype=torch.float64)
            common_qpos = torch.tensor(angular_difference(cells["zero_replay"]["target_qpos"][:2], cells["zero_replay"]["center"]["qpos"]), device="cuda", dtype=torch.float64)
            for cell in cells.values():
                analytic = torch.tensor(cell["model_analytic_response"], device="cuda", dtype=torch.float64)
                cell["common_task_model_analytic_geometry"] = geometry(analytic, common_rgb, physical=True)
                for probe in cell["probes"]:
                    for key, residual in (("model_fd", common_rgb), ("physical_rgb_fd", common_rgb), ("physical_qpos_fd", common_qpos)):
                        response = torch.tensor(probe[key], device="cuda", dtype=torch.float64)
                        probe[key + "_common_task_geometry"] = geometry(response, residual, physical=True)
            row = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "goal_vector": target.tolist(), "common_rgb_residual": common_rgb.tolist(), "common_qpos_residual": common_qpos.tolist(), "methods": cells}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
    finally:
        model.predict = original_predict
        env.close()
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == protocol["root_seeds"],
        "all8_methods_and5184_physical_futures": len(rows) == 8 and all(set(r["methods"]) == set(METHODS) for r in rows),
        "all256_prefix_boundaries_exact": all(c["prefix_observation_exact"] for r in rows for c in r["methods"].values()),
        "parameters_unchanged": shared._parameter_hash(model) == parameters, "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "all_modules_eval": not any(m.training for m in model.modules()), "predict_sample_accounting": samples == 64 * 164,
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"] and digest(initial_path) == protocol["initial_selection_sha256"],
        "prefix_snapshot_reference_unchanged": digest(args.prefix_snapshot_reference) == protocol["prefix_snapshot_reference_sha256"],
        "weights_file_unchanged": digest(args.policy / "weights.pt") == protocol["weights_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_feedback_response_geometry.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]), "gates": gates,
        "actual_model_predict_samples": samples, "physical_futures": 8 * 8 * 81, "rows": rows})
    print(json.dumps({"gates": gates, "actual_model_predict_samples": samples}), flush=True)


if __name__ == "__main__":
    main()
