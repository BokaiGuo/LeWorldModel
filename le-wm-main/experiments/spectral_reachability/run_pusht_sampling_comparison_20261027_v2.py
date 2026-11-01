"""V2: sealed CEM/random planning, whole-stack RGB identity, guarded physics.

V1 sources/results stay immutable. Only its optimizer/CUDA cost/replay helpers
are reused, NOT its physical executor. Calibration only; P1 remains closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027 as v1
from experiments.spectral_reachability import pusht_visible_command_executor as visible
from experiments.spectral_reachability import pusht_rgb_observability_guard as guard
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels

TAG = "pusht_sampling_comparison_20261027_v2"
METHODS, SEEDS = v1.METHODS, v1.SEEDS
write = v1.write


def sha_rgb(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_rgb_context(frames, goal_rgb, reference):
    """Historical field is ONE SHA over stacked uint8 frames, never a list."""
    stack, goal = np.stack(frames), np.asarray(goal_rgb)
    if (stack.shape != (3, 224, 224, 3) or stack.dtype != np.uint8
            or goal.shape != (224, 224, 3) or goal.dtype != np.uint8):
        raise ValueError("actual three-frame uint8 RGB history and goal required")
    for field in ("history_rgb_sha256", "goal_rgb_sha256"):
        value = reference[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("reference must contain single whole-stack/goal SHA strings")
    identity = {"history_rgb_sha256": sha_rgb(stack), "goal_rgb_sha256": sha_rgb(goal)}
    if any(identity[field] != reference[field] for field in identity):
        raise ValueError("physical whole-stack RGB history or goal differs")
    return identity


def official_checked(env, state, goal):
    value = labels(state, goal)
    success, distance = env.unwrapped.eval_state(np.asarray(goal), np.asarray(state))
    if (bool(success) != value["official_success"]
            or not np.isclose(distance, value["official_state_distance"], atol=1e-10, rtol=0)):
        raise ValueError("official labels disagree with installed evaluator")
    return value


def guarded_branch(env, commands, *, root_id, goal, state_of, pose_label):
    """Real shared RGB executor; evaluator labels never influence dispatch.

    Keeps complete/aborted/failed execution receipts, terminal and every
    observed step. A missing initial agent means zero commands, not padding.
    """
    commands = np.asarray(commands)
    if commands.shape != (10, 2) or not np.isfinite(commands).all() or np.any(np.abs(commands) > 1):
        raise ValueError("sealed ten-primitive program required")
    state = state_of(env)
    initial = {"state": state.tolist(), "official": official_checked(env, state, goal),
               "pose": pose_label(state, goal)}
    trace = []
    def record(step, command, transition, rgb):
        state = state_of(env)
        trace.append({"primitive_step": step, "command": command.tolist(),
            "rgb_sha256": sha_rgb(rgb), "state": state.tolist(),
            "reward": float(transition[1]), "terminated": bool(transition[2]),
            "truncated": bool(transition[3]), "official": official_checked(env, state, goal),
            "pose": pose_label(state, goal)})
    result = visible.execute(env, commands, root_id=root_id, agent_rgb=(65,105,225), on_observation=record)
    rgb = result.pop("terminal_rgb")
    if (result["planned_commands"] != 10 or result["completed_commands"] != len(trace)
            or not 0 <= result["completed_commands"] <= result["attempted_commands"] <= 10
            or result["remaining_commands_not_executed"] != 10-len(trace)
            or result["guard"]["root_id"] != root_id
            or result["guard"]["primitive_commands_authorized"] != result["attempted_commands"]
            or (result["status"] == "completed" and len(trace) != 10)
            or any(t["primitive_step"] != i+1 or not np.array_equal(t["command"], commands[i]) for i,t in enumerate(trace))):
        raise ValueError("executor receipt/observed trace differs from sealed prefix")
    state = state_of(env)
    return {**result, "initial": initial, "trace": trace,
        "terminal_rgb_sha256": sha_rgb(rgb), "final_state": state.tolist(),
        "official": official_checked(env, state, goal), "pose": pose_label(state, goal),
        "trace_matches_sealed_prefix": all(np.array_equal(t["command"], commands[i]) for i, t in enumerate(trace)),
        "full_horizon_completed": result["status"] == "completed" and result["completed_commands"] == 10}


def prepare(compilation, policy, output):
    checked, base, replay, metric_for, _, create_seal, _ = v1.dependencies()
    source, bindings, config = checked.validate_inputs(compilation, policy)
    for row in source["rows"]:
        v1.root_context(row["compiled"])
        metric_for(row["compiled"])
    # Import every planning helper before collecting source snapshots.
    from experiments.spectral_reachability import d2p_cuda_sampling_cost
    files = checked.sources_before_allocate(source, config)
    files += [Path(__file__), Path(v1.__file__), Path(v1.kernels.__file__), Path(replay.__file__),
        Path(metric_for.__code__.co_filename), Path(d2p_cuda_sampling_cost.__file__),
        Path(visible.__file__), Path(guard.__file__), Path(labels.__code__.co_filename),
        Path(__file__).resolve().parents[2]/f"tests/test_{TAG}.py",
        Path(__file__).resolve().parents[3]/f"docs/{TAG}.md"]
    files = sorted(set(p.resolve() for p in files))
    p = {"schema": TAG+".protocol.v2", "bindings": bindings,
        "compilation": str(Path(compilation).resolve()), "policy": str(Path(policy).resolve()),
        "roots": source["protocol"]["roots"], "seeds": list(SEEDS), "methods": list(METHODS),
        "context": "root-anchored emb3 + incoming[1:] between2; exactly two new tokens",
        "parameter_box": [-3.,3.], "mapping": "float32 tanh, repeated5 per token",
        "horizon": 2, "microbatch": 16, "candidates_per_root_method": 257,
        "cem": {"samples": 64, "iterations": 4, "elite": 8, "initial_mean": 0.,
                "initial_std": 1., "std_floor": .1, "ddof": 0, "mean_in_population": True, "smoothing": 0},
        "random": {"samples": 256, "proposal": "uniform_parameter_box_not_uniform_live"},
        "selection": "best-ever including explicit zero; stable first tie",
        "objective": "mean((Q @ (terminal - goal0))**2); no control cost",
        "expected_planning_predict_calls": 272, "expected_planning_predict_samples": 4112,
        "expected_verification_predict_calls": 16, "expected_verification_predict_samples": 16,
        "physical": {"executor": "pusht_visible_command_executor.execute", "agent_rgb": [65,105,225],
            "feedback_every_primitive": True, "retain_all_aborts": True, "branches": 8,
            "history_identity": "sha256(np.stack(frames).tobytes()) single string",
            "official_labels_and_pose": True, "model_loads": 0, "neural_encodes": 0,
            "branch_order": "cem then random_shooting; fixed all roots"},
        "scope": "known-root interface/cost calibration; Q can reward agent motion with static object",
        "failure_policy": "planning failure blocks all physics; guarded abort retained, no retry/padding",
        "task_efficacy_repaired": False, "P1_admission": False, "P2_admission": False,
        "P3_admission": False, "official_MPPI_or_GRASP": False, "equal_total_compute": False}
    output = Path(output).resolve()
    if TAG not in output.name or output.parent != Path("results").resolve():
        raise ValueError("new tagged results directory required")
    output.mkdir(exist_ok=False)
    seals = [create_seal(files=files[i:i+40], args_json=json.dumps(p),
        output_dir=output/f"source_seal_{i//40}") for i in range(0, len(files), 40)]
    write(output/"protocol.json", p)
    write(output/"prepared.json", {"protocol_sha256": base.digest(output/"protocol.json"), "seals": seals})
    print(f"V2 prepared without CUDA allocation: {output}", flush=True)


def load_prepared(run):
    checked, base, _, _, _, _, verify_seal = v1.dependencies()
    run = Path(run).resolve()
    prepared = json.loads((run/"prepared.json").read_text())
    if base.digest(run/"protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("protocol changed")
    p = json.loads((run/"protocol.json").read_text())
    if p["schema"] != TAG+".protocol.v2":
        raise ValueError("V2 prepared protocol required; V1 is not accepted")
    for s in prepared["seals"]:
        verify_seal(output_dir=s["output_dir"], expected_manifest_sha256=s["manifest_sha256"])
    source, bindings, _ = checked.validate_inputs(p["compilation"], p["policy"])
    if bindings != p["bindings"]:
        raise ValueError("bindings changed")
    return run, p, source, prepared


def plan(run):
    checked, base, replay, metric_for, Instrumentation, _, _ = v1.dependencies()
    run, p, source, prepared = load_prepared(run)
    write(run/"plan_started.json", {"unix": time.time(), "version": TAG})
    import torch
    torch.manual_seed(SEEDS[0])
    host = base.require_cuda_allocation("cuda")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    model = base.load_compatible_pretrained(Path(p["policy"]), device=device).eval().requires_grad_(False)
    before = checked.tensor_state_hash(model)
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats(device)
    rows, failure = [], None
    try:
        for saved in source["rows"]:
            index, compiled = saved["root_index"], saved["compiled"]
            for method in METHODS:
                name = f"root_{index}.{method}"
                result = {"root_index": index, "root_sha256": saved["root_sha256"], "method": method}
                counter = Instrumentation(model)
                work = {"calls": 0, "samples": 0}
                def hook(module, args):
                    work["calls"] += 1
                    work["samples"] += int(args[0].shape[0])
                handle = model.action_encoder.register_forward_pre_hook(hook)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                try:
                    with (run/(name+".candidates.jsonl")).open("x") as journal, counter:
                        cost = v1.CandidateCost(model, device, compiled, journal, counter, method)
                        with counter.phase("planning", method=method, horizon_tokens=2):
                            params, receipt = v1.optimize(method, cost, SEEDS[index])
                        torch.cuda.synchronize(device)
                        result["optimizer_total_wall_seconds"] = time.perf_counter()-t0
                        result["optimizer_receipt"] = receipt
                        params = v1.checked_params(params)
                        commands = torch.tanh(torch.tensor(params, device=device)).cpu().numpy()
                        with counter.phase("evaluation", method=method, horizon_tokens=2):
                            terminal = replay.CudaReplay(model, device)(v1.root_context(compiled), np.repeat(commands, 5, axis=0))
                        value = metric_for(compiled)(terminal)
                        if abs(value-receipt["best_cost"]) > 1e-5:
                            raise ValueError("independent selected-plan replay mismatch")
                        counts = counter.records
                        if (counts[0]["predict_samples"] != 514 or counts[0]["predict_calls"] != 34
                                or counts[1]["predict_samples"] != 2 or counts[1]["predict_calls"] != 2
                                or not all(r["completed"] for r in counts)):
                            raise ValueError("actual predict count mismatch")
                        witness = {**result, "protocol_sha256": prepared["protocol_sha256"],
                            "params_float32": params.tolist(), "live_commands": commands.tolist(),
                            "root_context": v1.root_context(compiled), "seed": SEEDS[index],
                            "selected_replay_cost": value, "predict": counts, "action_encoder_work": dict(work),
                            "candidate_journal_sha256": base.digest(run/(name+".candidates.jsonl")),
                            "status": "sealed_not_physically_executed", "P1_admission": False}
                        write(run/(name+".plan.json"), witness)
                        result.update(plan=str(run/(name+".plan.json")), plan_sha256=base.digest(run/(name+".plan.json")), status="planned")
                except Exception as exc:
                    result.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                        predict=counter.records, action_encoder_work=work, attempted_wall_seconds=time.perf_counter()-t0)
                    if hasattr(exc, "receipt"):
                        result["optimizer_failure_receipt"] = exc.receipt
                    raise
                finally:
                    handle.remove()
                    write(run/(name+".result.json"), result)
                    rows.append(result)
                print(f"V2 root {index} {method}: sealed; 514 planning + 2 verification samples", flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    gates = {"eight_plans": len(rows) == 8 and all(r["status"] == "planned" for r in rows),
        "state_unchanged": before == checked.tensor_state_hash(model),
        "frozen_eval": all(not m.training for m in model.modules()) and all(not x.requires_grad for x in model.parameters())}
    try:
        load_prepared(run)
        gates["sources_inputs_protocol_unchanged"] = True
    except Exception as exc:
        gates["sources_inputs_protocol_unchanged"] = False
        failure = failure or {"type": type(exc).__name__, "message": str(exc)}
    write(run/"plans_report.json", {"schema": TAG+".plans.v2",
        "status": "plans_sealed" if failure is None and all(gates.values()) else "failed",
        "rows": rows, "failure": failure, "gates": gates, "host": host,
        "setup_load_hash_wall_seconds": setup_seconds, "total_wall_seconds": time.perf_counter()-started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "protocol_sha256": prepared["protocol_sha256"], "P1_admission": False,
        "timing_scope": "optimizer includes proposal/transfer/action_encoder/predict/Q/elite/journal; verification separate from optimizer; total includes both"})
    if failure is not None or not all(gates.values()):
        raise RuntimeError("planning failed; retain partial costs; no physics permitted")


def validate_plans(run, p, source, prepared, report, digest):
    if (report.get("schema") != TAG+".plans.v2" or report.get("status") != "plans_sealed"
            or not report.get("gates") or not all(v is True for v in report["gates"].values())
            or report.get("protocol_sha256") != prepared["protocol_sha256"]):
        raise ValueError("all eight V2 protocol-bound sealed plans required")
    witnesses = []
    for entry in report["rows"]:
        path = Path(entry["plan"])
        if path.parent != run or digest(path) != entry["plan_sha256"]:
            raise ValueError("plan path or hash differs")
        w = json.loads(path.read_text())
        index, method = w["root_index"], w["method"]
        if type(index) is not int or index not in range(4) or method not in METHODS:
            raise ValueError("invalid plan root/method")
        params = v1.checked_params(w["params_float32"])
        commands = np.asarray(w["live_commands"], dtype=np.float32)
        if (w["protocol_sha256"] != prepared["protocol_sha256"]
                or commands.shape != (2,2) or not np.isfinite(commands).all() or np.any(np.abs(commands)>1)
                or not np.allclose(commands, np.tanh(params), atol=2e-7, rtol=2e-7)
                or w["root_sha256"] != p["roots"][index] or w["seed"] != SEEDS[index]
                or (entry["root_index"], entry["root_sha256"], entry["method"]) != (index, w["root_sha256"], method)
                or w["root_context"] != v1.root_context(source["rows"][index]["compiled"])
                or digest(run/f"root_{index}.{method}.candidates.jsonl") != w["candidate_journal_sha256"]):
            raise ValueError("plan/context/action/journal binding mismatch")
        witnesses.append((w, np.repeat(commands, 5, axis=0)))
    if sorted((w["root_index"], w["method"]) for w, _ in witnesses) != sorted((i,m) for i in range(4) for m in METHODS):
        raise ValueError("complete unique four-root/two-method matrix required")
    return witnesses


def execute(run, output):
    checked, base, _, _, _, create_seal, verify_seal = v1.dependencies()
    run, p, source, prepared = load_prepared(run)
    report = json.loads((run/"plans_report.json").read_text())
    witnesses = validate_plans(run, p, source, prepared, report, base.digest)
    candidates = [json.loads(Path(path).read_text()) for path in p["bindings"] if Path(path).name == "report.json"]
    interventions = [r for r in candidates if "bank_path" in r.get("protocol", {})]
    if len(interventions) != 1:
        raise ValueError("unique upstream root source required")
    bank = json.loads(Path(interventions[0]["protocol"]["bank_path"]).read_text())
    upstream = base.physical._load_preflight(Path(bank["upstream_path"]))
    references = {r["root_sha256"]: r for r in upstream["rows"]}
    output = Path(output).resolve()
    if TAG not in output.name or output.parent != Path("results").resolve():
        raise ValueError("new tagged results directory required")
    output.mkdir(exist_ok=False)
    bindings = {str(run/"plans_report.json"): base.digest(run/"plans_report.json")}
    bindings.update({e["plan"]: e["plan_sha256"] for e in report["rows"]})
    protocol = {"plans_bindings": bindings, "protocol_sha256": prepared["protocol_sha256"],
        "physical": p["physical"], "reoptimization": False, "P1_admission": False}
    env = base.gym.make("swm/PushT-v1", render_mode="rgb_array")
    rows, failure = [], None
    try:
        evaluator = Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        files = [Path(__file__).resolve(), Path(v1.__file__).resolve(), Path(visible.__file__).resolve(),
            Path(guard.__file__).resolve(), Path(labels.__code__.co_filename).resolve(), evaluator]
        seal = create_seal(files=list(dict.fromkeys(files)), args_json=json.dumps(protocol), output_dir=output/"execution_source_seal")
        write(output/"protocol.json", protocol)
        if tuple(env.unwrapped.variation_space["agent"]["color"].value.tolist()) != (65,105,225):
            raise ValueError("unsupported agent renderer")
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        if len(roots) != 4 or [base.root_fingerprint(r) for r in roots] != p["roots"]:
            raise ValueError("physical root cohort/order differs")
        for w, commands in witnesses:
            index, key = w["root_index"], w["root_sha256"]
            row = {"root_index": index, "root_sha256": key, "method": w["method"]}
            try:
                root = roots[index]
                frames, history, goal_rgb = base.physical._reset_and_warmup(env, root, history_len=3)
                identity = verify_rgb_context(frames, goal_rgb, references[key])
                identity.update(history=np.asarray(history).tolist(), state=base.p0.packets.state_of(env).tolist())
                peers = [r for r in rows if r["root_index"] == index and "context_identity" in r]
                if peers and identity != peers[0]["context_identity"]:
                    raise ValueError("method reset context differs")
                row["context_identity"] = identity
                row.update(guarded_branch(env, commands, root_id=key, goal=root["goal"],
                    state_of=base.p0.packets.state_of, pose_label=base.physical._pose_label))
            except Exception as exc:
                row.update(status="branch_failed", failure={"type": type(exc).__name__, "message": str(exc)})
            write(output/f"root_{index}.{w['method']}.physical.json", row)
            rows.append(row)
            print(f"V2 physical root {index} {w['method']}: {row['status']}", flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        env.close()
    gates = {"all_eight_retained": len(rows) == 8,
        "no_execution_errors": len(rows) == 8 and all(r["status"] in ("completed", "observation_unavailable", "environment_terminated") for r in rows),
        "trace_matches_sealed_prefix": len(rows) == 8 and all(r.get("trace_matches_sealed_prefix") is True for r in rows),
        "official_and_pose_saved": len(rows) == 8 and all("official" in r and "pose" in r for r in rows),
        "plan_bytes_unchanged": all(base.digest(path) == sha for path, sha in bindings.items())}
    try:
        load_prepared(run)
        verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
        gates["source_inputs_unchanged"] = True
    except Exception as exc:
        gates["source_inputs_unchanged"] = False
        failure = failure or {"type": type(exc).__name__, "message": str(exc)}
    write(output/"report.json", {"schema": TAG+".physical.v2",
        "status": "guarded_calibration_recorded" if failure is None and all(gates.values()) else "integrity_fail",
        "rows": rows, "gates": gates, "failure": failure, "protocol": protocol,
        "completed_full_horizon_branches": sum(r.get("full_horizon_completed") is True for r in rows),
        "aborted_or_failed_branches": sum(r.get("full_horizon_completed") is not True for r in rows),
        "P1_admission": False, "P2_admission": False, "task_efficacy_repaired": False,
        "new_predict_calls": 0, "new_encoder_calls": 0})
    if failure is not None or not all(gates.values()):
        raise RuntimeError("physical integrity failed; all available abort/failure receipts retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    pre = commands.add_parser("prepare")
    pre.add_argument("--compilation", default=v1.DEFAULT_COMPILATION, type=Path)
    pre.add_argument("--policy", default=v1.DEFAULT_POLICY, type=Path)
    pre.add_argument("--output", required=True, type=Path)
    planning = commands.add_parser("plan")
    planning.add_argument("--run", required=True, type=Path)
    physical = commands.add_parser("execute")
    physical.add_argument("--run", required=True, type=Path)
    physical.add_argument("--output", required=True, type=Path)
    args = vars(parser.parse_args())
    phase = args.pop("phase")
    {"prepare": prepare, "plan": plan, "execute": execute}[phase](**args)


if __name__ == "__main__":
    main()
