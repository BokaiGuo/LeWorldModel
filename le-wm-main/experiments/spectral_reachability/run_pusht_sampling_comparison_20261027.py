"""Frozen PushT CEM/random calibration. prepare is CPU-only; plan is CUDA-only.

Q is an interface/cost diagnostic, NOT a repaired object-task objective. No P1
admission, official MPPI/GRASP reproduction, or equal-total-compute claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import d2p_sampling_baselines as kernels

TAG = "pusht_sampling_comparison_20261027"
METHODS = ("cem", "random_shooting")
SEEDS = (202610270, 202610271, 202610272, 202610273)
DEFAULT_COMPILATION = "results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json"
DEFAULT_POLICY = "/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm"


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_params(value):
    a = np.asarray(value)
    if (a.dtype.kind not in "fi" or a.shape[-2:] != (2, 2)
            or a.ndim not in (2, 3) or not a.size
            or not np.isfinite(a).all() or np.any(np.abs(a) > 3)):
        raise ValueError("finite two-token parameters in [-3,3] required")
    return np.array(a, dtype=np.float32, copy=True)


def root_context(compiled):
    initial = compiled["input_context"]
    emb, incoming = np.asarray(initial["emb"]), np.asarray(initial["actions"])
    if (emb.shape != (1, 3, 192) or incoming.shape != (1, 3, 10)
            or not np.isfinite(emb).all() or not np.isfinite(incoming).all()):
        raise ValueError("saved root-anchored emb3/incoming3 required")
    # A0 enters the oldest image; only A1,A2 connect the three saved images.
    return {"layout": "states_with_between_state_actions_v1", "context_length": 3,
            "latent_dim": 192, "action_dim": 10, "latent_history": emb[0].tolist(),
            "action_history": incoming[0, 1:].tolist()}


def recurrence(model, latent, between, tokens, *, cat):
    """Algebra shared with metadata/NumPy tests, production receives CUDA only."""
    if (latent.shape[1] != 3 or between.shape[1] != 2
            or tokens.shape[1] != 2 or latent.shape[0] != between.shape[0]
            or latent.shape[0] != tokens.shape[0]):
        raise ValueError("root emb3 + between2 + exactly two new tokens required")
    for step in range(2):
        outgoing = cat((between, tokens[:, step:step+1]), 1)
        terminal = model.predict(latent, model.action_encoder(outgoing))[:, -1:]
        latent = cat((latent[:, 1:], terminal), 1)
        between = outgoing[:, 1:]
    return latent[:, -1]


def cem(cost, seed):
    """Clipped diagonal Gaussian CEM; best-ever including explicit zero.

    Same NumPy evaluator as random shooting; not the old P0 bootstrap CEM.
    Population std uses ddof=0; elite ties use stable bank order. No smoothing.
    """
    initial = np.zeros((2, 2))
    lower, upper = np.full((2, 2), -3.), np.full((2, 2), 3.)
    evaluator = kernels._Evaluator(cost, initial, lower=lower, upper=upper,
        configuration={"seed": seed, "samples": 64, "iterations": 4, "elite": 8,
                       "initial_std": 1., "std_floor": .1, "ddof": 0},
        algorithm="clipped_diagonal_gaussian_cem",
        contract={"selection": "best_ever", "initial_scored": True,
                  "parameter_to_live": "float32_tanh_then_repeat5", "control_cost": "none"})
    evaluator.score(initial[None])
    mean, std = initial.copy(), np.ones_like(initial)
    rng = np.random.default_rng(seed)
    for _ in range(4):
        bank = np.clip(mean + std*rng.normal(size=(64, 2, 2)), lower, upper)
        bank[0] = mean
        costs = evaluator.score(bank)
        elite = bank[np.argsort(costs, kind="stable")[:8]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), .1)
        evaluator.history.append(evaluator.best_cost)
    return evaluator.result()


def optimize(method, cost, seed):
    if method == "cem":
        return cem(cost, seed)
    if method == "random_shooting":
        return kernels.random_shooting(cost, initial=np.zeros((2, 2)), lower=-3.,
                                       upper=3., samples=256, seed=seed)
    raise ValueError("unknown method")


def dependencies():
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as replay
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    return checked, base, replay, metric_for, PredictInstrumentation, create_seal, verify_seal


def prepare(compilation, policy, output):
    checked, base, replay, metric_for, _, create_seal, _ = dependencies()
    source, bindings, config = checked.validate_inputs(compilation, policy)
    for row in source["rows"]:
        root_context(row["compiled"])
        metric_for(row["compiled"])
    sources = checked.sources_before_allocate(source, config)
    sources += [Path(__file__).resolve(), Path(kernels.__file__).resolve(),
                Path(replay.__file__).resolve(), Path(metric_for.__code__.co_filename).resolve(),
                Path(__file__).with_name("d2p_cuda_sampling_cost.py").resolve(),
                Path(__file__).resolve().parents[2]/"tests/test_pusht_sampling_comparison_20261027.py",
                Path(__file__).resolve().parents[3]/"docs/PUSHT_SAMPLING_COMPARISON_20261027.md"]
    sources = sorted(set(sources))
    protocol = {"schema": TAG+".protocol.v1", "bindings": bindings,
        "compilation": str(Path(compilation).resolve()), "policy": str(Path(policy).resolve()),
        "roots": source["protocol"]["roots"], "seeds": list(SEEDS), "methods": list(METHODS),
        "parameter_box": [-3., 3.], "mapping": "float32 tanh; each command repeated 5 times",
        "horizon": 2, "microbatch": 16, "candidates_per_root_method": 257,
        "cem": {"samples": 64, "iterations": 4, "elite": 8, "initial_std": 1.,
                "std_floor": .1, "ddof": 0, "smoothing": 0, "mean_in_population": True},
        "random": {"samples": 256, "proposal": "uniform_parameter_box_not_uniform_live"},
        "selection": "best_ever_including_zero; stable first tie",
        "objective": "mean((Q @ (predicted_terminal - saved_goal0))**2); no control cost",
        "context": "saved root-anchored emb3; between incoming[1:] only; no bootstrap",
        "expected_planning_predict_samples": 4112, "expected_planning_predict_calls": 272,
        "expected_verification_predict_samples": 16, "expected_verification_predict_calls": 16,
        "planned_physical_branches": 8, "optimization_backward_calls": 0,
        "metric_warning": "Q may respond to agent motion while object stays still; not task efficacy",
        "failure_policy": "log failed candidate batch; stop method; no deployable plan on any failure",
        "P1_admission": False, "P2_admission": False, "P3_admission": False,
        "official_MPPI_or_GRASP": False, "equal_total_compute": False,
        "scope": "four known calibration roots only; no training or held-out claims"}
    output = Path(output).resolve()
    if TAG not in output.name or output.parent != Path("results").resolve():
        raise ValueError("new results directory with task tag required")
    output.mkdir(exist_ok=False)
    # Snapshot sources in bounded groups; do not copy weights or big reports.
    seals = [create_seal(files=sources[i:i+40], args_json=json.dumps(protocol),
                        output_dir=output/f"source_seal_{i//40}") for i in range(0, len(sources), 40)]
    write(output/"protocol.json", protocol)
    write(output/"prepared.json", {"protocol_sha256": base.digest(output/"protocol.json"), "seals": seals})
    print(f"Prepared without CUDA allocation: {output}", flush=True)


def load_prepared(run):
    checked, base, _, _, _, _, verify_seal = dependencies()
    run = Path(run).resolve()
    prepared = json.loads((run/"prepared.json").read_text())
    if base.digest(run/"protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("protocol changed")
    p = json.loads((run/"protocol.json").read_text())
    if p["schema"] != TAG+".protocol.v1":
        raise ValueError("wrong protocol")
    for seal in prepared["seals"]:
        verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
    source, bindings, _ = checked.validate_inputs(p["compilation"], p["policy"])
    if bindings != p["bindings"]:
        raise ValueError("bindings changed")
    return run, p, source, prepared


class CandidateCost:
    """Each submitted candidate survives in JSONL, even failed CUDA batches.

    Independent model.predict wrapper counts actual invocations, not estimates.
    Wall time surrounds *all* optimizer work including sampling/elite updates,
    transfer, CPU Q evaluation and journal I/O, not just GPU kernels.
    """
    def __init__(self, model, device, compiled, journal, counter, method):
        import torch
        from experiments.spectral_reachability.d2p_cuda_sampling_cost import _require_frozen_eval
        if device.type != "cuda":
            raise ValueError("host CUDA only; CPU neural fallback forbidden")
        _require_frozen_eval(model, device)
        if any(b.device != device for b in model.buffers()):
            raise ValueError("CUDA buffers required")
        context = root_context(compiled)
        self.emb = torch.tensor(context["latent_history"], device=device, dtype=torch.float32)[None]
        self.between = torch.tensor(context["action_history"], device=device, dtype=torch.float32)[None]
        self.model, self.device, self.journal = model, device, journal
        self.counter, self.method = counter, method
        self.metric = dependencies()[3](compiled)
        self.next_id = 0

    def __call__(self, candidates):
        import torch
        from experiments.spectral_reachability import preflight_pusht_d2p_p1_reconfiguration as p1
        candidates = checked_params(candidates)
        result = []
        for offset in range(0, len(candidates), 16):
            batch = candidates[offset:offset+16]
            start = time.perf_counter()
            before = sum(r["predict_samples"] for r in self.counter.records)
            costs, error = None, None
            try:
                params = torch.tensor(batch, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    tokens = p1._batched_live_action_tokens(params, action_block=5)
                    terminal = recurrence(self.model, self.emb.expand(len(batch), -1, -1),
                        self.between.expand(len(batch), -1, -1), tokens,
                        cat=lambda arrays, axis: torch.cat(arrays, dim=axis))
                if not torch.isfinite(terminal).all():
                    raise ValueError("nonfinite terminal")
                values = terminal.cpu().numpy().astype(float)
                costs = [self.metric({"latent_history": [v]}) for v in values]
                if not np.isfinite(costs).all():
                    raise ValueError("nonfinite cost")
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                after = sum(r["predict_samples"] for r in self.counter.records)
                elapsed = time.perf_counter()-start
                for j, params_np in enumerate(batch):
                    record = {"candidate_id": self.next_id, "method": self.method,
                        "params_float32": params_np.tolist(),
                        "cost": None if error is not None or costs is None else float(costs[j]),
                        "status": "completed" if error is None and costs is not None else "failed",
                        "failure": error, "microbatch_size": len(batch),
                        "microbatch_predict_samples": after-before,
                        "microbatch_wall_seconds": elapsed,
                        "cost_accounting_note": "microbatch totals repeated per member; do not sum them"}
                    self.journal.write(json.dumps(record, allow_nan=False)+"\n")
                    self.next_id += 1
                self.journal.flush()
            result.extend(costs)
        return np.asarray(result)


def plan(run):
    checked, base, replay, _, Instrumentation, _, _ = dependencies()
    run, p, source, prepared = load_prepared(run)
    # Exclusive claim: a failed/partial run is immutable and requires a new prepare.
    write(run/"plan_started.json", {"unix": time.time()})
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
                encoder_work = {"calls": 0, "samples": 0}
                def hook(module, args):
                    encoder_work["calls"] += 1
                    encoder_work["samples"] += int(args[0].shape[0])
                handle = model.action_encoder.register_forward_pre_hook(hook)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                try:
                    with (run/(name+".candidates.jsonl")).open("x") as journal, counter:
                        cost = CandidateCost(model, device, compiled, journal, counter, method)
                        with counter.phase("planning", method=method, horizon_tokens=2):
                            params, receipt = optimize(method, cost, SEEDS[index])
                        torch.cuda.synchronize(device)
                        result["optimizer_total_wall_seconds"] = time.perf_counter()-t0
                        result["optimizer_receipt"] = receipt
                        params = checked_params(params)
                        commands = torch.tanh(torch.tensor(params, device=device)).cpu().numpy()
                        # Independent existing packet replay verifies selected plan from root.
                        with counter.phase("evaluation", method=method, horizon_tokens=2):
                            terminal = replay.CudaReplay(model, device)(root_context(compiled),
                                np.repeat(commands, 5, axis=0))
                        value = dependencies()[3](compiled)(terminal)
                        if abs(value-receipt["best_cost"]) > 1e-5:
                            raise ValueError("independent selected-plan replay mismatch")
                        counts = counter.records
                        if (counts[0]["predict_samples"] != 514 or counts[0]["predict_calls"] != 34
                                or counts[1]["predict_samples"] != 2 or counts[1]["predict_calls"] != 2
                                or not all(r["completed"] for r in counts)):
                            raise ValueError("actual predict count mismatch")
                        witness = {**result, "protocol_sha256": prepared["protocol_sha256"],
                            "params_float32": params.tolist(), "live_commands": commands.tolist(),
                            "root_context": root_context(compiled), "seed": SEEDS[index],
                            "selected_replay_cost": value, "predict": counts,
                            "action_encoder_work": dict(encoder_work),
                            "candidate_journal_sha256": base.digest(run/(name+".candidates.jsonl")),
                            "status": "sealed_not_physically_executed", "P1_admission": False}
                        write(run/(name+".plan.json"), witness)
                        result.update(plan=str(run/(name+".plan.json")),
                            plan_sha256=base.digest(run/(name+".plan.json")), status="planned")
                except Exception as exc:
                    result.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                                  predict=counter.records, action_encoder_work=encoder_work,
                                  attempted_wall_seconds=time.perf_counter()-t0)
                    if hasattr(exc, "receipt"):
                        result["optimizer_failure_receipt"] = exc.receipt
                    raise
                finally:
                    handle.remove()
                    write(run/(name+".result.json"), result)
                    rows.append(result)
                print(f"root {index} {method}: sealed, 514 planning + 2 verification predict samples", flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    after = checked.tensor_state_hash(model)
    gates = {"eight_plans": len(rows) == 8 and all(r["status"] == "planned" for r in rows),
             "state_unchanged": before == after,
             "frozen_eval": all(not m.training for m in model.modules()) and all(not x.requires_grad for x in model.parameters())}
    try:
        load_prepared(run)
        gates["sources_inputs_protocol_unchanged"] = True
    except Exception as exc:
        gates["sources_inputs_protocol_unchanged"] = False
        failure = failure or {"type": type(exc).__name__, "message": str(exc)}
    write(run/"plans_report.json", {"status": "plans_sealed" if failure is None and all(gates.values()) else "failed",
        "rows": rows, "failure": failure, "gates": gates, "host": host,
        "setup_load_hash_wall_seconds": setup_seconds, "total_wall_seconds": time.perf_counter()-started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "protocol_sha256": prepared["protocol_sha256"], "P1_admission": False,
        "timing_scope": "optimizer includes proposal generation, transfer, action encoder, predict, Q, elite updates, journaling; verification outside optimizer; no FLOP/equal-total-compute claim"})
    if failure is not None or not all(gates.values()):
        raise RuntimeError("planning failed; partial costs/failures retained; no execution allowed")


def execute(run, output):
    """Physics only: no model load, no GPU allocation, no reselection/replanning."""
    checked, base, _, _, _, create_seal, verify_seal = dependencies()
    run, p, source, prepared = load_prepared(run)
    report = json.loads((run/"plans_report.json").read_text())
    if report["status"] != "plans_sealed" or not all(report["gates"].values()):
        raise ValueError("all eight sealed plans required")
    witnesses = []
    for entry in report["rows"]:
        path = Path(entry["plan"])
        if base.digest(path) != entry["plan_sha256"]:
            raise ValueError("plan changed")
        w = json.loads(path.read_text())
        params = checked_params(w["params_float32"])
        commands = np.asarray(w["live_commands"], dtype=np.float32)
        if (w["protocol_sha256"] != prepared["protocol_sha256"]
                or commands.shape != (2, 2) or not np.isfinite(commands).all()
                or np.any(np.abs(commands) > 1)
                or not np.allclose(commands, np.tanh(params), atol=2e-7, rtol=2e-7)
                or w["root_sha256"] != p["roots"][w["root_index"]]
                or w["root_context"] != root_context(source["rows"][w["root_index"]]["compiled"])
                or base.digest(run/f'root_{w["root_index"]}.{w["method"]}.candidates.jsonl') != w["candidate_journal_sha256"]):
            raise ValueError("plan/context/action/journal binding mismatch")
        witnesses.append((w, commands))
    if sorted((w["root_index"], w["method"]) for w, _ in witnesses) != sorted((i, m) for i in range(4) for m in METHODS):
        raise ValueError("complete unique four-root/two-method matrix required")
    candidates = [json.loads(Path(path).read_text()) for path in p["bindings"] if Path(path).name == "report.json"]
    intervention = [r for r in candidates if "bank_path" in r.get("protocol", {})]
    if len(intervention) != 1:
        raise ValueError("unique root source required")
    bank = json.loads(Path(intervention[0]["protocol"]["bank_path"]).read_text())
    upstream = base.physical._load_preflight(Path(bank["upstream_path"]))
    output = Path(output).resolve()
    if TAG not in output.name or output.parent != Path("results").resolve():
        raise ValueError("new tagged results directory required")
    output.mkdir(exist_ok=False)
    bindings = {str(run/"plans_report.json"): base.digest(run/"plans_report.json")}
    bindings.update({e["plan"]: e["plan_sha256"] for e in report["rows"]})
    protocol = {"plans_bindings": bindings, "protocol_sha256": prepared["protocol_sha256"],
                "model_loads": 0, "reoptimization": False, "P1_admission": False}
    seal = create_seal(files=[Path(__file__).resolve(), Path(base.physical.__file__).resolve(),
        Path(base.p0.__file__).resolve(), Path(base.p0.packets.__file__).resolve(),
        Path(base.collect_nonterminal_boundary_contact_roots.__code__.co_filename).resolve()],
        args_json=json.dumps(protocol), output_dir=output/"execution_source_seal")
    write(output/"protocol.json", protocol)
    env = base.gym.make("swm/PushT-v1", render_mode="rgb_array")
    rows = []
    try:
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4,
            seed=int(upstream["protocol"]["root_seed"]), oversample_factor=4)
        for w, commands in witnesses:
            index = w["root_index"]
            row = {"root_index": index, "root_sha256": w["root_sha256"], "method": w["method"], "stages": [], "steps": []}
            try:
                root = roots[index]
                if base.root_fingerprint(root) != w["root_sha256"]:
                    raise ValueError("physical root mismatch")
                frames, history, goal_rgb = base.physical._reset_and_warmup(env, root, history_len=3)
                if base.physical._sha256(goal_rgb) != upstream["rows"][index]["goal_rgb_sha256"]:
                    raise ValueError("physical goal mismatch")
                if [base.physical._sha256(f) for f in frames] != upstream["rows"][index]["history_rgb_sha256"]:
                    raise ValueError("physical root-anchored RGB context mismatch")
                # Compare replayed actual history across both method resets.
                identity = {"frames": [base.physical._sha256(f) for f in frames], "history": np.asarray(history).tolist(),
                            "state": base.p0.packets.state_of(env).tolist()}
                peers = [r for r in rows if r["root_index"] == index and "context_identity" in r]
                if peers and identity != peers[0]["context_identity"]:
                    raise ValueError("method reset context differs")
                row["context_identity"] = identity
                row["initial_pose"] = base.physical._pose_label(np.asarray(identity["state"]), root["goal"])
                def on_step(result):
                    row["steps"].append({"state": base.p0.packets.state_of(env).tolist(),
                        "reward": float(result[1]), "terminated": bool(result[2]), "truncated": bool(result[3])})
                for command in commands:
                    base.p0.packets.execute_packet(env, base.p0.packets.repeat_command(command), on_step=on_step)
                    state = base.p0.packets.state_of(env)
                    row["stages"].append({"state": state.tolist(), "pose": base.physical._pose_label(state, root["goal"]),
                        "rgb_sha256": base.physical._sha256(np.asarray(env.render(), dtype=np.uint8))})
                row["status"] = "executed"
            except Exception as exc:
                row.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            write(output/f'root_{index}.{w["method"]}.physical.json', row)
            rows.append(row)
    finally:
        env.close()
    gates = {"eight_executed": len(rows) == 8 and all(r["status"] == "executed" for r in rows),
             "plan_bytes_unchanged": all(base.digest(path) == sha for path, sha in bindings.items())}
    try:
        load_prepared(run)
        verify_seal(output_dir=seal["output_dir"], expected_manifest_sha256=seal["manifest_sha256"])
        gates["source_inputs_unchanged"] = True
    except Exception as exc:
        gates["source_inputs_unchanged"] = False
        write(output/"integrity_failure.json", {"message": str(exc)})
    write(output/"report.json", {"status": "calibration_executed" if all(gates.values()) else "failed",
        "rows": rows, "gates": gates, "protocol": protocol, "P1_admission": False,
        "task_efficacy_repaired": False, "new_predict_calls": 0, "new_encoder_calls": 0})
    if not all(gates.values()):
        raise RuntimeError("physical calibration failure; all branch outcomes retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    pre = commands.add_parser("prepare")
    pre.add_argument("--compilation", default=DEFAULT_COMPILATION, type=Path)
    pre.add_argument("--policy", default=DEFAULT_POLICY, type=Path)
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
