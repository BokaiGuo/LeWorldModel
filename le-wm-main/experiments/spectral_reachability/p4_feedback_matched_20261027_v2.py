"""Equal-round-budget imagined versus actual feedback; engineering calibration only.

CLI is frozen host CUDA only. Unit tests inject explicitly non-neural algebra.
No stage admission, tuning, training, new roots or outcome-based action selection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

TAG = "p4_feedback_matched_20261027_v2"
ARMS = ("imagined", "actual")
SEED = 202610274


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rgb_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def paths_for(root_index):
    if type(root_index) is not int or root_index not in range(4):
        raise ValueError("known four root indices only")
    pairs = [(i,j) for i in range(16) for j in range(16)]
    rng = np.random.default_rng(SEED+root_index)
    return [pairs[int(i)] for i in rng.permutation(256)[:80]]


def validate_bank(bank):
    a = np.asarray(bank)
    if (a.dtype.kind not in "fi" or a.shape != (16,5,2)
            or not np.isfinite(a).all() or np.any(np.abs(a)>1)):
        raise ValueError("complete finite legal P2 bank16 of five-step packets required")
    return a.copy()


def advance_observed(frames, between, terminal_rgb, executed_packet, execution):
    """No imagined latents, reset, second bootstrap or repeated action shift."""
    if (execution["status"] != "completed" or execution["completed_commands"] != 5
            or execution["remaining_commands_not_executed"] != 0):
        raise ValueError("partial/aborted packet cannot supply next planning boundary")
    if (np.shape(frames) != (3,224,224,3) or np.shape(between) != (2,5,2)
            or np.shape(terminal_rgb) != (224,224,3) or np.shape(executed_packet) != (5,2)):
        raise ValueError("three actual RGB boundaries and two between packets required")
    if any(np.asarray(f).dtype != np.uint8 for f in [*frames,terminal_rgb]):
        raise ValueError("uint8 RGB required")
    return ([np.array(f,copy=True) for f in [frames[1],frames[2],terminal_rgb]],
            np.stack([between[1],executed_packet]).copy())


def select_pairs(*, context, bank, paths, replay, metric, journal):
    """Evaluate all scheduled full programs; preserve failures and refuse selection."""
    best = None
    failed = False
    for path in paths:
        start = time.perf_counter()
        event = {"path":list(path),"completed":False}
        try:
            actions = np.concatenate([bank[i] for i in path])
            terminal = replay(deepcopy(context), actions, "planning")
            cost = float(metric(terminal))
            if not np.isfinite(cost):
                raise ValueError("nonfinite metric")
            event.update(completed=True,cost=cost)
            candidate = (cost,tuple(path),terminal)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        except Exception as exc:
            event.update(error_type=type(exc).__name__,error=str(exc))
            failed = True
        finally:
            event["wall_seconds"] = time.perf_counter()-start
            journal(event)
    if failed or best is None:
        raise RuntimeError("candidate failures retained; no deployable selection")
    cost, path, terminal = best
    return {"path":list(path),"cost":cost,"terminal":terminal,
            "actions":np.concatenate([bank[i] for i in path]).tolist()}


def branch_loop(*, arm, root_index, bank, frames, between, encode, replay, metric,
                execute, seal_decision, journal):
    """Same schedules/horizon/verification in both arms; only feedback differs.

    Both encode actual RGB at each boundary, but imagined round1 selects using
    the saved model rollout of round0's executed packet. Physics labels and
    guard states never reach metric/replay. A partial packet ends the branch.
    """
    if arm not in ARMS:
        raise ValueError("explicit arm required")
    bank = validate_bank(bank)
    paths = paths_for(root_index)
    rounds, previous, imagined = [], None, None
    for iteration in range(2):
        start = time.perf_counter()
        observed = encode(frames, between)
        planning = imagined if arm == "imagined" and iteration == 1 else observed
        schedule = paths[iteration*40:(iteration+1)*40]
        selected = select_pairs(context=planning, bank=bank, paths=schedule,
            replay=replay, metric=metric, journal=lambda e:journal(iteration,e))
        # Identical 2+1+1-token verification in each arm at each round.
        full = replay(deepcopy(planning), np.asarray(selected["actions"]), "evaluation")
        mid = replay(deepcopy(planning), bank[selected["path"][0]], "evaluation")
        end = replay(deepcopy(mid), bank[selected["path"][1]], "evaluation")
        if full != selected["terminal"] or end != full:
            raise ValueError("selected plan independent/segmented replay mismatch")
        packet = bank[selected["path"][0]].copy()
        decision = {"arm":arm, "round":iteration, "root_index":root_index,
            "observed":{"frame_sha256":[rgb_sha(f) for f in frames],
                "stack_sha256":rgb_sha(np.stack(frames)),
                "between_raw_packets":np.asarray(between).tolist(),
                "encoded_context":observed, "context_sha256":content_sha(observed)},
            "planning_context":planning, "planning_context_sha256":content_sha(planning),
            "observed_context_used_for_selection":not (arm=="imagined" and iteration==1),
            "schedule":schedule, "search_horizon_tokens":2,
            "selected":selected, "executed_bank_index":selected["path"][0],
            "executed_packet":packet.tolist(), "model_next_context":mid,
            "replay_exact":[True,True], "previous_decision_sha256":previous,
            "planning_encode_selection_replay_wall_seconds":time.perf_counter()-start}
        previous = seal_decision(iteration,decision)
        execution, terminal = execute(packet,iteration)
        rounds.append({"round":iteration,"decision_sha256":previous,"execution":execution})
        complete = (execution["status"]=="completed" and execution["completed_commands"]==5
                    and execution["remaining_commands_not_executed"]==0)
        if not complete:
            break
        if iteration == 0:
            imagined = deepcopy(mid)
            frames,between = advance_observed(frames,between,terminal,packet,execution)
    return {"arm":arm, "rounds":rounds,
        "status":"two_rounds_completed" if len(rounds)==2 and all(
            r["execution"]["status"]=="completed" and r["execution"]["completed_commands"]==5
            and r["execution"]["remaining_commands_not_executed"]==0 for r in rounds) else "guarded_abort",
        "physics_steps":sum(r["execution"]["completed_commands"] for r in rounds)}


def execute_packet(*, env, packet, iteration, root_id, prefix, executor, state_of):
    """Persist all guarded trajectories before validation or final labels.

    executor is the existing live RGB guard integration; tests inject a
    non-neural callback fake. No outcome score is computed in this path.
    """
    trace=[]
    attempts=[]
    class AttemptJournalEnv:
        def render(self):
            return env.render()

        def step(self, command):
            attempt={"primitive_step":iteration*5+len(attempts)+1,
                "command":np.asarray(command).tolist(),
                "status":"dispatch_authorized_before_env_step"}
            write(Path(str(prefix)+f".attempt_{len(attempts)+1}.json"),attempt)
            attempts.append(attempt)
            return env.step(command)
    def record(step,command,transition,rgb):
        event={"primitive_step":iteration*5+step,"command":command.tolist(),
            "state":state_of(env).tolist(),"rgb_sha256":rgb_sha(rgb),
            "reward":float(transition[1]),"terminated":bool(transition[2]),
            "truncated":bool(transition[3])}
        trace.append(event)
        write(Path(str(prefix)+f".step_{step}.json"),event)
    value=executor(AttemptJournalEnv(),packet,root_id=root_id,agent_rgb=(65,105,225),on_observation=record)
    terminal=value.pop("terminal_rgb")
    value.update(trace=trace,attempt_before_step=attempts,terminal_rgb_sha256=rgb_sha(terminal))
    write(Path(str(prefix)+".execution.json"),value)
    with Path(str(prefix)+".terminal.npy").open("xb") as stream:
        np.save(stream,terminal,allow_pickle=False)
    if (len(trace)!=value["completed_commands"] or value["planned_commands"]!=5
            or (value["status"]=="completed" and len(trace)!=5)
            or any(not np.array_equal(t["command"],packet[j]) for j,t in enumerate(trace))):
        raise ValueError("executor trace differs from sealed packet")
    return value,terminal


def encode_cuda(model,device,frames,between,p0,transform):
    if getattr(device,"type",None) != "cuda":
        raise ValueError("host CUDA only; CPU neural fallback forbidden")
    import torch
    if np.shape(between)!=(2,5,2):
        raise ValueError("exactly two between-state raw packets required")
    with torch.no_grad():
        pixels = p0._frames_tensor(frames,transform=transform,device=device)
        emb = model.encode({"pixels":pixels})["emb"]
        tokens = p0.packets.normalize_live_packet(torch.tensor(np.array(between,copy=True),dtype=torch.float32,device=device),
            mean=p0.PUBLIC_ACTION_MEAN,std=p0.PUBLIC_ACTION_STD)
        if emb.device != device or emb.shape != (1,3,192) or not torch.isfinite(emb).all():
            raise ValueError("finite CUDA root-anchored emb3 required")
        return {"layout":"states_with_between_state_actions_v1","context_length":3,
            "latent_dim":192,"action_dim":10,"latent_history":emb[0].cpu().tolist(),
            "action_history":tokens.cpu().tolist()}


def materialize_blocked(output,rows,roots,setup_failure):
    """Retain all scheduled identities without inventing attempted work."""
    present={(r["root_index"],r["arm"]) for r in rows}
    for index,key in enumerate(roots):
        for arm in ARMS:
            if (index,arm) not in present:
                row={"root_index":index,"root_sha256":key,"arm":arm,
                    "status":"blocked_setup_failure","setup_failure":setup_failure,
                    "rounds":[],"physics_steps":0,"predict":[],
                    "retained_executions":[],"retained_decisions":[]}
                write(Path(output)/f"root_{index}.{arm}.json",row)
                rows.append(row)


def round_accounting_valid(row):
    """Audit actual intercepted calls, not just caller-declared token budgets."""
    rounds = row.get("rounds", [])
    if len(rounds) not in (1, 2) or [r["round"] for r in rounds] != list(range(len(rounds))):
        return False
    for i,r in enumerate(rounds):
        execution=r["execution"]
        steps=execution["completed_commands"]
        if (type(steps) is not int or not 0 <= steps <= 5
                or execution["remaining_commands_not_executed"] != 5-steps
                or i < len(rounds)-1 and (execution["status"] != "completed" or steps != 5)):
            return False
    if row.get("status")=="guarded_abort" and rounds[-1]["execution"]["status"] not in (
            "observation_unavailable","environment_terminated"):
        return False
    if row.get("status") == "two_rounds_completed" and (
            len(rounds) != 2 or any(r["execution"]["status"] != "completed"
                or r["execution"]["completed_commands"] != 5
                or r["execution"]["remaining_commands_not_executed"] != 0 for r in rounds)):
        return False
    records = row.get("predict", [])
    if len(records) != 43 * len(rounds):
        return False
    for iteration in range(len(rounds)):
        group = [r for r in records if r.get("round") == iteration]
        for phase, horizons in (("planning", [2]*40), ("evaluation", [2,1,1])):
            subset = [r for r in group if r["phase"] == phase]
            if [r["horizon_tokens"] for r in subset] != horizons:
                return False
            if any(not r["completed"] or r["failed_predict_calls"] != 0
                   or r["predict_calls"] != h or r["predict_samples"] != h
                   or r["context_lengths"] != {3:h} and r["context_lengths"] != {"3":h}
                   or r["method"] != row["arm"] for r,h in zip(subset,horizons)):
                return False
    n = len(rounds)
    return (row.get("predict_planning_calls") == row.get("predict_planning_samples") == 80*n
        and row.get("predict_evaluation_calls") == row.get("predict_evaluation_samples") == 4*n
        and row.get("image_encode_calls") == n and row.get("image_encode_frames") == 3*n
        and row.get("action_encoder_work") == {"calls":84*n,"samples":84*n}
        and row.get("physics_steps") == sum(r["execution"]["completed_commands"] for r in rounds))


def finalize(output,rows,bindings,verify_sources,state_check,protocol,resources):
    failures=[]
    gates={"eight_branches_retained":len(rows)==8 and {(r["root_index"],r["arm"]) for r in rows}=={(i,a) for i in range(4) for a in ARMS},
        "no_runtime_errors":len(rows)==8 and all(r.get("status") in ("two_rounds_completed","guarded_abort")
            and all(t["execution"]["status"] in ("completed","observation_unavailable","environment_terminated") for t in r.get("rounds",[])) for r in rows),
        "setup_and_close_succeeded":resources.get("setup_failure") is None}
    for name, check in (("inputs_unchanged",lambda:all(digest(Path(p))==s for p,s in bindings.items())),
                        ("sources_unchanged",verify_sources),("frozen_model_unchanged",state_check)):
        try:
            gates[name] = check() is True
        except Exception as exc:
            gates[name]=False
            failures.append({"check":name,"type":type(exc).__name__,"message":str(exc)})
    complete=[r for r in rows if r.get("status")=="two_rounds_completed"]
    gates["per_round_actual_accounting"] = all(round_accounting_valid(r) for r in rows)
    gates["completed_branches_actual_budget"] = all(r.get("predict_planning_samples")==160 and r.get("predict_evaluation_samples")==8
        and r.get("predict_planning_calls")==160 and r.get("predict_evaluation_calls")==8
        and r.get("action_encoder_work")=={"calls":168,"samples":168}
        and r.get("image_encode_frames")==6 and r.get("image_encode_calls")==2
        and all(p["completed"] and p["failed_predict_calls"]==0 for p in r.get("predict",[])) for r in complete)
    pairs=[]
    for index in range(4):
        paired={r["arm"]:r for r in rows if r["root_index"]==index}
        pairs.append({"root_index":index,"complete_pair":set(paired)==set(ARMS) and all(r.get("status")=="two_rounds_completed" for r in paired.values()),
            "arms":{a:{k:r.get(k) for k in ("status","physics_steps","predict_planning_samples","predict_evaluation_samples","image_encode_frames","final_official","final_pose")} for a,r in paired.items()}})
    report={"schema":TAG,"status":"engineering_calibration_recorded" if all(gates.values()) else "integrity_fail",
        "scientific_status":"blocked_upstream","P1_admission":False,"P4_admission":False,
        "protocol":protocol,"gates":gates,"failures":failures,"rows":rows,"root_pairs":pairs,"resources":resources,
        "full_two_round_branches":len(complete),"all_branches_complete":len(complete)==8,
        "equal_round_budget_observed_all_pairs":len(complete)==8 and gates["per_round_actual_accounting"],
        "equal_total_compute":False,"efficacy_verified":False}
    write(Path(output)/"report.json",report)
    return report


def run(compilation,policy,output):
    # All code resolution and seals precede the first CUDA allocation/model load.
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    from experiments.spectral_reachability import pusht_visible_command_executor as visible
    from experiments.spectral_reachability import pusht_rgb_observability_guard as guard
    from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027_v2 as identity
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    source,bindings,config=checked.validate_inputs(compilation,policy)
    candidates=[json.loads(Path(p).read_text()) for p in bindings if Path(p).name=="report.json"]
    interventions=[r for r in candidates if "bank_path" in r.get("protocol",{})]
    if len(interventions)!=1:
        raise ValueError("unique known calibration source required")
    bank_report=json.loads(Path(interventions[0]["protocol"]["bank_path"]).read_text())
    upstream=base.physical._load_preflight(Path(bank_report["upstream_path"]))
    refs={r["root_sha256"]:r for r in upstream["rows"]}
    banks=[validate_bank([np.repeat(np.asarray(r["program_live_commands"][0])[None],5,axis=0)
        for r in row["compiled"]["receipts"]]) for row in source["rows"]]
    expected_roots=(
        "9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333",
        "23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050",
        "4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca",
        "cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2")
    if tuple(source["protocol"]["roots"]) != expected_roots:
        raise ValueError("original four known roots/order required")
    output=Path(output).resolve()
    if TAG not in output.name or output.parent!=Path("results").resolve():
        raise ValueError("new results directory with task tag required")
    output.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[3]
    for p in (repo/"docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md",repo/"docs/P45_PREREQUISITE_SNAPSHOT_20261027.md"):
        bindings[str(p)]=digest(p)
    protocol={"schema":TAG,"scope":"engineering_calibration_only","scientific_status":"blocked_upstream",
        "bindings":deepcopy(bindings),"roots":source["protocol"]["roots"],"seed":SEED,"arms":list(ARMS),
        "paths_per_root":[paths_for(i) for i in range(4)],"banks":[b.tolist() for b in banks],
        "rounds":2,"executed_steps_per_round":5,"search_horizon_tokens":2,
        "per_round":{"candidate_pairs":40,"planning_predict_calls":80,"planning_predict_samples":80,
            "verification_predict_calls":4,"verification_predict_samples":4,
            "rgb_encode_calls":1,"rgb_encode_frames":3},
        "imagined":"40 pairs per round; second boundary is model rollout of first executed packet",
        "actual":"same40 pairs per round; second boundary is actual RGB plus executed packet history",
        "per_unaborted_arm":{"planning_predict_calls":160,"planning_predict_samples":160,
            "verification_predict_calls":8,"verification_predict_samples":8,"rgb_encode_calls":2,"rgb_encode_frames":6},
        "feedback":"identical per-round schedule, horizon, bank, goal/Q and verification; both encode real3, imagined ignores second real encoding for selection",
        "goal":"saved goal0 and Q, extra goal-image-set provenance unchanged; not single-image metric learning",
        "Q_warning":"agent motion with static object can improve metric; no task efficacy repair",
        "stopping":"two packets or first guard/termination/error; no padding/retries; retain partial resource differences",
        "budget_claim":"scheduled actual predictor samples/calls matched only when both complete; no FLOP/wall-cap/utility match",
        "P1_admission":False,"P4_admission":False,"readout_fit":False,"training":False}
    protocol["cuda_matmul_allow_tf32"]=False
    files=checked.sources_before_allocate(source,config)
    files += [Path(__file__),Path(adapter.__file__),Path(visible.__file__),Path(guard.__file__),Path(identity.__file__), Path(__file__).with_name("p4_receding_runtime_20261027_v1.py"),
        Path(metric_for.__code__.co_filename),Path(__file__).resolve().parents[2]/f"tests/test_{TAG}.py",repo/f"docs/{TAG}.md"]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+40],args_json=json.dumps(protocol),output_dir=output/f"source_seal_{i//40}") for i in range(0,len(files),40)]
    write(output/"protocol.json",protocol)
    protocol_sha=digest(output/"protocol.json")
    bindings[str(output/"protocol.json")]=protocol_sha
    write(output/"branch_roster.json",[
        {"root_index":i,"root_sha256":key,"arm":arm,"status":"pending_setup"}
        for i,key in enumerate(protocol["roots"]) for arm in ARMS])
    host=base.require_cuda_allocation("cuda")
    device=torch.device("cuda:0")
    started=time.perf_counter()
    model=base.load_compatible_pretrained(Path(policy),device=device).eval().requires_grad_(False)
    before=checked.tensor_state_hash(model)
    replay_impl=adapter.CudaReplay(model,device)
    transform=base.p0.image_transform()
    torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    env=None
    setup_failure=None
    try:
        env=base.gym.make("swm/PushT-v1",render_mode="rgb_array")
        evaluator=Path(env.unwrapped.eval_state.__code__.co_filename).resolve()
        extra=create_seal(files=[evaluator,Path(identity.labels.__code__.co_filename).resolve()],args_json=json.dumps({"protocol_sha256":protocol_sha}),output_dir=output/"evaluator_seal")
        seals.append(extra)
        if tuple(env.unwrapped.variation_space["agent"]["color"].value.tolist())!=(65,105,225):
            raise ValueError("unsupported renderer")
        roots=base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream["protocol"]["root_seed"]),oversample_factor=4)
        if len(roots)!=4 or [base.root_fingerprint(r) for r in roots]!=protocol["roots"]:
            raise ValueError("root cohort/order differs")
        for index,root in enumerate(roots):
            key=protocol["roots"][index]
            for arm in (ARMS if index%2==0 else tuple(reversed(ARMS))):
                row={"root_index":index,"root_sha256":key,"arm":arm}
                counters=PredictInstrumentation(model)
                events=[]
                image_work={"calls":0,"frames":0}
                action_work={"calls":0,"samples":0}
                def action_hook(module,args):
                    action_work["calls"]+=1
                    action_work["samples"]+=int(args[0].shape[0])
                handle=model.action_encoder.register_forward_pre_hook(action_hook)
                t0=time.perf_counter()
                try:
                    frames,history,goal_rgb=base.physical._reset_and_warmup(env,root,history_len=3)
                    original_identity=identity.verify_rgb_context(frames,goal_rgb,refs[key])
                    row["initial_rgb_identity"]=original_identity
                    between=np.repeat(np.asarray(history)[1:,None,:],5,axis=1)
                    metric=metric_for(source["rows"][index]["compiled"])
                    def encode(frames,between):
                        start=time.perf_counter()
                        event={"kind":"rgb_encode","completed":False,"frames":3}
                        events.append(event)
                        image_work["calls"]+=1
                        image_work["frames"]+=3
                        prefix=output/f"root_{index}.{arm}.round_{image_work['calls']-1}"
                        for suffix,array in (("rgb",np.stack(frames)),("between",np.asarray(between))):
                            with Path(str(prefix)+f".{suffix}.npy").open("xb") as stream:
                                np.save(stream,array,allow_pickle=False)
                        try:
                            context=encode_cuda(model,device,frames,between,base.p0,transform)
                            event["completed"]=True
                            return context
                        finally:
                            torch.cuda.synchronize(device)
                            event["wall_seconds"]=time.perf_counter()-start
                    def replay(context,actions,phase):
                        start=time.perf_counter()
                        event={"kind":"predict_dispatch","round":image_work["calls"]-1,"phase":phase,"tokens":len(actions)//5,"completed":False}
                        events.append(event)
                        try:
                            with counters.phase(phase,method=arm,horizon_tokens=len(actions)//5) as receipt:
                                receipt["round"]=image_work["calls"]-1
                                value=replay_impl(context,actions)
                            torch.cuda.synchronize(device)
                            event["completed"]=True
                            return value
                        finally:
                            event["wall_seconds"]=time.perf_counter()-start
                    def seal_decision(iteration,decision):
                        decision.update(root_sha256=key,goal_rgb_sha256=original_identity["goal_rgb_sha256"],
                            protocol_sha256=protocol_sha,source_manifest_sha256=[s["manifest_sha256"] for s in seals])
                        path=output/f"root_{index}.{arm}.round_{iteration}.decision.json"
                        write(path,decision)
                        sha=digest(path)
                        bindings[str(path)]=sha
                        return sha
                    def execute(packet,iteration):
                        return execute_packet(env=env,packet=packet,iteration=iteration,
                            root_id=key,prefix=output/f"root_{index}.{arm}.round_{iteration}",
                            executor=visible.execute,state_of=base.p0.packets.state_of)
                    with (output/f"root_{index}.{arm}.candidates.jsonl").open("x") as journal,counters:
                        def log(iteration,event):
                            journal.write(json.dumps({"round":iteration,**event},allow_nan=False)+"\n")
                            journal.flush()
                            os.fsync(journal.fileno())
                        result=branch_loop(arm=arm,root_index=index,bank=banks[index],frames=frames,between=between,
                            encode=encode,replay=replay,metric=metric,execute=execute,seal_decision=seal_decision,journal=log)
                    row.update(result)
                except Exception as exc:
                    row.update(status="runtime_error",error_type=type(exc).__name__,error=str(exc))
                finally:
                    handle.remove()
                    row["retained_executions"]=[str(p) for p in sorted(output.glob(f"root_{index}.{arm}.round_*.execution.json"))]
                    row["retained_decisions"]=[str(p) for p in sorted(output.glob(f"root_{index}.{arm}.round_*.decision.json"))]
                    row.update(predict=counters.records,resource_events=events,image_encode_calls=image_work["calls"],image_encode_frames=image_work["frames"],
                        action_encoder_work=action_work,wall_seconds=time.perf_counter()-t0)
                    for phase in ("planning","evaluation"):
                        row[f"predict_{phase}_samples"]=sum(r["predict_samples"] for r in counters.records if r["phase"]==phase)
                        row[f"predict_{phase}_calls"]=sum(r["predict_calls"] for r in counters.records if r["phase"]==phase)
                    try:
                        state=base.p0.packets.state_of(env)
                        row.update(final_official=identity.official_checked(env,state,root["goal"]),final_pose=base.physical._pose_label(state,root["goal"]))
                    except Exception as exc:
                        row.update(status="runtime_error",final_label_error=str(exc))
                    write(output/f"root_{index}.{arm}.json",row)
                    rows.append(row)
                print(f"P4 engineering root {index} {arm}: {row['status']}",flush=True)
    except Exception as exc:
        setup_failure={"type":type(exc).__name__,"message":str(exc)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                setup_failure=setup_failure or {"type":type(exc).__name__,"message":str(exc)}
    materialize_blocked(output,rows,protocol["roots"],setup_failure)
    def verify_sources():
        for s in seals:
            verify_seal(output_dir=s["output_dir"],expected_manifest_sha256=s["manifest_sha256"])
        return True
    report=finalize(output,rows,bindings,verify_sources,
        lambda:before==checked.tensor_state_hash(model) and all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        protocol,{"host":host,"total_wall_seconds":time.perf_counter()-started,"setup_failure":setup_failure,
            "source_seals":seals,
            "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(device),
            "backward_calls":0,"optimizer_steps":0,"goal_encode_calls":0,
            "not_measured":["FLOPs","joules","upstream compilation/goal-set encoding costs"],"wall_time_cap_enforced":False})
    if report["status"]!="engineering_calibration_recorded" or setup_failure is not None:
        raise RuntimeError("engineering runtime integrity failure; report/partial branches retained")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compilation",type=Path,default="results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json")
    parser.add_argument("--policy",type=Path,default="/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
    parser.add_argument("--output",type=Path,required=True)
    run(**vars(parser.parse_args()))

