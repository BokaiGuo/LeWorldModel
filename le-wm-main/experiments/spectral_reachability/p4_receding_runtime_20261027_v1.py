"""Two-decision PushT engineering calibration, NOT P4 scientific admission.

Fixed uses 80 initial pairs; receding uses 40 pairs at each real boundary.
Scheduled predictor budget matches, not total compute or an isolated feedback
ablation. Only CLI neural path is frozen host CUDA; CPU tests inject algebra.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

TAG = "p4_receding_runtime_20261027_v1"
ARMS = ("receding", "fixed")
SEED = 202610274


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


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
    """Callback orchestrator; tests use non-neural fakes, CLI uses CUDA only.

    seal_decision must persist before returning. execute returns raw terminal RGB
    separately from a JSON receipt. All context encodings use observed images.
    """
    if arm not in ARMS:
        raise ValueError("explicit arm required")
    bank = validate_bank(bank)
    paths = paths_for(root_index)
    rounds, original = [], None
    previous = None
    frozen = None
    for iteration in range(2):
        start = time.perf_counter()
        observed = encode(frames,between)
        if original is None:
            original = deepcopy(observed)
        input_receipt = {"frame_sha256":[rgb_sha(f) for f in frames],
            "stack_sha256":rgb_sha(np.stack(frames)),"between_raw_packets":np.asarray(between).tolist(),
            "encoded_context":observed,"context_sha256":content_sha(observed)}
        verification = []
        if arm == "receding" or iteration == 0:
            schedule = paths[iteration*40:(iteration+1)*40] if arm == "receding" else paths
            selected = select_pairs(context=observed,bank=bank,paths=schedule,replay=replay,
                metric=metric,journal=lambda event:journal(iteration,event))
            # Receding: full2 each round. Fixed: segmented2 + full2 once.
            full = replay(deepcopy(observed),np.asarray(selected["actions"]),"evaluation")
            verification.append(full == selected["terminal"])
            if arm == "fixed":
                mid = replay(deepcopy(observed),bank[selected["path"][0]],"evaluation")
                end = replay(mid,bank[selected["path"][1]],"evaluation")
                verification.append(end == full)
                frozen = deepcopy(selected)
            if not all(verification):
                raise ValueError("selected plan independent replay mismatch")
        else:
            selected = deepcopy(frozen)
        token_index = selected["path"][0 if arm == "receding" else iteration]
        packet = bank[token_index]
        decision = {"arm":arm,"round":iteration,"root_index":root_index,
            "observed":input_receipt,"planning_context_sha256":content_sha(observed if arm=="receding" else original),
            "observed_context_used_for_selection":arm=="receding" or iteration==0,
            "selected":selected,"executed_bank_index":token_index,"executed_packet":packet.tolist(),
            "replay_exact":verification,"previous_decision_sha256":previous,
            "planning_encode_selection_replay_wall_seconds":time.perf_counter()-start}
        previous = seal_decision(iteration,decision)
        execution, terminal = execute(packet,iteration)
        rounds.append({"round":iteration,"decision_sha256":previous,"execution":execution})
        if execution["status"] != "completed" or execution["completed_commands"] != 5:
            break
        if iteration == 0:
            frames,between = advance_observed(frames,between,terminal,packet,execution)
    return {"arm":arm,"rounds":rounds,"status":"two_rounds_completed" if len(rounds)==2 and all(
        r["execution"]["status"]=="completed" and r["execution"]["completed_commands"]==5 for r in rounds) else "guarded_abort",
        "physics_steps":sum(r["execution"]["completed_commands"] for r in rounds)}


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
    gates["completed_branches_actual_budget"] = all(r.get("predict_planning_samples")==160 and r.get("predict_evaluation_samples")==4
        and r.get("predict_planning_calls")==160 and r.get("predict_evaluation_calls")==4
        and r.get("action_encoder_work")=={"calls":164,"samples":164}
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
        "equal_total_compute":False,"efficacy_verified":False}
    write(Path(output)/"report.json",report)
    return report


def run(compilation,policy,output):
    # All code resolution and seals precede the first CUDA allocation/model load.
    import torch
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
        "fixed":"80 pairs at original observed root; lock both tokens before physics",
        "receding":"40 first pairs at root; remaining40 at newly observed boundary",
        "per_unaborted_arm":{"planning_predict_calls":160,"planning_predict_samples":160,
            "verification_predict_calls":4,"verification_predict_samples":4,"rgb_encode_calls":2,"rgb_encode_frames":6},
        "feedback":"same RGB guard every primitive; encode real3 boundaries at both decisions; fixed ignores second encoding for selection",
        "goal":"saved goal0 and Q, extra goal-image-set provenance unchanged; not single-image metric learning",
        "Q_warning":"agent motion with static object can improve metric; no task efficacy repair",
        "stopping":"two packets or first guard/termination/error; no padding/retries; retain partial resource differences",
        "budget_claim":"scheduled actual predictor samples/calls matched only when both complete; no FLOP/wall-cap/utility match",
        "P1_admission":False,"P4_admission":False,"readout_fit":False,"training":False}
    files=checked.sources_before_allocate(source,config)
    files += [Path(__file__),Path(adapter.__file__),Path(visible.__file__),Path(guard.__file__),Path(identity.__file__),
        Path(metric_for.__code__.co_filename),Path(__file__).resolve().parents[2]/f"tests/test_{TAG}.py",repo/"docs/P4_RECEDING_RUNTIME_20261027.md"]
    files=sorted(set(p.resolve() for p in files))
    seals=[create_seal(files=files[i:i+40],args_json=json.dumps(protocol),output_dir=output/f"source_seal_{i//40}") for i in range(0,len(files),40)]
    write(output/"protocol.json",protocol)
    protocol_sha=digest(output/"protocol.json")
    bindings[str(output/"protocol.json")]=protocol_sha
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
                        try:
                            context=encode_cuda(model,device,frames,between,base.p0,transform)
                            event["completed"]=True
                            return context
                        finally:
                            torch.cuda.synchronize(device)
                            event["wall_seconds"]=time.perf_counter()-start
                    def replay(context,actions,phase):
                        start=time.perf_counter()
                        event={"kind":"predict_dispatch","phase":phase,"tokens":len(actions)//5,"completed":False}
                        events.append(event)
                        try:
                            with counters.phase(phase,method=arm,horizon_tokens=len(actions)//5):
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
                        trace=[]
                        def record(step,command,transition,rgb):
                            state=base.p0.packets.state_of(env)
                            trace.append({"primitive_step":iteration*5+step,"command":command.tolist(),"state":state.tolist(),
                                "rgb_sha256":rgb_sha(rgb),"reward":float(transition[1]),"terminated":bool(transition[2]),"truncated":bool(transition[3]),
                                "official":identity.official_checked(env,state,root["goal"]),"pose":base.physical._pose_label(state,root["goal"])})
                        value=visible.execute(env,packet,root_id=key,agent_rgb=(65,105,225),on_observation=record)
                        terminal=value.pop("terminal_rgb")
                        if (len(trace)!=value["completed_commands"] or value["planned_commands"]!=5
                                or (value["status"]=="completed" and len(trace)!=5)
                                or any(not np.array_equal(t["command"],packet[j]) for j,t in enumerate(trace))):
                            raise ValueError("executor trace differs from sealed packet")
                        state=base.p0.packets.state_of(env)
                        value.update(trace=trace,terminal_rgb_sha256=rgb_sha(terminal),
                            official=identity.official_checked(env,state,root["goal"]),pose=base.physical._pose_label(state,root["goal"]))
                        write(output/f"root_{index}.{arm}.round_{iteration}.execution.json",value)
                        return value,terminal
                    with (output/f"root_{index}.{arm}.candidates.jsonl").open("x") as journal,counters:
                        def log(iteration,event):
                            journal.write(json.dumps({"round":iteration,**event},allow_nan=False)+"\n")
                            journal.flush()
                        result=branch_loop(arm=arm,root_index=index,bank=banks[index],frames=frames,between=between,
                            encode=encode,replay=replay,metric=metric,execute=execute,seal_decision=seal_decision,journal=log)
                    row.update(result)
                except Exception as exc:
                    row.update(status="runtime_error",error_type=type(exc).__name__,error=str(exc))
                finally:
                    handle.remove()
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
    def verify_sources():
        for s in seals:
            verify_seal(output_dir=s["output_dir"],expected_manifest_sha256=s["manifest_sha256"])
        return True
    report=finalize(output,rows,bindings,verify_sources,
        lambda:before==checked.tensor_state_hash(model) and all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        protocol,{"host":host,"total_wall_seconds":time.perf_counter()-started,"setup_failure":setup_failure,
            "source_seals":seals,
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
