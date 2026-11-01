"""Known-panel causal prefix capability audit; all neural work host CUDA.

Four presealed roles, deduplicated only by exact prefix index. Measure complete
actual-history model and physical responses after executing each prefix. No
measured future or state coordinate enters a selector. Not a new P1 outcome.
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
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.analyze_reacher_d2p_masked_suffix_search import verify_files
from experiments.spectral_reachability import reacher_d2p_capability_reference as source

SCALES = (.1, .25)


def roles(plan, feasible):
    bank = plan["bank"]
    if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
        raise ValueError("all17 ordered original prefixes required")
    result = {"common_candidate": plan["decisions"]["cem_then_gn4"]["native_index"],
        "feasible_selector": feasible, "greedy": min(range(17), key=lambda i: (bank[i]["prefix_cost"], i)), "zero": 0}
    if any(type(i) is not int or not 0 <= i < 17 for i in result.values()):
        raise ValueError("sealed role indices required")
    return result


def probe_commands(task, scale):
    if task not in chart.TASKS or scale not in SCALES:
        raise ValueError("predeclared exact mask and scale required")
    n = len(chart.active_columns(task, 2))
    return [(i, np.eye(n, dtype=np.float32)[i]*scale) for i in range(n)]


def expand_roles(mapping,all_programs):
    if type(all_programs) is not bool: raise ValueError("explicit all-programs contract required")
    return {**mapping, **({f"program_{i}":i for i in range(17)} if all_programs else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--selection-reference", type=Path)
    parser.add_argument("--metric-confirmation", action="store_true")
    parser.add_argument("--full-goal-compiler", action="store_true")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-programs",action="store_true")
    parser.add_argument("--alignment-reference",type=Path)
    parser.add_argument("--render-policy", choices=("native", "fixed-warm-render-v1"), default="native")
    args = parser.parse_args()
    if args.metric_confirmation and args.full_goal_compiler:
        raise ValueError("one explicit capability source contract required")
    simple_reference = args.metric_confirmation or args.full_goal_compiler
    if simple_reference:
        if args.selection_reference or args.all_programs or args.alignment_reference or args.render_policy != "fixed-warm-render-v1":
            raise ValueError("metric confirmation uses its own sealed four roles and fixed warm renderer; no alternate selection reference")
    elif args.selection_reference is None:
        raise ValueError("original suffix audit requires its explicit selection reference")
    if args.all_programs != bool(args.alignment_reference):
        raise ValueError("all-program expansion requires the complete original partial audit as alignment reference")
    output = args.output
    paths = {name: output.with_suffix(ext) for name, ext in (("protocol", ".protocol.json"),
        ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (output, *paths.values())):
        raise FileExistsError("new immutable capability audit paths required")
    if args.full_goal_compiler:
        from experiments.spectral_reachability import analyze_reacher_d2p_goal_metric_compiler as full_analysis
        full_analysis.analyze(args.reference)
    elif args.metric_confirmation:
        from experiments.spectral_reachability import analyze_reacher_d2p_root_metric_confirmation as metric_analysis
        metric_analysis.verify_files(args.reference)
    else:
        verify_files(args.reference)
    reference = json.loads(args.reference.read_text())
    source_schema = reference["schema"]
    if source_schema != (source.FULL if args.full_goal_compiler else source.METRIC if args.metric_confirmation else source.SUFFIX):
        raise ValueError("explicit source contract disagrees with reference schema")
    pred_path = args.reference.with_suffix(".predictions.json")
    predicted = json.loads(pred_path.read_text())
    selection_reference = args.reference if simple_reference else args.selection_reference
    selected = json.loads(selection_reference.read_text())
    selection_pred_path = selection_reference.with_suffix(".predictions.json")
    if (selected.get("status") != "completed" or not selected.get("gates")
            or any(v is not True for v in selected["gates"].values())
            or digest(selection_pred_path) != selected["predictions_sha256"]):
        raise ValueError("completed immutable selection reference required")
    selected_plans = {r["seed"]: r for r in json.loads(selection_pred_path.read_text())["rows"]}
    plans = {r["seed"]: r for r in predicted["rows"]}
    actual = {r["seed"]: r for r in reference["rows"]}
    seeds = sorted(plans)
    if len(seeds) != 8 or set(actual) != set(seeds) or not set(seeds) <= set(selected_plans):
        raise ValueError("all eight original root identities required")
    prior = reference["sealed_protocol"]
    if any(selected["sealed_protocol"][k] != prior[k] for k in ("weights_sha256", "adapter_sha256", "source_sha256_receipt")):
        raise ValueError("selection and suffix references must share the same model/source/statistics")
    for seed in seeds:
        if any(selected_plans[seed][k] != plans[seed][k] for k in
                ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256", "goal_vector", "goal_admission", "initial_snapshot")):
            raise ValueError("selection and suffix references must share complete root/goal identity")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if any(prior[k] != value for k, value in (("weights_sha256", digest(args.policy/"weights.pt")),
            ("adapter_sha256", digest(args.adapter_artifact)), ("source_sha256_receipt", source_sha))):
        raise ValueError("original official checkpoint/source/statistics required")
    assignments = {str(s): {task: (source.full_roles(source.task_plan(plans[s],task,source_schema)) if args.full_goal_compiler else source.metric_roles(plans[s]["tasks"][task]) if args.metric_confirmation else
        expand_roles(roles(plans[s]["tasks"][task], selected_plans[s]["tasks"][task]["decisions"]["feasible_incumbent"]),args.all_programs))
        for task in chart.TASKS} for s in seeds}
    cells = sum(len(set(r.values())) for tasks in assignments.values() for r in tasks.values())
    expected_futures = sum(len(set(r.values()))*(2+4*len(chart.active_columns(t,2))) for tasks in assignments.values() for t,r in tasks.items())
    expected_predict = sum(len(set(r.values()))*(9+8*len(chart.active_columns(t,2))) for tasks in assignments.values() for t,r in tasks.items())
    modules = {"auditor": Path(__file__), "context": Path(execution.__file__), "chart": Path(chart.__file__),
        "compiler": Path(compiler.__file__), "goals": Path(goals.__file__), "p0": Path(p0.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    bindings = {str(p.resolve()): digest(p) for p in (args.reference, pred_path, selection_reference, selection_pred_path)}
    modules["capability_reference"] = Path(source.__file__)
    if args.metric_confirmation: modules["metric_reference_verifier"] = Path(metric_analysis.__file__)
    if args.full_goal_compiler: modules["full_reference_verifier"] = Path(full_analysis.__file__)
    alignment_rows,alignment={},None
    if args.alignment_reference:
        alignment=json.loads(args.alignment_reference.read_text())
        if (alignment.get("schema")!="reacher_d2p_prefix_capability.v1" or alignment.get("status")!="completed"
                or not alignment.get("gates") or any(v is not True for v in alignment["gates"].values())
                or alignment["sealed_protocol"]["seeds"]!=seeds):
            raise ValueError("complete original eight-root capability audit required")
        if any(k not in modules or sha!=digest(modules[k]) for k,sha in alignment["sealed_protocol"]["implementation_sha256"].items() if k!="auditor"):
            raise ValueError("all-program expansion must preserve every model/execution implementation")
        alignment_rows={r["seed"]:r for r in alignment["rows"]}
        bindings[str(args.alignment_reference.resolve())]=digest(args.alignment_reference)
    warm_module = None
    if args.render_policy != "native":
        from experiments.spectral_reachability import reacher_d2p_fixed_warm_render as warm_module
        modules["render_policy"] = Path(warm_module.__file__)
    protocol = {"seeds": seeds, "roles": assignments, "scales": list(SCALES),
        "render_policy": args.render_policy,
        "source_sha256_receipt": source_sha, "weights_sha256": digest(args.policy/"weights.pt"),
        "adapter_sha256": digest(args.adapter_artifact), "goal_contract": prior["goal_contract"],
        "input_bindings": bindings, "cells": cells, "expected_physical_futures": expected_futures,
        "expected_predict_samples": expected_predict, "analytic_jacobians": cells*2, "logical_vjp_rows": cells*384,
        "physical_labels_used_for_selection": False, "P1_admission": False,
        "implementation_sha256": {k: digest(p) for k,p in modules.items()},
        "scope": "Known-panel evaluator-only prefix capability diagnosis; all roots and four model-sealed roles retained. No P1 admission, parameter fitting, new efficacy confirmation, or deployment of physical probes."}
    if simple_reference:
        protocol["source_contract"] = source_schema
        protocol["reference_zero_suffix_prediction_saved"] = args.full_goal_compiler
        protocol["scope"] = "Complete8 sealed metric-confirmation roots and all3 tasks; evaluator-only physical prefix capability at both fixed scales. Euclidean/metric/greedy/zero roles derive only from sealed model decisions. No outcome-filtered roots/tasks, no physical-oracle deployment or P1 admission. Original predicted-zero vectors were not saved; original prefix vectors and all192-by-n responses must replay exactly."
        if args.full_goal_compiler:
            protocol["scope"] = "All8 full-Q compiler roots/all3 tasks; native/common/greedy/zero roles derive only from presealed model decisions. Both fixed physical probe scales, exact original prefix/zero-suffix/response replay; evaluator-only, no new deployment selection or P1 admission."
    if args.all_programs:
        protocol["all_programs"]=True
        protocol["alignment_reference"]={"path":str(args.alignment_reference.resolve()),"sha256":digest(args.alignment_reference),
            "expected_exact_cells":alignment["sealed_protocol"]["cells"]}
        protocol["scope"]="Complete all17-prefix/all8-root/all3-mask evaluator-only capability diagnosis, no outcome-picked candidates. Original partial cells must reproduce exactly. No P1 admission, new efficacy or physical-oracle deployment."
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k,p in modules.items(): shutil.copyfile(p,paths["implementation"]/(k+".py"))
    print(json.dumps({"host_cuda_only": True, "cells": cells, "physical_futures": expected_futures, "predict_samples": expected_predict}),flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean,std = shared._load_action_stats(args.adapter_artifact,"swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean,std)
    processor = shared._Preprocessor(normalizer)
    mean_t,std_t = (torch.tensor(v,device="cuda") for v in (normalizer.block_mean,normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding=True
    parameters,buffers = shared._parameter_hash(model),{k:v.clone() for k,v in model.named_buffers()}
    original,count = model.predict,0
    def counted(emb,actions):
        nonlocal count
        if emb.device.type != "cuda" or actions.device.type != "cuda": raise RuntimeError("CPU neural fallback forbidden")
        count += len(emb)
        return original(emb,actions)
    model.predict=counted
    def prepare(c):
        return p0._prepare(processor,c["frames"],c["root_frame"],c["goal"],np.asarray(c["history_action_blocks"],np.float32),torch.device("cuda"))
    rows,futures,aligned= [],0,0
    render_receipts=[]
    try:
        for seed in seeds:
            old=plans[seed]
            env=goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
            renderer=None
            try:
                if warm_module is not None: renderer=warm_module.FixedWarmRenderer(env)
                ctx=shared._source_state_context(env,dataset=dataset,seed=seed)
                ctx["execution_snapshot"]["observed_root_rgb"]=execution.pack_rgb_observation(ctx["root_frame"])
                if any(ctx[k]!=old[k] for k in ("source_episode","source_start_row","root_rgb_sha256","goal_rgb_sha256")) or ctx["execution_snapshot"]!=old["initial_snapshot"]:
                    raise RuntimeError("original full source/root/goal state changed")
                pixels,goal_pixels,history=prepare(ctx)
                with torch.no_grad():
                    root_emb=p0._latent_context(model,pixels,history)
                    target=p0._goal_latent(model,goal_pixels)
                if target.tolist()!=old["goal_vector"]: raise RuntimeError("original target representation changed")
                tasks={}
                for task in chart.TASKS:
                    mapping=assignments[str(seed)][task]
                    measurements={}
                    width=len(chart.active_columns(task))
                    for index in sorted(set(mapping.values())):
                        before=count
                        bank=source.task_plan(old,task,source_schema)["bank"][index]
                        raw=np.asarray(source.raw_actions(bank,source_schema),np.float32)[:5]
                        frame,goal=execution.restore_execution_snapshot(env,seed=seed,snapshot=ctx["execution_snapshot"],root_identity="sealed_observation_exact_state")
                        p0._verify_execution_identity(frame,goal,ctx["root_frame"],ctx["goal"])
                        current=execution.advance_executed_context(env,context=ctx,raw_action_block=raw,action_block=5,verified_root_frame=ctx["root_frame"])
                        current["execution_snapshot"]["observed_root_rgb"]=execution.pack_rgb_observation(current["root_frame"])
                        pix,_,hist=prepare(current)
                        with torch.no_grad(): emb=p0._latent_context(model,pix,hist).detach()
                        packet={"packet":1,"observed_vector":emb[0,-1].tolist(),"qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),
                            "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]}
                        if packet!=source.packets(source.physical_branch(actual[seed],task,index,source_schema),source_schema)[0]:
                            raise RuntimeError("complete original prefix observation changed")
                        prefix=torch.tensor(raw.ravel()[chart.active_columns(task)],device="cuda").reshape(1,1,width)
                        predicted_emb,predicted_history,prefix_vector=compiler.after_prefix(model,root_emb,history,prefix,task=task,mean=mean_t,std=std_t)
                        if prefix_vector[0].tolist()!=bank["prefix_vector"]: raise RuntimeError("original model prefix changed")
                        center=torch.zeros(2*width,device="cuda")
                        def predicted_terminal(x): return chart.terminal(model,predicted_emb,predicted_history,x.reshape(1,2,width),task=task,mean=mean_t,std=std_t)[0]
                        def terminal(x): return chart.terminal(model,emb,hist,x.reshape(1,2,width),task=task,mean=mean_t,std=std_t)[0]
                        with torch.no_grad(): predicted_zero=predicted_terminal(center)
                        predicted_response=torch.autograd.functional.jacobian(predicted_terminal,center.requires_grad_(True),vectorize=True).detach()
                        if ((source_schema==source.SUFFIX and predicted_zero.tolist()!=bank["zero_suffix_vector"])
                                or (source_schema==source.FULL and predicted_zero.tolist()!=bank["own_zero_suffix_vector"])
                                or predicted_response.tolist()!=bank["response"]):
                            raise RuntimeError("original full predicted-state response changed")
                        replay_layers=[]
                        def observe(x):
                            nonlocal futures
                            actions=chart.expand_commands(x.reshape(1,2,width),task)[0].reshape(10,2).detach().cpu().numpy()
                            restored,goal_image=execution.restore_execution_snapshot(env,seed=seed,snapshot=current["execution_snapshot"],root_identity="sealed_observation_exact_state")
                            p0._verify_execution_identity(restored,goal_image,current["root_frame"],current["goal"])
                            advanced=current
                            for j in range(2): advanced=execution.advance_executed_context(env,context=advanced,raw_action_block=actions[5*j:5*(j+1)],action_block=5,verified_root_frame=advanced["root_frame"])
                            px,_,h=prepare(advanced)
                            with torch.no_grad(): observed=p0._latent_context(model,px,h)[0,-1]
                            futures+=1
                            # Preserve failure-layer evidence without changing the
                            # cell contract, model calls, actions or acceptance.
                            replay_layers.append({"snapshot":advanced["execution_snapshot"],
                                "rgb_sha256":advanced["root_rgb_sha256"],
                                "history_rgb_sha256":advanced["history_rgb_sha256"]})
                            if len(replay_layers)>2: replay_layers.pop(0)
                            return {"observed":observed.tolist(),"qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),"raw_actions":actions.tolist(),"root_rgb_sha256":advanced["root_rgb_sha256"]}
                        with torch.no_grad():
                            base=observe(center)
                            repeated=observe(center)
                            model_zero=terminal(center)
                        if base!=repeated:
                            write_new(output.with_suffix(".failure.json"),{
                                "schema":"reacher_d2p_prefix_replay_failure.v1",
                                "seed":seed,"task":task,"index":index,"physical_futures":futures,
                                "predict_samples":count,"first":base,"repeat":repeated,"layers":replay_layers,
                                "state_exact":replay_layers[0]["snapshot"]==replay_layers[1]["snapshot"],
                                "history_rgb_exact":replay_layers[0]["history_rgb_sha256"]==replay_layers[1]["history_rgb_sha256"],
                                "terminal_rgb_exact":base["root_rgb_sha256"]==repeated["root_rgb_sha256"],
                                "observed_max_error":float(np.max(np.abs(np.asarray(base["observed"])-np.asarray(repeated["observed"])))),
                                "protocol_sha256":digest(paths["protocol"]),"P1_admission":False})
                            raise RuntimeError("physical zero future replay not exact; layered failure receipt saved")
                        response=torch.autograd.functional.jacobian(terminal,center,vectorize=True).detach()
                        probes=[]
                        for scale in SCALES:
                            endpoints=[]
                            for coordinate,value in probe_commands(task,scale):
                                endpoint={"coordinate":coordinate}
                                for sign,v in (("plus",value),("minus",-value)):
                                    x=torch.tensor(v,device="cuda")
                                    with torch.no_grad(): endpoint[sign]={**observe(x),"model":terminal(x).tolist()}
                                endpoints.append(endpoint)
                            probes.append({"scale":scale,"endpoints":endpoints})
                        if count-before!=9+8*len(center): raise RuntimeError("exact per-cell predict accounting changed")
                        measurements[str(index)]={"prefix_packet":packet,"prefix_raw_actions":raw.tolist(),
                            "center":base,"center_repeat_exact":True,"model_zero":model_zero.tolist(),
                            "predicted_zero":predicted_zero.tolist(),"predicted_state_response":predicted_response.tolist(),
                            "actual_history_response":response.tolist(),"probes":probes,"predict_samples":count-before}
                        if args.all_programs and str(index) in alignment_rows[seed]["tasks"][task]["cells"]:
                            if measurements[str(index)]!=alignment_rows[seed]["tasks"][task]["cells"][str(index)]:
                                raise RuntimeError("complete original partial cell no longer reproduces exactly")
                            aligned+=1
                    tasks[task]={"roles":mapping,"cells":measurements}
                    print(json.dumps({"seed":seed,"task":task,"complete_cells":len(measurements),"physical_futures":futures}),flush=True)
                row={"seed":seed,"source_episode":old["source_episode"],"source_start_row":old["source_start_row"],
                    "goal_vector":old["goal_vector"],"goal_admission":old["goal_admission"],"tasks":tasks}
                write_new(paths["roots"]/f"root_{seed}.json",row)
                rows.append(row)
                if renderer is not None: render_receipts.append({"seed":seed,"stats":dict(renderer.stats)})
            finally:
                try:
                    if renderer is not None: renderer.restore()
                finally: env.close()
    finally: model.predict=original
    gates={"cuda_only":True,"all8_roots": [r["seed"] for r in rows]==seeds,
        "all_cells":sum(len(t["cells"]) for r in rows for t in r["tasks"].values())==cells,
        "all_physical_futures":futures==expected_futures,"predict_accounting":count==expected_predict,
        "parameters_unchanged":shared._parameter_hash(model)==parameters,
        "buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),
        "references_unchanged":all(digest(Path(p))==s for p,s in bindings.items()),
        "weights_unchanged":digest(args.policy/"weights.pt")==protocol["weights_sha256"],
        "implementation_unchanged":all(digest(p)==protocol["implementation_sha256"][k] for k,p in modules.items())}
    if args.all_programs: gates["partial_alignment_exact"]=aligned==protocol["alignment_reference"]["expected_exact_cells"]
    if warm_module is not None: gates["render_policy_verified"]=warm_module.valid_receipts(render_receipts,seeds)
    write_new(output,{"schema":"reacher_d2p_prefix_capability.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"cuda_host":host,
        "actual_predict_samples":count,"physical_futures":futures,"rows":rows,"gates":gates,
        "render_receipts":render_receipts,"P1_admission":False})
    print(json.dumps({"gates":gates,"predict_samples":count,"physical_futures":futures}),flush=True)
    if not all(gates.values()): raise RuntimeError("prefix capability integrity failed")


if __name__=="__main__": main()
