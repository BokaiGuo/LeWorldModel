"""Host CUDA goal-image Jacobians and model-only fixed co-metric scoring.

Inputs contain sealed model predictions and goal RGB only: no measured future,
qpos readout, training, parameter search, or P1 admission. Seal all matrices and
both score modes; a separate evaluator may subsequently join physical outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_goal_encoder_metric as metric_module
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, TASKS
from experiments.spectral_reachability.audit_reacher_d2p_root_metric_score import OFFICIAL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--goal-images-dir",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    paths={k:args.output.with_suffix(s) for k,s in (("protocol",".protocol.json"),("metrics",".metrics.json"),("implementation",".implementation"))}
    if any(p.exists() for p in (args.output,*paths.values())): raise FileExistsError("new immutable goal co-metric outputs required")
    predictions=json.loads(args.predictions.read_text())
    prior=json.loads(args.protocol.read_text())
    if (predictions["protocol_sha256"]!=digest(args.protocol) or prior["weights_sha256"]!=OFFICIAL
            or digest(args.policy/"weights.pt")!=OFFICIAL or digest(args.adapter_artifact)!=prior["adapter_sha256"]):
        raise ValueError("original sealed predictions/official frozen model/stats required")
    seeds=prior["root_seeds"]
    if len(seeds)!=8 or any(type(s) is not int for s in seeds) or len(set(seeds))!=8 or [r["seed"] for r in predictions["rows"]]!=seeds:
        raise ValueError("complete ordered original8 model roots required")
    images={s:args.goal_images_dir/f"root_{s}_goal.png" for s in seeds}
    modules={"auditor":Path(__file__),"metric":Path(metric_module.__file__),"compiler":Path(compiler.__file__),
        "qp":Path(qp.__file__),"shared":Path(shared.__file__),"loader":Path(shared.load_compatible_pretrained.__code__.co_filename),
        "preprocessor":Path(shared._Preprocessor.__module__.replace(".","/")+".py")}
    bindings={str(p.resolve()):digest(p) for p in (args.predictions,args.protocol,args.adapter_artifact,*images.values())}
    protocol={"policy":metric_module.POLICY,"ridge":metric_module.RIDGE,"seeds":seeds,"input_bindings":bindings,
        "weights_sha256":OFFICIAL,"goal_input":"original normalized224 RGB goal only; full192 encoder outputs",
        "metric_formula":"Q=.01*(J_E@J_E.T+.01*I)^-1; sameQ for all3 actuator tasks per goal",
        "gradient_method":"192 individual input VJPs per goal; all neural computation CUDA",
        "gradient_checks":{"directions":3,"steps":[.001,.003],"direction":"Gaussian with RMS1; seed=root XOR0xA61E",
            "maximum_relative_error":.15,"minimum_cosine":.98},
        "expected_encoder_forwards":104,"expected_input_vjp_rows":1536,"expected_certified_qps":408,
        "score_modes":["gain_only_original_costs","consistent_Q_costs"],
        "bank":"same original17 actions/suffixes; no new action optimization",
        "physical_future_labels_read":False,"no_training":True,"P1_admission":False,
        "implementation_sha256":{k:digest(p) for k,p in modules.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with paths["protocol"].open("x") as f: json.dump(protocol,f,indent=2,allow_nan=False)
    paths["implementation"].mkdir()
    for k,p in modules.items(): shutil.copyfile(p,paths["implementation"]/(k+".py"))
    host=shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean,std=shared._load_action_stats(args.adapter_artifact,"swm/ReacherDMControl-v0")
    processor=shared._Preprocessor(shared.ActionNormalizer(mean,std))
    model=shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding=True
    modules["model_encode"]=Path(model.encode.__func__.__code__.co_filename)
    # Bound loader archives already name the neural implementation; additionally
    # include its exact source receipt in the model-matrix seal below.
    encode_source_sha=digest(modules["model_encode"])
    shutil.copyfile(modules["model_encode"],paths["implementation"]/"model_encode.py")
    parameters,buffers=shared._parameter_hash(model),{k:v.clone() for k,v in model.named_buffers()}
    forwards,vjps=0,0
    def encode(image):
        nonlocal forwards
        if image.device.type!="cuda" or image.dtype!=torch.float32: raise RuntimeError("CPU neural fallback forbidden")
        forwards+=1
        return model.encode({"pixels":image})["emb"][0,-1]
    matrices,operators=[],{}
    for row in predictions["rows"]:
        seed=row["seed"]
        pixels=np.asarray(Image.open(images[seed]).convert("RGB"))
        if (pixels.shape!=(224,224,3) or hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()!=row["goal_rgb_sha256"]):
            raise ValueError("unaltered original RGB goal hash required")
        goal=processor._prepare_info({"goal":pixels[None,None]})["goal"].to("cuda").detach().requires_grad_(True)
        value=encode(goal)
        if value.detach().tolist()!=row["goal_vector"] or value.shape!=(192,): raise RuntimeError("original goal latent changed")
        gradients=[]
        for i in range(192):
            gradients.append(torch.autograd.grad(value[i],goal,retain_graph=i<191)[0].detach().flatten())
            vjps+=1
        jacobian=torch.stack(gradients)
        if not torch.isfinite(jacobian).all(): raise RuntimeError("nonfinite complete encoder input Jacobian")
        checks=[]
        generator=torch.Generator(device="cuda").manual_seed(seed^0xA61E)
        for coordinate in range(3):
            direction=torch.randn(goal.shape,device="cuda",generator=generator)
            direction=direction/direction.square().mean().sqrt()
            analytic=jacobian.double()@direction.flatten().double()
            for step in protocol["gradient_checks"]["steps"]:
                with torch.no_grad(): finite=(encode(goal.detach()+step*direction).double()-encode(goal.detach()-step*direction).double())/(2*step)
                rel=float(torch.linalg.vector_norm(finite-analytic)/torch.maximum(torch.linalg.vector_norm(finite),torch.linalg.vector_norm(analytic)).clamp_min(1e-12))
                cosine=float(finite@analytic/(torch.linalg.vector_norm(finite)*torch.linalg.vector_norm(analytic)).clamp_min(1e-24))
                checks.append({"direction":coordinate,"step":step,"relative_error":rel,"cosine":cosine})
        if any(not math.isfinite(c["relative_error"]) or not math.isfinite(c["cosine"])
                or c["relative_error"]>.15 or c["cosine"]<.98 for c in checks):
            with args.output.with_suffix(".gradient_failure.json").open("x") as f: json.dump({"seed":seed,"checks":checks,"P1_admission":False},f,indent=2,allow_nan=False)
            raise RuntimeError("fixed full encoder-Jacobian directional FD check failed")
        gram=jacobian.double()@jacobian.double().T
        metric=metric_module.GoalEncoderMetric(gram)
        operators[seed]=metric
        matrices.append({"seed":seed,"gram":gram.tolist(),"gram_eigenvalues":metric.eigenvalues.tolist(),"gradient_checks":checks})
        del gradients,jacobian,goal,value
        print(json.dumps({"seed":seed,"phase":"goal_metric","input_vjp_rows":vjps,"encoder_forwards":forwards,
            "max_fd_relative_error":max(c["relative_error"] for c in checks)}),flush=True)
    with paths["metrics"].open("x") as f: json.dump({"protocol_sha256":digest(paths["protocol"]),
        "model_encode_sha256":encode_source_sha,"rows":matrices},f,indent=2,allow_nan=False)
    metric_sha=digest(paths["metrics"])
    results=[]
    qps=0
    for row in predictions["rows"]:
        seed=row["seed"]
        metric=operators[seed]
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device="cuda")
        target=tensor(row["goal_vector"])
        tasks={}
        if set(row["tasks"])!=set(TASKS): raise ValueError("all3 original tasks required")
        for task in TASKS:
            plan=row["tasks"][task]
            if [b["index"] for b in plan["bank"]]!=list(range(17)): raise ValueError("complete17 model actions required")
            banks={mode:[] for mode in protocol["score_modes"]}
            residual=metric.apply(tensor(plan["common_residual"]))
            for b in plan["bank"]:
                gain=qp.certified_response_gain(metric.apply(tensor(b["response"])),residual,damping=.01)
                qps+=1
                for mode,bank in banks.items():
                    prefix_cost=(float(metric.apply(tensor(b["prefix_vector"])-target).square().mean()) if mode=="consistent_Q_costs" else b["prefix_cost"])
                    terminal_cost=(float(metric.apply(tensor(b["terminal_vector"])-target).square().mean()) if mode=="consistent_Q_costs" else b["terminal_cost"])
                    bank.append({"index":b["index"],"prefix_cost":prefix_cost,"terminal_cost":terminal_cost,"common_response_gain":gain})
            tasks[task]={mode:{"bank":bank,"decision":compiler.select(bank)} for mode,bank in banks.items()}
        results.append({"seed":seed,"tasks":tasks})
        print(json.dumps({"seed":seed,"phase":"model_only_decisions","decisions":{t:{m:r["decision"] for m,r in modes.items()} for t,modes in tasks.items()}}),flush=True)
    gates={"cuda_only":True,"all8_goal_metrics":len(matrices)==8,"all1536_input_vjps":vjps==1536,
        "all104_encoder_forwards":forwards==104,"all408_certified_qps":qps==408,
        "same_metric_all3_tasks":True,"parameters_unchanged":shared._parameter_hash(model)==parameters,
        "buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),"all_modules_eval":not any(m.training for m in model.modules()),
        "inputs_unchanged":all(digest(Path(p))==sha for p,sha in bindings.items()),"weights_unchanged":digest(args.policy/"weights.pt")==OFFICIAL,
        "model_encode_unchanged":digest(modules["model_encode"])==encode_source_sha,"metrics_unchanged":digest(paths["metrics"])==metric_sha,
        "implementation_unchanged":all(digest(p)==protocol["implementation_sha256"][k] for k,p in modules.items() if k!="model_encode")}
    with args.output.open("x") as f: json.dump({"schema":"reacher_d2p_goal_encoder_metric_score.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"metrics_sha256":metric_sha,"cuda_host":host,"gates":gates,
        "rows":results,"P1_admission":False,"physical_future_labels_read":False,
        "scope":"Known frozen model-bank score intervention only, no fresh physical calibration or new suffix optimization. Normalized-pixel tangent metric is not guaranteed to match joint-task geometry."},f,indent=2,allow_nan=False)
    print(json.dumps({"gates":gates}),flush=True)
    if not all(gates.values()): raise RuntimeError("goal encoder metric integrity failed")


if __name__=="__main__": main()
