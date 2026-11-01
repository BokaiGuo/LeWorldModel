"""Episode1 only: native normalized-action JVP columns; prepare is non-neural."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

NAME="p5_jepa_native_action_response_20261027_v8"
V7="p5_jepa_official_action_fidelity_20261027_v7"
V7_SHA="f437110bc5f105afbff89f95556ee06ecace0c967641474ca47f0dbdbe43ab91"
EPISODE=1
EPSILONS=(.01,.005)
DAMPING=.01
DIM=256*384
SCRATCH_BUDGET=4*1024**3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):h.update(b)
    return h.hexdigest()


def protocol():
    return dict(schema=NAME,episode=EPISODE,parent_fixed_panel=list(range(1,9)),frames=[0,5,10,15],
        context=[0,5,10],target=15,episode_selection="fixed episode1 only; no sample/column replacement or screening",
        coordinates="ten native normalized last-packet coordinates, time-major x/y; no tanh, bounds, clipping or extra /100",
        function="F(delta)=native_predictor(official_real_context, concat(two_real_history_packets, actual_new_packet+delta), same_proprio)[0][:,-1]",
        nominal="delta=0 around actual normalized packet, NOT normalized zero action",
        jvp="torch.autograd.functional.jvp(strict=True,create_graph=False), ten sequential standard basis directions; double-backward implementation disclosed",
        output_shape=[1,256,384],response_shape=[DIM,10],residual="actual official frame15 feature minus nominal successor",
        finite_difference=dict(epsilons=list(EPSILONS),directions="all ten standard basis",method="central",role="diagnostic only, no threshold/selection/replacement"),
        metric="global patch MSE, G=J.T@J/D, b=J.T@residual/D; CUDA float64 reductions, D=98304",
        damping=DAMPING,solve="delta_normalized=solve(G+.01*I10,b); no candidate forward or action execution",
        counts=dict(official_encoder=1,proprio_encoder=1,predictor=51,jvp_requested=10,jvp_completed=10,
            autograd_grad_requested=20,autograd_grad_completed=20,backward_api_calls=0),
        predictor_phases=dict(nominal=1,jvp=10,finite_difference=40),
        frame_cost=dict(official_rgb_frames=4,proprio_context_frames=3,predictor_context_frames=153,
            predictor_context_patch_tokens=153*256),
        preflight_free_scratch_budget_bytes=SCRATCH_BUDGET,
        memory="sequential JVP; retain only D*10 float32 response and 10x10 float64 Gram; no D-square Jacobian/metric/identity",
        audit_arrays="exclusive output-stem_assets directory: response_columns.npy float32[D,10], nominal_successor.npy and official_target.npy float32[1,256,384], target_minus_successor.npy float64[1,256,384]; SHA/shape/dtype, no pickle; no JSON full arrays",
        gates=dict(preprocess_max_abs=1e-6,nominal_vs_v7_mse_abs=1e-6,jvp_primal_vs_nominal_max_abs=1e-6),
        interpretation="local differentiability/response compilation prerequisite, not physical efficacy, optimized actions, full D2P or scientific P5",
        normalized_to_data_raw="data_raw=100*(normalized*std+mean); not applied to JVP coordinates",
        official_source="unchanged native .1 offset/false antialias/37x37 grid; cached HF weights strict mapped, conditional provenance",
        P5_admission=False,training=False,cpu_neural_fallback=False,physical_execution=False)


def helper(root):
    p=root/"experiments/spectral_reachability"/(V7+".py")
    if sha(p)!=V7_SHA:raise ValueError("immutable v7 SHA rejected")
    spec=importlib.util.spec_from_file_location("sealed_v7_for_v8",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def prepare(root):
    v7=helper(root)
    frozen_path=root/f"results/{V7}/manifest_frozen.json"
    frozen=json.loads(frozen_path.read_text());v7.verify_manifest(root,frozen)
    receipt_path=root/f"results/{V7}/pusht_official_action_fidelity_cuda_main_v1.json"
    receipt=json.loads(receipt_path.read_text())
    if (receipt["status"]!="official_conditional_accuracy_recorded" or not receipt["cuda_executed"]
        or receipt["protocol"]!=frozen["protocol"] or receipt["model_before"]!=receipt["model_after"]
        or not receipt["source_data_seals_unchanged"] or not receipt["forward_count_contract_matches"]
        or receipt["actual_successful_forward_counts"]!=frozen["protocol"]["counts"]
        or [r["episode"] for r in receipt["rows"]]!=list(range(1,9))
        or any(r["status"]!="completed" for r in receipt["rows"])
        or receipt["manifest_canonical_sha256"]!=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()):
        raise ValueError("v7 actual CUDA receipt rejected")
    _,v5,_=v7.dependencies(root);v4=v5.import_v4(root)
    arrays=v5.read_arrays(v4,root);stats=v4.source_stats(v4.native_root(root)/"app/plan_common/datasets/pusht_dset.py")
    values,rgb,length=v5.row_inputs(v4,root,arrays,stats,EPISODE)
    saved=frozen["rows"][0]
    if saved["episode"]!=EPISODE or saved["rgb_sha256"]!=v4.array_sha(rgb) or any(saved["inputs"][k]["sha256"]!=v4.array_sha(v) for k,v in values.items()):
        raise ValueError("fixed episode1 real history mismatch")
    seals=dict(frozen["seals"])
    paths=[Path(__file__).resolve(),root/"tests"/f"test_{NAME}.py",root.parent/"docs"/f"{NAME}.md",
        frozen_path,receipt_path,Path(inspect.getfile(torch.autograd.functional.jvp))]
    seals.update({str(p):sha(p) for p in paths})
    if any(sha(p)!=h for p,h in seals.items()):raise ValueError("prepare drift")
    return dict(schema=NAME,status="prepared_non_neural",protocol=protocol(),seals=seals,
        row=saved,official_origin=frozen["official_origin"],official_weight_mapping=frozen["official_weight_mapping"],
        expected_dynamics_schema=frozen["expected_dynamics_schema"],v7_nominal_actual_mse=receipt["rows"][0]["arms"]["actual"]["target_global_patch_mse"],
        expected_v7_features={k:receipt["rows"][0][k] for k in ("official_features_sha256","target_features_sha256")},
        expected_model_receipts=receipt["model_after"],torch_version=torch.__version__,
        memory_math=dict(response_float32_bytes=DIM*10*4,gram_float64_bytes=10*10*8,
            dense_D_square_float32_bytes=DIM*DIM*4,backward_peak="unmeasured; 4GiB free scratch is a conservative preflight policy, not a proven upper bound"),
        cuda_executed=False,P5_admission=False)


def verify_manifest(root,manifest):
    if manifest!=prepare(root):raise ValueError("v8 frozen source/data/protocol replay differs")


def require_cuda():
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; no CPU neural fallback")


def load_cuda(root,manifest):
    """No model construction outside explicit CUDA device; no HF model/shim."""
    v7=helper(root);v3,v5,v6=v7.dependencies(root);v4=v5.import_v4(root)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",XFORMERS_DISABLED="1")
    if any(k=="dinov2" or k.startswith("dinov2.") for k in sys.modules):
        raise ValueError("preimported dinov2 rejected; fresh process required")
    sys.path.insert(0,str(v6.official_root(root)));sys.path.insert(0,str(v4.native_root(root)))
    from dinov2.hub.backbones import dinov2_vits14
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
    from safetensors.torch import load_file
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.device("cuda"):
        encoder=dinov2_vits14(**v7.protocol()["official_config"])
        predictor=vit_predictor_AdaLN(img_size=224,patch_size=14,num_frames=4,tubelet_size=1,
            embed_dim=384,predictor_embed_dim=384,depth=6,num_heads=16,use_silu=False,use_rope=True,
            local_window=(3,-1,-1),action_dim=10,proprio_dim=4,act_mlp=False,prop_mlp=False,
            proprio_encoder_inpred=False,action_encoder_inpred=True,proprio_encoding="feature",proprio_emb_dim=16,proprio_tokens=0)
        proprio=ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=4,tokens_per_step=1,embed_dim=16,shift_input=False,use_mlp=False)
    cached=load_file(str(root/".cache/dinov2/dinov2-small/model.safetensors"),device="cuda")
    v6.compare_headers({k:dict(shape=list(t.shape),dtype="F32" if t.dtype==torch.float32 else str(t.dtype)) for k,t in cached.items()},manifest["official_weight_mapping"]["source_header"])
    mapped={k:torch.cat([cached[s] for s in r["sources"]],dim=0) if len(r["sources"])>1 else cached[r["sources"][0]] for k,r in v6.mapping().items()}
    v6.compare_headers({k:list(t.shape) for k,t in encoder.state_dict().items()},{k:list(t.shape) for k,t in mapped.items()})
    encoder.load_state_dict(mapped,strict=True);del cached,mapped
    checkpoint=root/"data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"
    if sha(checkpoint)!=v4.CKPT_SHA or checkpoint.stat().st_size>256*1024**2:raise ValueError("checkpoint SHA/size rejected")
    weights=torch.load(checkpoint,weights_only=True,map_location="cuda");strict={}
    for part,module in (("predictor",predictor),("proprio_encoder",proprio)):
        shapes={k:list(t.shape) for k,t in module.state_dict().items()}
        v6.compare_headers(shapes,manifest["expected_dynamics_schema"][part]);strict[part]=v3.compare(weights[part],shapes)
        if not strict[part]["exact_match"]:raise ValueError(json.dumps(strict[part]))
        module.load_state_dict(v3.normalize(weights[part]),strict=True)
    del weights
    models=dict(official_encoder=encoder,proprio_encoder=proprio,predictor=predictor)
    for m in models.values():m.eval().requires_grad_(False)
    receipts={k:v4.model_receipt(m) for k,m in models.items()}
    if receipts!=manifest["expected_model_receipts"]:raise ValueError("v7 module state identity rejected")
    sources={**v7.imported_sources("dinov2",v6.official_root(root)),**v7.imported_sources("app",v4.native_root(root)),**v7.imported_sources("src",v4.native_root(root))}
    if any(p not in manifest["seals"] or sha(p)!=manifest["seals"][p] for p in sources.values()):raise ValueError("imported sources unsealed")
    return models,strict,sources,receipts


@contextmanager
def count_autograd(counts):
    """Scoped pass-through instrumentation only; native/module sources unchanged.

    functional.jvp calls autograd.grad twice (double backward). Count actual API
    requests/completions rather than falsely naming these native forward-mode.
    Always restore global functions, including on unsupported op or OOM.
    """
    original_grad=torch.autograd.grad;original_backward=torch.autograd.backward
    def grad(*args,**kwargs):
        counts["autograd_grad_requested"]+=1
        out=original_grad(*args,**kwargs);counts["autograd_grad_completed"]+=1
        return out
    def backward(*args,**kwargs):
        counts["backward_api_calls"]+=1
        return original_backward(*args,**kwargs)
    torch.autograd.grad=grad;torch.autograd.backward=backward
    try:yield
    finally:torch.autograd.grad=original_grad;torch.autograd.backward=original_backward


def ridge_arrays(response,residual,damping=DAMPING):
    """Non-neural test oracle, never a CPU neural fallback."""
    j=np.asarray(response,np.float64);r=np.asarray(residual,np.float64)
    if j.ndim!=2 or j.shape[1]!=10 or r.shape!=(len(j),) or len(j)==0 or not np.isfinite(j).all() or not np.isfinite(r).all() or not np.isfinite(damping) or damping<=0:
        raise ValueError("finite aligned response/residual/positive damping required")
    gram=j.T@j/len(j);rhs=j.T@r/len(j)
    return dict(gram=gram.tolist(),rhs=rhs.tolist(),delta=np.linalg.solve(gram+damping*np.eye(10),rhs).tolist())


def save_audit_arrays(directory,arrays):
    """Exclusive numeric artifacts, never object arrays or overwrite/pickle."""
    directory=Path(directory)
    expected={"response_columns":((DIM,10),np.dtype("float32")),
        "nominal_successor":((1,256,384),np.dtype("float32")),
        "official_target":((1,256,384),np.dtype("float32")),
        "target_minus_successor":((1,256,384),np.dtype("float64"))}
    if set(arrays)!=set(expected):raise ValueError("complete audit arrays required")
    for key,value in arrays.items():
        shape,dtype=expected[key]
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():
            raise ValueError("finite audit array shape/dtype rejected: "+key)
    directory.mkdir(parents=True,exist_ok=False)
    records={}
    for key,value in arrays.items():
        path=directory/(key+".npy")
        with path.open("xb") as f:np.save(f,value,allow_pickle=False)
        records[key]=dict(path=str(path.resolve()),sha256=sha(path),shape=list(value.shape),dtype=str(value.dtype),file_bytes=path.stat().st_size)
    return records


def validate_cuda(root,manifest,artifact_dir):
    verify_manifest(root,manifest);require_cuda()
    if Path(artifact_dir).exists():raise FileExistsError("exclusive audit artifact directory required")
    v7=helper(root);_,v5,_=v7.dependencies(root);v4=v5.import_v4(root)
    counts=dict(official_encoder=0,proprio_encoder=0,predictor=0,jvp_requested=0,jvp_completed=0,
        autograd_grad_requested=0,autograd_grad_completed=0,backward_api_calls=0)
    report=dict(schema=NAME,status="validation_failed",protocol=protocol(),episode=EPISODE,
        cuda_executed=True,P5_admission=False,counts=counts,trace=[],jvp_columns=[],finite_differences=[],
        manifest_canonical_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        neural_backend="official/native predictor SDPA math, float32 forward/AD, float64 reductions; TF32 off",
        instrumentation_boundary="Scoped pass-through wrappers count torch.autograd.grad/backward API entries; restored in finally. No source/module forward modifications.",
        physical_execution=False,compiled_delta_executed=False)
    models=None;handles=[];phase=dict(kind="loading");start=time.perf_counter()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    report["cuda_memory_before_load"]=dict(zip(("free","total"),torch.cuda.mem_get_info()))
    try:
        models,strict,sources,before=load_cuda(root,manifest)
        report.update(strict_comparison=strict,imported_native_sources=sources,model_before=before)
        free,total=torch.cuda.mem_get_info();report["cuda_memory_after_load"]=dict(free=free,total=total)
        if free<SCRATCH_BUDGET:raise RuntimeError("less than frozen 4GiB free scratch policy; no execution/fallback")
        def hook(key):
            def record(module,args,out):
                counts[key]+=1
                report["trace"].append(dict(module=key,phase=dict(phase),input_shapes=[list(a.shape) for a in args if isinstance(a,torch.Tensor)],
                    frames=int(args[0].shape[0] if key=="official_encoder" else args[0].shape[0]*args[0].shape[1])))
            return record
        for key,m in models.items():handles.append(m.register_forward_hook(hook(key)))
        arrays=v5.read_arrays(v4,root);stats=v4.source_stats(v4.native_root(root)/"app/plan_common/datasets/pusht_dset.py")
        values,rgb,length=v5.row_inputs(v4,root,arrays,stats,EPISODE)
        saved=manifest["row"]
        if v4.array_sha(rgb)!=saved["rgb_sha256"] or any(v4.array_sha(a)!=saved["inputs"][k]["sha256"] for k,a in values.items()):raise ValueError("input replay drift")
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import sdpa_kernel, SDPBackend
        transform=make_transforms(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False)
        # no_grad creates ordinary tensors usable as fixed inputs to backward.
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            pixels=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to("cuda",torch.float32)/255.;x=transform(pixels.clone())
            ref=torch.nn.functional.interpolate(pixels,size=(224,224),mode="bilinear",align_corners=False)
            ref=(ref-torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None])/torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
            prep_delta=(x-ref).abs().max().item()
            if prep_delta>1e-6:raise ValueError("native preprocessing mismatch")
            phase.update(kind="official_real_encoding")
            patches=models["official_encoder"](x,is_training=True)["x_norm_patchtokens"]
            if patches.shape!=(4,256,384) or not torch.isfinite(patches).all():raise ValueError("official patch shape/finite rejected")
            visual=patches[:3].reshape(1,3,1,16,16,384).detach();target=patches[3:4].detach()
            if (v4.array_sha(patches.cpu().numpy())!=manifest["expected_v7_features"]["official_features_sha256"]
                or v4.array_sha(target.cpu().numpy())!=manifest["expected_v7_features"]["target_features_sha256"]):
                raise ValueError("official real features differ from v7 SHA receipts")
            prop=torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            phase.update(kind="real_proprio_encoding");features=models["proprio_encoder"](prop).expand(1,3,256,16).detach()
            history=torch.from_numpy(values["packets"][:2].copy()).unsqueeze(0).to("cuda")
            nominal_action=torch.from_numpy(values["packets"][-1].copy()).to("cuda")
        def successor(delta):
            if delta.shape!=(10,) or delta.device.type!="cuda":raise ValueError("native normalized delta[10] on CUDA required")
            acts=torch.cat((history,(nominal_action+delta)[None,None]),dim=1)
            result=models["predictor"](visual,acts,features)
            if result[0].shape!=(1,3,256,384) or result[2].shape!=(1,3,256,16):raise ValueError("native successor shape rejected")
            y=result[0][:,-1]
            if not torch.isfinite(y).all():raise ValueError("nonfinite successor")
            return y.reshape(DIM)
        zero=torch.zeros(10,device="cuda");basis=torch.eye(10,device="cuda")
        columns=torch.empty((DIM,10),device="cuda",dtype=torch.float32)
        with sdpa_kernel(SDPBackend.MATH):
            phase.clear();phase.update(kind="nominal")
            with torch.no_grad():nominal=successor(zero)
            residual=target.reshape(DIM).double()-nominal.double();mse=residual.square().mean().item()
            if abs(mse-manifest["v7_nominal_actual_mse"])>1e-6:raise ValueError("v7 nominal successor MSE replay mismatch")
            report["input_binding"]=dict(episode=EPISODE,frames=[0,5,10,15],video_length=length,rgb_sha256=saved["rgb_sha256"],
                histories_sha256={k:r["sha256"] for k,r in saved["inputs"].items()},actual_normalized_packet=values["packets"][-1].reshape(5,2).tolist(),
                native_features_sha256=v4.array_sha(patches.cpu().numpy()),target_sha256=v4.array_sha(target.cpu().numpy()),
                successor_sha256=v4.array_sha(nominal.reshape(1,256,384).cpu().numpy()),preprocess_max_abs=prep_delta)
            report["target_residual"]=dict(shape=[1,256,384],sign="target minus successor",storage="target_minus_successor.npy",global_patch_mse=mse)
            with count_autograd(counts):
                for i in range(10):
                    phase.clear();phase.update(kind="jvp",coordinate=i)
                    counts["jvp_requested"]+=1
                    with torch.enable_grad():primal,column=torch.autograd.functional.jvp(successor,zero,basis[i],create_graph=False,strict=True)
                    counts["jvp_completed"]+=1
                    if not torch.isfinite(column).all() or column.shape!=(DIM,):raise ValueError("nonfinite/mismatched JVP column")
                    primal_delta=(primal-nominal).abs().max().item()
                    if primal_delta>1e-6:raise ValueError("JVP primal differs from nominal")
                    columns[:,i]=column.detach()
                    report["jvp_columns"].append(dict(coordinate=i,step=i//2,axis="x" if i%2==0 else "y",
                        primal_max_abs=primal_delta,l2_norm=column.double().norm().item(),max_abs=column.abs().max().item(),sha256=v4.array_sha(column.cpu().numpy())))
                    del primal,column
            with torch.no_grad():
                for epsilon in EPSILONS:
                    for i in range(10):
                        phase.clear();phase.update(kind="finite_difference",epsilon=epsilon,coordinate=i,sign="plus")
                        plus=successor(epsilon*basis[i])
                        phase["sign"]="minus";minus=successor(-epsilon*basis[i])
                        fd=(plus.double()-minus.double())/(2*epsilon);ad=columns[:,i].double();difference=fd-ad
                        report["finite_differences"].append(dict(epsilon=epsilon,coordinate=i,
                            rmse=difference.square().mean().sqrt().item(),max_abs=difference.abs().max().item(),
                            relative_l2=difference.norm().item()/max(ad.norm().item(),1e-12),fd_l2=fd.norm().item(),
                            cosine=(fd@ad).item()/max(fd.norm().item()*ad.norm().item(),1e-24),diagnostic_only=True))
                        del plus,minus,fd,ad,difference
                j=columns.double();gram=j.T@j/DIM;rhs=j.T@residual/DIM
                if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():raise ValueError("nonfinite Gram/rhs")
                eigenvalues=torch.linalg.eigvalsh(gram)
                delta=torch.linalg.solve(gram+DAMPING*torch.eye(10,device="cuda",dtype=torch.float64),rhs)
                if not torch.isfinite(delta).all():raise ValueError("nonfinite damped parameters")
                report["response_compile"]=dict(response_shape=[DIM,10],gram_divisor=DIM,gram=gram.cpu().tolist(),rhs=rhs.cpu().tolist(),
                    eigenvalues_normalized_gram=eigenvalues.cpu().tolist(),singular_values_mse_response=eigenvalues.clamp_min(0).sqrt().flip(0).cpu().tolist(),
                    singular_values_raw_J=(eigenvalues.clamp_min(0)*DIM).sqrt().flip(0).cpu().tolist(),
                    damping=DAMPING,delta_normalized=delta.reshape(5,2).cpu().tolist(),
                    linear_residual_mse=(residual-j@delta).square().mean().item(),
                    solved_equation_residual_norm=((gram+DAMPING*torch.eye(10,device="cuda",dtype=torch.float64))@delta-rhs).norm().item(),
                    applied=False,nonlinear_candidate_evaluated=False)
                report["audit_arrays"]=save_audit_arrays(artifact_dir,dict(
                    response_columns=columns.cpu().numpy(),nominal_successor=nominal.reshape(1,256,384).cpu().numpy(),
                    official_target=target.cpu().numpy(),target_minus_successor=residual.reshape(1,256,384).cpu().numpy()))
        report["status"]="native_action_response_recorded"
    except Exception as exc:
        report.update(status="validation_failed",error=type(exc).__name__+": "+str(exc),failed_phase=dict(phase))
    finally:
        for h in handles:h.remove()
        torch.cuda.synchronize()
        report["elapsed_wall_seconds"]=time.perf_counter()-start
        report["cuda_peak_memory"]=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        report["source_data_seals_unchanged"]=all(sha(p)==h for p,h in manifest["seals"].items())
        if models is not None:
            try:
                report["model_after"]={k:v4.model_receipt(m) for k,m in models.items()}
                report["frozen_state_unchanged"]=report["model_after"]==report.get("model_before")
            except Exception as exc:report["freeze_receipt_error"]=str(exc);report["frozen_state_unchanged"]=False
        report["count_contract_matches"]=counts==protocol()["counts"]
        report["frame_cost_observed"]={k:sum(t["frames"] for t in report["trace"] if t["module"]==k) for k in ("official_encoder","proprio_encoder","predictor")}
        report["predictor_phase_counts"]={k:sum(t["module"]=="predictor" and t["phase"]["kind"]==k for t in report["trace"]) for k in ("nominal","jvp","finite_difference")}
        report["frame_contract_matches"]=report["frame_cost_observed"]==dict(official_encoder=4,proprio_encoder=3,predictor=153)
        report["phase_contract_matches"]=report["predictor_phase_counts"]==protocol()["predictor_phases"]
        if not all((report["source_data_seals_unchanged"],report.get("frozen_state_unchanged",False),report["count_contract_matches"],report["frame_contract_matches"],report["phase_contract_matches"])):
            report["status"]="validation_failed"
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--validate-cuda",action="store_true");parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.validate_cuda and args.manifest is None:parser.error("--manifest required")
    artifact_dir=Path(str(args.output.with_suffix(""))+"_assets")
    result=validate_cuda(args.root.resolve(),json.loads(args.manifest.read_text()),artifact_dir) if args.validate_cuda else prepare(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(result["status"])
    if result["status"]=="validation_failed":raise SystemExit(1)


if __name__=="__main__":main()
