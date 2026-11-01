"""New official-native encoder consumer; default preparation does no neural work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

NAME = "p5_jepa_official_action_fidelity_20261027_v7"
DEPS = {
    "p5_jepa_strict_metadata_20261027_v3": "2821b5dcec368be6d60a7369220f2907d1988080a505512ad079cb70c1ec76a4",
    "p5_jepa_action_fidelity_20261027_v5": "a45e8fdaa0d42d4324571d0131d982cb99970678a4e5ef18592a5008cb965759",
    "p5_jepa_official_equivalence_20261027_v6": "27ae9a6bdd2cb7bb64394cfd55d9dca90f8c925f9a2f9ee4b9a9c9fc5b2acc25",
}
V6 = "p5_jepa_official_equivalence_20261027_v6"
EPISODES = list(range(1,9))
ARMS = ["actual", "normalized_zero", "negative_normalized", "reverse_steps"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""): h.update(b)
    return h.hexdigest()


def dependencies(root):
    modules=[]
    for name,digest in DEPS.items():
        path=root/"experiments/spectral_reachability"/(name+".py")
        if sha(path)!=digest: raise ValueError("sealed dependency drift: "+name)
        spec=importlib.util.spec_from_file_location("sealed_v7_"+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def protocol():
    return dict(schema=NAME,episodes=EPISODES,frames=[0,5,10,15],context=[0,5,10],target=15,
        action_slices=[[0,5],[5,10],[10,15]],arms=ARMS,
        intervention="replace normalized last 5x2 packet only; preserve first two packets and proprio",
        zero_semantics="native action mean, NOT raw zero",
        negative_semantics="negative normalized new packet, NOT negative raw action",
        reverse_semantics="reverse five steps of new packet only; preserve x/y order",
        encoder="unmodified official dinov2_vits14, cached HF weights strict mapped, native x_norm_patchtokens",
        official_config=dict(img_size=518,patch_size=14,init_values=1.,block_chunks=0,
            num_register_tokens=0,interpolate_offset=.1,interpolate_antialias=False,pretrained=False),
        preprocessing="native make_transforms full frame 224, ImageNet RGB; explicit comparator max_abs <=1e-6",
        metric="CUDA float64 global MSE successor last time vs actual frame15 official patch features (1,256,384)",
        ordering="actual lower/equal/higher vs each control, exact ties retained; no efficacy threshold",
        counts=dict(official_encoder=8,proprio_encoder=8,predictor=32),
        attention="official native SDPA math, xformers disabled; native predictor unchanged; TF32 disabled",
        input_binding="all decoded RGB, normalized proprio and full action histories SHA/values bound before CUDA",
        selection="all fixed episodes; failure retains unavailable/not_run rows, no replacement/retry/filtering",
        prospective_scope="new implementation protocol fixed after v5/v6 outcomes opened; same val not fresh holdout",
        provenance="official SOURCE with cached HF WEIGHTS, not independent official checkpoint authentication",
        v5_retroactive_validation=False,v6_equivalence_repaired=False,P5_admission=False,control_benefit_claim=False)


def review_v6(root, v6, frozen):
    path=root/f"results/{V6}/pusht_official_vs_hf_cuda_main_v1.json"
    r=json.loads(path.read_text())
    canonical=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    if (r["schema"]!=V6 or r["status"]!="paired_equivalence_failed" or not r["cuda_executed"]
        or r["protocol"]!=v6.protocol() or r["counts"]!=v6.protocol()["counts"]
        or r["receipts_before"]!=r["receipts_after"] or not r["strict_loads"]
        or not r["source_hashes_unchanged"] or r["frozen_manifest_sha256"]!=canonical
        or [row["episode"] for row in r["rows"]]!=EPISODES):
        raise ValueError("v6 negative receipt/protocol binding rejected")
    for receipt in r["receipts_after"].values():
        if not all(receipt[k] for k in ("all_cuda_finite","eval","frozen")):
            raise ValueError("v6 state receipt rejected")
    for path_string in r["imported_official_sources"].values():
        p=Path(path_string)
        if not p.is_relative_to(v6.official_root(root)) or str(p) not in frozen["seals"]:
            raise ValueError("v6 official import provenance rejected")
    rows=r["rows"]
    if any(row["passed"] or row["preprocess_max_abs"]!=0 for row in rows):
        raise ValueError("v6 reported failure/preprocessing facts differ")
    def span(key,metric):
        values=[row["features"][key][metric] for row in rows]
        if not np.isfinite(values).all(): raise ValueError("nonfinite v6 metric")
        return [min(values),max(values)]
    return dict(report_path=str(path),report_sha256=sha(path),all_eight_failed=True,
        preprocessing_max_abs=0.,positional_max_abs=[row["positional_max_abs"] for row in rows],
        patch_rmse_range=span("x_norm_patchtokens","rmse"),patch_max_abs_range=span("x_norm_patchtokens","max_abs"),
        class_rmse_range=span("x_norm_clstoken","rmse"),
        finding="native official offset scale_factor vs installed HF size interpolation differs before transformer",
        attribution_limit="known concrete discrepancy; no causal isolation of attention backends or proof of sole cause",
        resolution="preserve negative v6; new official-native consumer protocol, not fix thresholds/interpolation")


def prepare(root):
    import yaml
    v3,v5,v6=dependencies(root)
    v6manifest=json.loads((root/f"results/{V6}/manifest_frozen.json").read_text())
    if v6.prepare(root)!=v6manifest: raise ValueError("v6 manifest replay differs")
    review=review_v6(root,v6,v6manifest)
    v4=v5.import_v4(root)
    seals=dict(v6manifest["seals"])
    config=root/"external/jepa-wms/configs/vjepa_wm"/v3.CONFIGS["pusht"]
    paths=[Path(__file__).resolve(),root/"tests"/f"test_{NAME}.py",root.parent/"docs"/f"{NAME}.md",
           root/f"results/{V6}/manifest_frozen.json",Path(review["report_path"]),config]
    paths += [root/"experiments/spectral_reachability"/(name+".py") for name in DEPS]
    seals.update({str(p):sha(p) for p in paths})
    checkpoint=root/"data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"
    metadata=v3.load_meta(checkpoint,v4.CKPT_SHA)
    schema=v3.expected_schema(yaml.safe_load(config.read_text()),4)
    strict={part:v3.compare(metadata[part],expected) for part,expected in schema.items()}
    if not all(r["exact_match"] for r in strict.values()): raise ValueError(json.dumps(strict))
    rows=v5.prepared_rows(v4,root)
    if [r["episode"] for r in rows]!=EPISODES or any(r["status"]!="prepared" for r in rows):
        raise ValueError("fixed real rows unavailable; no CUDA authorized: "+json.dumps(rows))
    if any(sha(p)!=digest for p,digest in seals.items()): raise ValueError("preparation drift")
    return dict(schema=NAME,status="prepared_non_neural",protocol=protocol(),rows=rows,seals=seals,
        v6_readonly_review=review,restricted_meta_comparison=strict,expected_dynamics_schema=schema,
        official_weight_mapping=v6manifest["weight_audit"],official_origin=v6manifest["official_origin"],
        cuda_executed=False,P5_admission=False)


def verify_manifest(root,manifest):
    if manifest!=prepare(root): raise ValueError("v7 frozen protocol/source/data replay differs")


def require_cuda():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU neural fallback")


def imported_sources(prefix,base):
    result={k:str(Path(m.__file__).resolve()) for k,m in sys.modules.items()
        if (k==prefix or k.startswith(prefix+".")) and getattr(m,"__file__",None)}
    if not result or any(not Path(p).is_relative_to(base) for p in result.values()):
        raise ValueError("imported-source boundary rejected: "+prefix)
    return result


def validate_cuda(root,manifest):
    verify_manifest(root,manifest); require_cuda()
    v3,v5,v6=dependencies(root); v4=v5.import_v4(root)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",XFORMERS_DISABLED="1")
    if any(k=="dinov2" or k.startswith("dinov2.") for k in sys.modules):
        raise ValueError("preimported dinov2 rejected; fresh process required")
    sys.path.insert(0,str(v6.official_root(root)));sys.path.insert(0,str(v4.native_root(root)))
    from dinov2.hub.backbones import dinov2_vits14
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
    from app.plan_common.datasets.transforms import make_transforms
    from safetensors.torch import load_file
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.device("cuda"):
        encoder=dinov2_vits14(**protocol()["official_config"])
        predictor=vit_predictor_AdaLN(img_size=224,patch_size=14,num_frames=4,tubelet_size=1,
            embed_dim=384,predictor_embed_dim=384,depth=6,num_heads=16,use_silu=False,use_rope=True,
            local_window=(3,-1,-1),action_dim=10,proprio_dim=4,act_mlp=False,prop_mlp=False,
            proprio_encoder_inpred=False,action_encoder_inpred=True,proprio_encoding="feature",
            proprio_emb_dim=16,proprio_tokens=0)
        proprio=ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=4,tokens_per_step=1,
            embed_dim=16,shift_input=False,use_mlp=False)
    cached=load_file(str(root/".cache/dinov2/dinov2-small/model.safetensors"),device="cuda")
    v6.compare_headers({k:dict(shape=list(t.shape),dtype="F32" if t.dtype==torch.float32 else str(t.dtype)) for k,t in cached.items()},manifest["official_weight_mapping"]["source_header"])
    mapped={k:torch.cat([cached[s] for s in r["sources"]],dim=0) if len(r["sources"])>1 else cached[r["sources"][0]] for k,r in v6.mapping().items()}
    native_shapes={k:list(t.shape) for k,t in encoder.state_dict().items()}
    v6.compare_headers(native_shapes,{k:list(t.shape) for k,t in mapped.items()})
    if any(t.dtype!=torch.float32 or t.device.type!="cuda" for t in encoder.state_dict().values()):
        raise ValueError("official state dtype/device rejected")
    encoder.load_state_dict(mapped,strict=True);del cached,mapped
    checkpoint=root/"data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"
    if sha(checkpoint)!=v4.CKPT_SHA or checkpoint.stat().st_size>256*1024**2:
        raise ValueError("dynamics checkpoint SHA/size rejected")
    weights=torch.load(checkpoint,weights_only=True,map_location="cuda")
    strict={}
    for part,module in (("predictor",predictor),("proprio_encoder",proprio)):
        actual={k:list(t.shape) for k,t in module.state_dict().items()}
        v6.compare_headers(actual,manifest["expected_dynamics_schema"][part])
        strict[part]=v3.compare(weights[part],actual)
        if not strict[part]["exact_match"]: raise ValueError(json.dumps(strict[part]))
        module.load_state_dict(v3.normalize(weights[part]),strict=True)
    del weights
    models=dict(official_encoder=encoder,proprio_encoder=proprio,predictor=predictor)
    for m in models.values(): m.eval().requires_grad_(False)
    before={k:v4.model_receipt(m) for k,m in models.items()}
    sources={**imported_sources("dinov2",v6.official_root(root)),
             **imported_sources("app",v4.native_root(root)),**imported_sources("src",v4.native_root(root))}
    if any(p not in manifest["seals"] or sha(p)!=manifest["seals"][p] for p in sources.values()):
        raise ValueError("imported native sources absent from seal")
    trace=[];handles=[]
    for key,m in models.items():
        handles.append(m.register_forward_hook(lambda mod,args,out,key=key:trace.append(key)))
    transform=make_transforms(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),
        random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False)
    arrays=v5.read_arrays(v4,root);stats=v4.source_stats(v4.native_root(root)/"app/plan_common/datasets/pusht_dset.py")
    rows=[dict(episode=e,status="not_run") for e in EPISODES]
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            for index,episode in enumerate(EPISODES):
                row=rows[index];start=len(trace)
                try:
                    values,rgb,length=v5.row_inputs(v4,root,arrays,stats,episode);saved=manifest["rows"][index]
                    if v4.array_sha(rgb)!=saved["rgb_sha256"] or any(v4.array_sha(v)!=saved["inputs"][k]["sha256"] for k,v in values.items()):
                        raise ValueError("real input history binding differs")
                    pixels=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to("cuda",torch.float32)/255.
                    x=transform(pixels.clone())
                    ref=torch.nn.functional.interpolate(pixels,size=(224,224),mode="bilinear",align_corners=False)
                    ref=(ref-torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None])/torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
                    prep_delta=(x-ref).abs().max().item()
                    if prep_delta>1e-6: raise ValueError("native preprocessing mismatch")
                    patches=encoder(x,is_training=True)["x_norm_patchtokens"]
                    if patches.shape!=(4,256,384) or patches.device.type!="cuda" or not torch.isfinite(patches).all():
                        raise ValueError("official feature shape/finite/device rejected")
                    visual=patches[:3].reshape(1,3,1,16,16,384)
                    target=patches[3:4]
                    prop=torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
                    features=proprio(prop).expand(1,3,256,16)
                    costs={};controls=v5.controls(values["packets"]);row["arms"]={}
                    for arm,packets in controls.items():
                        if packets.tolist()!=saved["controls"][arm] or not np.array_equal(packets[:2],values["packets"][:2]):
                            raise ValueError("control/history drift")
                        acts=torch.from_numpy(packets.copy()).unsqueeze(0).to("cuda")
                        output=predictor(visual,acts,features)
                        if output[0].shape!=(1,3,256,384) or output[2].shape!=(1,3,256,16):
                            raise ValueError("native successor shape rejected")
                        if any(t.device.type!="cuda" or not torch.isfinite(t).all() for t in (output[0],output[2])):
                            raise ValueError("native successor nonfinite/nonCUDA")
                        costs[arm]=(output[0][:,-1].double()-target.double()).square().mean().item()
                        row["arms"][arm]=dict(target_global_patch_mse=costs[arm],normalized_new_packet=packets[-1].reshape(5,2).tolist(),
                            identical_to_actual_packet=bool(np.array_equal(packets,controls["actual"])))
                    row.update(status="completed",ordering=v5.ordering(costs),preprocessing_max_abs=prep_delta,
                        video_length=length,decoded_rgb_sha256=saved["rgb_sha256"],
                        histories_sha256={k:r["sha256"] for k,r in saved["inputs"].items()},
                        official_features_sha256=v4.array_sha(patches.cpu().numpy()),
                        target_features_sha256=v4.array_sha(target.cpu().numpy()),
                        feature_shape=list(patches.shape),context_feature_indices=[0,1,2],target_feature_index=3)
                except Exception as exc:
                    row.update(status="failed",error=type(exc).__name__+": "+str(exc));break
                finally: row["actual_successful_forward_trace"]=trace[start:]
    finally:
        for handle in handles:handle.remove()
    after={k:v4.model_receipt(m) for k,m in models.items()}
    unchanged=before==after
    seals_unchanged=all(sha(p)==h for p,h in manifest["seals"].items())
    counts={k:trace.count(k) for k in models};count_match=counts==protocol()["counts"]
    summary=v5.summarize(rows)
    return dict(schema=NAME,status="official_conditional_accuracy_recorded" if summary["all_eight_completed"] and unchanged and seals_unchanged and count_match else "validation_failed",
        protocol=protocol(),rows=rows,summary=summary,strict_comparison=strict,official_native_state_shapes=native_shapes,
        model_before=before,model_after=after,frozen_state_unchanged=unchanged,source_data_seals_unchanged=seals_unchanged,
        actual_successful_forward_counts=counts,forward_count_contract_matches=count_match,imported_native_sources=sources,
        manifest_canonical_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda),cuda_executed=True,
        official_implementation_used=True,official_checkpoint_provenance_proven=False,
        v5_retroactive_validation=False,P5_admission=False,control_benefit_claim=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--validate-cuda",action="store_true");parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.validate_cuda and args.manifest is None:parser.error("--manifest required")
    result=validate_cuda(args.root.resolve(),json.loads(args.manifest.read_text())) if args.validate_cuda else prepare(args.root.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(result["status"])
    if result["status"]=="validation_failed":raise SystemExit(1)


if __name__=="__main__":main()
