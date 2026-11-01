"""Prospective fixed episodes 1..8: real target conditional accuracy, CUDA host only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

NAME = "p5_jepa_action_fidelity_20261027_v5"
V4_SHA = "cb84d0d5fbdcc20a8f0891e4d7835989f76138370d1b734c38ace63c4e97c9cb"
EPISODES = tuple(range(1, 9))
ARMS = ("actual", "normalized_zero", "negative_normalized", "reverse_steps")


def protocol():
    return {"schema": NAME, "episodes": list(EPISODES), "excluded_opened_episode": 0,
            "frames": [0, 5, 10, 15], "context": [0, 5, 10], "target": 15,
            "action_slices": [[0,5], [5,10], [10,15]], "arms": list(ARMS),
            "intervention": "only normalized last packet 5x2; preserve two historical packets and proprio",
            "zero_semantics": "native action mean, NOT raw zero",
            "negative_semantics": "negate normalized last packet, NOT raw action",
            "reverse_semantics": "reverse five step axis only, preserve x/y order",
            "metric": "global patch MSE over 256x384; float64 difference/square/reduction on CUDA",
            "ordering": "actual strictly lower/equal/higher than each control; exact MSE ties retained",
            "root_policy": "all eight retained; no replace/select/retry; any unavailable frame/file blocks CUDA",
            "encoder_pair": "one native encoder + one direct SAME HF model per episode; not official equivalence",
            "expected_forward_counts": {"native_encoder":8, "HF_model_total_including_native":16,
                                        "proprio_encoder":8, "predictor":32},
            "training": False, "P5_admission": False, "control_benefit_claim": False}


def import_v4(root):
    path = root / "experiments/spectral_reachability/p5_jepa_real_forward_20261027_v4.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != V4_SHA:
        raise ValueError("immutable v4 SHA rejected")
    spec = importlib.util.spec_from_file_location("sealed_p5_v4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controls(packets):
    packets = np.asarray(packets, dtype=np.float32)
    if packets.shape != (3,10) or not np.isfinite(packets).all():
        raise ValueError("expected finite three 5x2 action packets")
    result = {arm: packets.copy() for arm in ARMS}
    result["normalized_zero"][-1] = 0
    result["negative_normalized"][-1] *= -1
    result["reverse_steps"][-1] = packets[-1].reshape(5,2)[::-1].reshape(10)
    return result


def row_inputs(v4, root, arrays, stats, episode):
    if episode not in EPISODES:
        raise ValueError("fixed held-out episodes only")
    states, velocities, actions = arrays
    if states.shape != (21,246,5) or velocities.shape != (21,246,2) or actions.shape != velocities.shape:
        raise ValueError("native array shape drift")
    raw_prop = np.concatenate([states[episode,[0,5,10,15],:2], velocities[episode,[0,5,10,15]]],axis=-1)
    prop = (raw_prop-np.array(stats["PROPRIO_MEAN"],np.float32))/np.array(stats["PROPRIO_STD"],np.float32)
    raw_actions = actions[episode,:15].copy()
    packets = ((raw_actions/100.-np.array(stats["ACTION_MEAN"],np.float32))/np.array(stats["ACTION_STD"],np.float32)).reshape(3,10)
    rgb, length = v4.decode_video(v4.data_root(root) / f"obses/episode_{episode:03d}.mp4")
    return {"raw_proprio":raw_prop, "normalized_proprio":prop,
            "raw_actions":raw_actions, "packets":packets}, rgb, length


def read_arrays(v4, root):
    d = v4.data_root(root)
    return tuple(v4.restricted_array(d/n,v4.ARRAY_HASHES[n]) for n in ("states.pth","velocities.pth","rel_actions.pth"))


def seals(v4, root):
    records = {r["path"]:r["sha256"] for r in v4.bindings(root)}
    paths = [Path(__file__).resolve(), root / "tests" / f"test_{NAME}.py",
             root.parent / "docs" / f"{NAME}.md",
             root / "results/p5_jepa_real_forward_20261027_v4/manifest_frozen.json",
             root / "results/p5_jepa_real_forward_20261027_v4/pusht_real_forward_cuda_main_v1.json"]
    paths += [v4.data_root(root)/f"obses/episode_{e:03d}.mp4" for e in EPISODES]
    for p in paths:
        records[str(p)] = v4.sha(p)  # missing file must reject preparation, not skip
    return [{"path":p,"sha256":h} for p,h in sorted(records.items())]


def prepared_rows(v4, root):
    arrays = read_arrays(v4,root)
    stats = v4.source_stats(v4.native_root(root)/"app/plan_common/datasets/pusht_dset.py")
    rows = []
    for episode in EPISODES:
        row = {"episode":episode,"status":"unavailable"}
        try:
            values,rgb,length = row_inputs(v4,root,arrays,stats,episode)
            row.update(status="prepared", video_length=length, rgb_shape=list(rgb.shape),
                       rgb_sha256=v4.array_sha(rgb),
                       inputs={k:{"sha256":v4.array_sha(v),"values":v.tolist()} for k,v in values.items()},
                       controls={k:v.tolist() for k,v in controls(values["packets"]).items()})
        except Exception as exc:
            row["error"] = type(exc).__name__+": "+str(exc)
        rows.append(row)
    return rows


def prepare(root):
    v4 = import_v4(root)
    v4.verify_seal(root,json.loads((root/"results/p5_jepa_real_forward_20261027_v4/manifest_frozen.json").read_text()))
    rows = [{"episode":e,"status":"not_read"} for e in EPISODES]
    try:
        before = seals(v4,root)
        rows = prepared_rows(v4,root)
        if before != seals(v4,root):
            raise ValueError("preparation byte drift")
        if any(r["status"]!="prepared" for r in rows):
            raise ValueError("one or more fixed rows unavailable; CUDA prohibited")
        return {"schema":NAME,"status":"prepared_non_neural","protocol":protocol(),
                "seals":before,"rows":rows,"cuda_executed":False,"P5_admission":False}
    except Exception as exc:
        return {"schema":NAME,"status":"preparation_failed_cuda_prohibited","protocol":protocol(),
                "rows":rows,"error":type(exc).__name__+": "+str(exc),"cuda_executed":False,"P5_admission":False}


def verify_manifest(v4,root,manifest):
    if (manifest["schema"]!=NAME or manifest["status"]!="prepared_non_neural"
            or manifest["protocol"]!=protocol() or manifest["seals"]!=seals(v4,root)
            or [r["episode"] for r in manifest["rows"]]!=list(EPISODES)
            or any(r["status"]!="prepared" for r in manifest["rows"])):
        raise ValueError("prospective protocol/source/input seal rejected")
    # All eight decoded inputs compared again BEFORE any CUDA neural execution.
    if prepared_rows(v4,root)!=manifest["rows"]:
        raise ValueError("all-eight real input replay mismatch")


def ordering(costs):
    if set(costs)!=set(ARMS) or not all(np.isfinite(v) for v in costs.values()):
        raise ValueError("missing/nonfinite arm cost")
    return {arm:{"control_minus_actual_mse":costs[arm]-costs["actual"],
                 "actual_order":"lower" if costs["actual"]<costs[arm] else
                    "higher" if costs["actual"]>costs[arm] else "equal"} for arm in ARMS[1:]}


def summarize(rows):
    if [r["episode"] for r in rows]!=list(EPISODES) or any(r["status"]!="completed" for r in rows):
        return {"all_eight_completed":False,"comparison_summary":None}
    return {"all_eight_completed":True,"comparison_summary":{
        arm:{"actual_lower_count":sum(r["ordering"][arm]["actual_order"]=="lower" for r in rows),
             "equal_count":sum(r["ordering"][arm]["actual_order"]=="equal" for r in rows),
             "actual_higher_count":sum(r["ordering"][arm]["actual_order"]=="higher" for r in rows),
             "mean_control_minus_actual_mse":float(np.mean([r["ordering"][arm]["control_minus_actual_mse"] for r in rows]))}
        for arm in ARMS[1:]}}


def validate_cuda(root,manifest):
    v4 = import_v4(root)
    verify_manifest(v4,root,manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU neural fallback")
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        JEPAWM_DINOV2_SOURCE=str(root/"external/local_dinov2"),
        LEWM_DINOV2_WEIGHTS=str(root/".cache/dinov2/dinov2-small"))
    sys.path.insert(0,str(v4.native_root(root)))
    from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
    from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
    from app.plan_common.models.dino import DinoEncoder
    from app.plan_common.datasets.transforms import make_transforms
    with torch.device("cuda"):
        predictor=vit_predictor_AdaLN(img_size=224,patch_size=14,num_frames=4,tubelet_size=1,
            embed_dim=384,predictor_embed_dim=384,depth=6,num_heads=16,use_silu=False,use_rope=True,
            local_window=(3,-1,-1),action_dim=10,proprio_dim=4,act_mlp=False,prop_mlp=False,
            proprio_encoder_inpred=False,action_encoder_inpred=True,proprio_encoding="feature",
            proprio_emb_dim=16,proprio_tokens=0)
        proprio=ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=4,tokens_per_step=1,
                                      embed_dim=16,shift_input=False,use_mlp=False)
        encoder=DinoEncoder("dinov2_vits14","x_norm_patchtokens")
    modules={"predictor":predictor,"proprio_encoder":proprio,"native_encoder":encoder}
    for module in modules.values():
        module.to("cuda").eval().requires_grad_(False)
    header=v4.encoder_header_comparison(root,encoder.base_model.model)
    path=root/"experiments/spectral_reachability/p5_jepa_strict_metadata_20261027_v3.py"
    spec=importlib.util.spec_from_file_location("sealed_p5_v3",path)
    v3=importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
    checkpoint=root/"data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"
    if v4.sha(checkpoint)!=v4.CKPT_SHA:
        raise ValueError("checkpoint SHA rejected")
    weights=torch.load(checkpoint,weights_only=True,map_location="cuda")
    strict={}
    for key,module in (("predictor",predictor),("proprio_encoder",proprio)):
        strict[key]=v3.compare(weights[key],{k:list(v.shape) for k,v in module.state_dict().items()})
        if not strict[key]["exact_match"]:
            raise ValueError(json.dumps(strict[key]))
        module.load_state_dict(v3.normalize(weights[key]),strict=True)
    del weights
    before={k:v4.model_receipt(v) for k,v in modules.items()}
    trace=[]; hooks=[]
    for key,module in (*modules.items(),("HF_model_total_including_native",encoder.base_model.model)):
        # Forward hooks count actual successful module calls, not expected work.
        hooks.append(module.register_forward_hook(lambda mod,args,out,key=key: trace.append(key)))
    transform=make_transforms(img_size=224,random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.,1.),random_resize_scale=(1.,1.),reprob=0.,auto_augment=False,motion_shift=False)
    arrays=read_arrays(v4,root)
    stats=v4.source_stats(v4.native_root(root)/"app/plan_common/datasets/pusht_dset.py")
    rows=[{"episode":e,"status":"not_run"} for e in EPISODES]
    try:
        for index,episode in enumerate(EPISODES):
            row=rows[index]; start=len(trace)
            try:
                values,rgb,length=row_inputs(v4,root,arrays,stats,episode)
                saved=manifest["rows"][index]
                if v4.array_sha(rgb)!=saved["rgb_sha256"] or any(v4.array_sha(v)!=saved["inputs"][k]["sha256"] for k,v in values.items()):
                    raise ValueError("episode input drift")
                pixels=torch.from_numpy(rgb.copy()).permute(0,3,1,2).to("cuda",torch.float32)/255.
                inputs=transform(pixels.clone())
                ref=torch.nn.functional.interpolate(pixels,size=(224,224),mode="bilinear",align_corners=False)
                ref=(ref-torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None])/torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
                prep_delta=(inputs-ref).abs().max().item()
                if prep_delta>1e-6: raise ValueError("preprocessing mismatch")
                with torch.inference_mode():
                    patches=encoder(inputs)
                    paired=encoder.base_model.model(pixel_values=inputs).last_hidden_state[:,1:]
                    interface_delta=(patches-paired).abs().max().item()
                    if patches.shape!=(4,256,384) or not torch.isfinite(patches).all() or interface_delta>1e-6:
                        raise ValueError("encoder shape/finite/interface rejected")
                    visual=patches[:3].reshape(1,3,1,16,16,384)
                    prop=torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
                    features=proprio(prop).expand(1,3,256,16)
                    costs={}; action_controls=controls(values["packets"])
                    row["arms"]={}
                    for arm,packets in action_controls.items():
                        acts=torch.from_numpy(packets.copy()).unsqueeze(0).to("cuda")
                        output=predictor(visual,acts,features)
                        if output[0].shape!=(1,3,256,384) or output[2].shape!=(1,3,256,16):
                            raise ValueError("native output shape mismatch")
                        if any(v.device.type!="cuda" or not torch.isfinite(v).all() for v in (output[0],output[2])):
                            raise ValueError("nonfinite/nonCUDA successor")
                        successor=output[0][:,-1]
                        costs[arm]=(successor.double()-patches[3:4].double()).square().mean().item()
                        row["arms"][arm]={"target_global_patch_mse":costs[arm],
                            "normalized_new_packet":packets[-1].reshape(5,2).tolist(),
                            "identical_to_actual_packet":bool(np.array_equal(packets,action_controls["actual"]))}
                row.update(status="completed",ordering=ordering(costs),preprocessing_max_abs=prep_delta,
                           same_HF_interface_max_abs=interface_delta,video_length=length)
            except Exception as exc:
                row.update(status="failed",error=type(exc).__name__+": "+str(exc))
                break  # no retry/replacement; other fixed roots remain explicitly not_run
            finally:
                row["actual_successful_forward_trace"]=trace[start:]
    finally:
        for hook in hooks: hook.remove()
    after={k:v4.model_receipt(v) for k,v in modules.items()}
    unchanged=before==after
    # Verify source/data bytes after run; do not redo neural or select roots.
    seal_unchanged=manifest["seals"]==seals(v4,root)
    summary=summarize(rows)
    counts={key:trace.count(key) for key in (*modules.keys(),"HF_model_total_including_native")}
    expected=protocol()["expected_forward_counts"]
    count_match=counts==expected
    return {"schema":NAME,"status":"conditional_accuracy_recorded" if summary["all_eight_completed"] and unchanged and seal_unchanged and count_match else "validation_failed",
        "protocol":protocol(),"rows":rows,"summary":summary,"strict_comparison":strict,
        "model_before":before,"model_after":after,"frozen_state_unchanged":unchanged,
        "source_data_seals_unchanged":seal_unchanged,"encoder_header":header,
        "actual_successful_forward_counts":counts,"forward_count_contract_matches":count_match,
        "cuda_executed":True,"official_DINO_equivalence_proven":False,"P5_admission":False,
        "control_benefit_claim":False,"interpretation":"observational target fidelity in fixed val slices under SAME HF shim; no physical counterfactual/control outcome; no efficacy claim from output differences"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare",action="store_true")
    modes.add_argument("--validate-cuda",action="store_true")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.validate_cuda and args.manifest is None: parser.error("--manifest required")
    result=validate_cuda(args.root,json.loads(args.manifest.read_text())) if args.validate_cuda else prepare(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"output":str(args.output),"status":result["status"]}))
    if result["status"] in {"validation_failed","preparation_failed_cuda_prohibited"}: raise SystemExit(1)


if __name__=="__main__": main()
