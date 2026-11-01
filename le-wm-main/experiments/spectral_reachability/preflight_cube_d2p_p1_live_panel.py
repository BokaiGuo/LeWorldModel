"""Fail-closed model-side Cube P1 preflight over sealed live-runtime roots."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source
from experiments.spectral_reachability import preflight_cube_d2p_p0 as cube
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as runtime

SCHEMA = "cube_d2p_p1_live_panel_preflight.v1"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--live-panel",type=Path,required=True)
    p.add_argument("--training-source-receipt",type=Path,required=True); p.add_argument("--policy",type=Path,default=Path("/home/gbk/.stable_worldmodel/checkpoints/cube/lewm"))
    p.add_argument("--device",choices=("cuda",),default="cuda"); p.add_argument("--proposals",type=int,default=16); p.add_argument("--seed",type=int,default=202610300)
    a=p.parse_args(); a.output=a.output.resolve(); a.live_panel=a.live_panel.resolve(); a.policy=a.policy.resolve(); a.training_source_receipt=a.training_source_receipt.resolve()
    if a.output.exists() or a.proposals < 2: raise ValueError("output must be new and proposals >= 2")
    report=json.loads((a.live_panel/"report.json").read_text()); gates=report.get("gates",{})
    if report.get("status")!="pass" or not all(gates.values()): raise ValueError("P1 requires a passing sealed live panel")
    inputs=np.load(a.live_panel/"model_inputs.npz"); expected=report.get("model_inputs",{})
    if sha(a.live_panel/"model_inputs.npz") != expected.get("sha256"): raise ValueError("sealed model inputs hash mismatch")
    receipt=source.validate_receipt(a.training_source_receipt); mean=np.asarray(receipt["calibration_action_mean"],np.float32); std=np.asarray(receipt["calibration_action_std"],np.float32)
    host=require_cuda_allocation("cuda"); d=torch.device("cuda"); model=load_compatible_pretrained(a.policy,device=d).eval(); model.requires_grad_(False); model.interpolate_pos_encoding=True
    before=runtime._parameter_hash(model); block=cube._validate_cube_action_protocol(model,mean,std); transform=_image_transform()
    low=torch.full((source.RAW_ACTION_DIM,),-1.,device=d); high=-low; rows=[]; started=time.time()
    gen=torch.Generator(device=d).manual_seed(a.seed)
    bank=2*torch.rand((a.proposals,source.ACTION_BLOCK,source.RAW_ACTION_DIM),generator=gen,device=d)-1; bank[0].zero_()
    for i in range(inputs["root_rgb"].shape[0]):
        # Index 1 is a sealed source-supported positive control: it is the
        # actual packet that generated this root's recorded successor.  It
        # exposes packet/temporal encoding defects before a selector failure
        # is attributed to the frozen model.
        root_bank=bank.clone(); root_bank[1]=torch.as_tensor(inputs["heldout_packets"][i],device=d)
        scores=[]
        for j in range(a.proposals):
            z=runtime._terminal_prediction(model,list(inputs["history_rgb"][i]),inputs["root_rgb"][i],inputs["history_packets"][i],root_bank[j:j+1],mean=mean,std=std,action_block=source.ACTION_BLOCK,action_dim=source.RAW_ACTION_DIM,transform=transform,device=d,low=low,high=high,future_packets_are_raw=True)
            goal=runtime._goal_latent(model,inputs["goal_rgb"][i],transform=transform,device=d); scores.append(float((z-goal).square().mean().detach()))
        best=int(np.argmin(scores)); rows.append({"root_id":report["rows"][i]["root_id"],"selected_proposal_index":best,"zero_goal_cost":scores[0],"selected_goal_cost":scores[best],"score_std":float(np.std(scores)),"positive_control_index":1,"proposal_goal_costs":scores,"proposal_raw_packets":root_bank.detach().cpu().tolist(),"selected_raw_packet":root_bank[best].detach().cpu().tolist()})
    g={"cuda_only":True,"frozen_parameter_hash_unchanged":before==runtime._parameter_hash(model),"sealed_live_inputs_hash_bound":True,"action_block_matches_training":block==source.ACTION_BLOCK,"rgb_action_only":True,"evaluator_state_not_used":True,"physical_candidates_not_executed":True,"every_root_action_sensitive":all(x["score_std"]>1e-10 for x in rows)}
    out={"schema":SCHEMA,"status":"pass" if all(g.values()) else "fail","scope":"CUDA-only sealed Cube P1 model-side ranking admission; no candidate or outcome executed.","cuda_host":{**host,"python":platform.python_version()},"upstream_live_panel":{"path":str(a.live_panel),"report_sha256":sha(a.live_panel/"report.json"),"inputs_sha256":expected["sha256"]},"policy":{"path":str(a.policy),"weights_sha256":sha(a.policy/"weights.pt"),"repair_manifest_sha256":sha(a.policy/"repair_manifest.json") if (a.policy/"repair_manifest.json").is_file() else None},"gates":g,"protocol":{"planner_input":"sealed RGB/action tensors only","proposal_bank":"frozen physical uniform five-action packets; score and replay consume identical raw packet values","future_packet_parameterization":"already-physical raw actions; no tanh compiler remapping","physical_execution":"deferred to root-disjoint calibration"},"rows":rows,"elapsed_seconds":time.time()-started}
    a.output.mkdir(parents=True); (a.output/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":out["status"],"gates":g,"output":str(a.output)},sort_keys=True))
if __name__=="__main__": main()
