"""V12: decoupled visual-control-state belief for PointMaze decision transfer.

V11 proved that causal visual history improves a state readout but not final
ranking.  This run isolates the plausible architectural cause: a joint task
loss may distort the belief needed by a state-conditioned action ranker.
V12 freezes a visual history -> state estimator before fitting the task ranker
on true source states, and deploys only the predicted state.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v7_task_contrast as v7
from experiments.gpu_preflight import require_cuda_recurrent

FROZEN_SCHEMA = "pointmaze_temporal_ood_v12_decoupled_belief_frozen.v1"
MODEL_SCHEMA = "pointmaze_temporal_ood_v12_decoupled_belief_model.v1"
SCORE_SCHEMA = "pointmaze_temporal_ood_v12_decoupled_belief_score.v1"
HIDDEN, STATE_EPOCHS, RANK_EPOCHS = 64, 800, 500


class VisualStateEstimator(nn.Module):
    def __init__(self, history_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(history_dim, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, final = self.gru(self.encoder(history)); return self.head(final[-1])


class StateTaskRanker(nn.Module):
    def __init__(self, action_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(4, hidden), nn.SiLU())
        self.candidate_encoder = v6.PulseAwareCandidateEncoder(action_dim, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, state: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(self.state_encoder(state)).chunk(2, dim=-1)
        return self.head((1.0 + scale) * self.candidate_encoder(candidate) + shift).squeeze(-1)


class ActionOnly(nn.Module):
    def __init__(self, action_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__(); self.candidate_encoder = v6.PulseAwareCandidateEncoder(action_dim, hidden); self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, candidate: torch.Tensor) -> torch.Tensor: return self.head(self.candidate_encoder(candidate)).squeeze(-1)


def _norm(x: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=axes).astype(np.float32), np.maximum(x.std(axis=axes), 1e-3).astype(np.float32)


def _model_path(frozen: Path) -> Path: return frozen.with_name(f"{frozen.stem}_model.pt")


def _history_chart(rows: list[dict[str, Any]], observation_chart: str) -> np.ndarray:
    if observation_chart == "latent_history":
        return v6._tensors(rows)[0]
    if observation_chart != "rgb_geometry":
        raise ValueError("unsupported observation chart")
    values = []
    for row in rows:
        value = np.asarray(row.get("source_history_visual_state"), dtype=np.float32)
        if value.shape != (v6.HISTORY_FRAMES, 4) or not np.isfinite(value).all():
            raise ValueError("rgb_geometry requires finite [8,4] source_history_visual_state")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _fit(rows: list[dict[str, Any]], arrays: Any, *, device: str, observation_chart: str = "latent_history") -> tuple[VisualStateEstimator, StateTaskRanker, ActionOnly, dict[str, np.ndarray]]:
    history = _history_chart(rows, observation_chart)
    _latent_history, candidate, state, _goal = v6._tensors(rows); cost = v7._source_task_costs(rows, arrays)
    hm, hs = _norm(history, (0, 1)); sm, ss = _norm(state, (0,)); cm, cs = _norm(candidate, (0, 1)); ym, ys = _norm(cost, (0,))
    unique = np.asarray([next(i for i, row in enumerate(rows) if str(row["instance_id"]) == root) for root in sorted({str(row["instance_id"]) for row in rows})], dtype=np.int64)
    v6._seed(2026103301)
    estimator, ranker, action = VisualStateEstimator(history.shape[-1]).to(device), StateTaskRanker(candidate.shape[-1]).to(device), ActionOnly(candidate.shape[-1]).to(device)
    th = torch.tensor((history - hm) / hs, device=device); ts = torch.tensor((state - sm) / ss, device=device); tc = torch.tensor((candidate - cm) / cs, device=device); ty = torch.tensor((cost - ym) / ys, device=device)
    opt = torch.optim.AdamW(estimator.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(STATE_EPOCHS):
        opt.zero_grad(set_to_none=True); loss = (estimator(th[unique]) - ts[unique]).square().mean(); loss.backward(); opt.step()
    ropt, aopt = torch.optim.AdamW(ranker.parameters(), lr=2e-3, weight_decay=1e-4), torch.optim.AdamW(action.parameters(), lr=2e-3, weight_decay=1e-4)
    groups = v7._root_index_groups(rows)
    for _ in range(RANK_EPOCHS):
        ropt.zero_grad(set_to_none=True); rp = ranker(ts, tc); loss = (rp - ty).square().mean() + v7._rank_loss(rp, ty, groups); loss.backward(); ropt.step()
        aopt.zero_grad(set_to_none=True); ap = action(tc); loss = (ap - ty).square().mean() + v7._rank_loss(ap, ty, groups); loss.backward(); aopt.step()
    return estimator.eval(), ranker.eval(), action.eval(), {"history_mean":hm,"history_scale":hs,"state_mean":sm,"state_scale":ss,"candidate_mean":cm,"candidate_scale":cs,"target_mean":np.asarray(ym,dtype=np.float32),"target_scale":np.asarray(ys,dtype=np.float32)}


def _predict(estimator: VisualStateEstimator, ranker: StateTaskRanker, action: ActionOnly, fit: dict[str, np.ndarray], rows: list[dict[str, Any]], *, cyclic: bool = False, observation_chart: str = "latent_history") -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    history = _history_chart(rows, observation_chart)
    _latent_history,candidate,state,_goal=v6._tensors(rows)
    if cyclic: history=v6._cyclic_root_history(rows,history)
    device=next(estimator.parameters()).device
    with torch.inference_mode():
        h=torch.tensor((history-fit["history_mean"])/fit["history_scale"],device=device); c=torch.tensor((candidate-fit["candidate_mean"])/fit["candidate_scale"],device=device); s=torch.tensor((state-fit["state_mean"])/fit["state_scale"],device=device)
        belief=estimator(h); bc=ranker(belief,c); ac=action(c); oracle=ranker(s,c)
    scale,offset=fit["target_scale"],fit["target_mean"]
    return np.asarray(bc.cpu().numpy()*scale+offset,dtype=np.float64),np.asarray(ac.cpu().numpy()*scale+offset,dtype=np.float64),np.asarray(oracle.cpu().numpy()*scale+offset,dtype=np.float64),np.asarray(belief.cpu().numpy()*fit["state_scale"]+fit["state_mean"],dtype=np.float64)


def _state_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str,float]:
    error=np.asarray(pred)-np.asarray(true); mask=np.abs(true[:,2:])>1e-4
    return {"state_rmse":float(np.sqrt(np.mean(error**2))),"position_rmse":float(np.sqrt(np.mean(error[:,:2]**2))),"velocity_rmse":float(np.sqrt(np.mean(error[:,2:]**2))),"velocity_sign_accuracy":float((np.sign(pred[:,2:])==np.sign(true[:,2:]))[mask].mean()) if bool(mask.any()) else float("nan")}


def _write(path: Path, estimator: VisualStateEstimator, ranker: StateTaskRanker, action: ActionOnly, fit: dict[str,np.ndarray]) -> str:
    torch.save({"schema":MODEL_SCHEMA,"history_dim":int(fit["history_mean"].shape[-1]),"action_dim":int(fit["candidate_mean"].shape[-1]),"estimator":{k:v.detach().cpu() for k,v in estimator.state_dict().items()},"ranker":{k:v.detach().cpu() for k,v in ranker.state_dict().items()},"action":{k:v.detach().cpu() for k,v in action.state_dict().items()},"normalizer":{k:torch.as_tensor(v,dtype=torch.float32) for k,v in fit.items()},"runner_source_sha256":v6._sha256(Path(__file__))},path)
    return v6._sha256(path)


def _load(frozen: dict[str,Any], frozen_path:Path, *, device:str) -> tuple[VisualStateEstimator,StateTaskRanker,ActionOnly,dict[str,np.ndarray]]:
    path=_model_path(frozen_path); expected=str(frozen.get("model_checkpoint_sha256",""))
    if not expected or not path.is_file() or v6._sha256(path)!=expected: raise ValueError("V12 model missing or hash-mismatched")
    payload=torch.load(path,map_location=device,weights_only=True)
    if payload.get("schema")!=MODEL_SCHEMA or payload.get("runner_source_sha256")!=v6._sha256(Path(__file__)): raise ValueError("V12 incompatible model")
    estimator=VisualStateEstimator(int(payload["history_dim"])).to(device); ranker=StateTaskRanker(int(payload["action_dim"])).to(device); action=ActionOnly(int(payload["action_dim"])).to(device)
    estimator.load_state_dict(payload["estimator"]);ranker.load_state_dict(payload["ranker"]);action.load_state_dict(payload["action"])
    normalizer=payload.get("normalizer"); needed={"history_mean","history_scale","state_mean","state_scale","candidate_mean","candidate_scale","target_mean","target_scale"}
    if not isinstance(normalizer,dict) or set(normalizer)!=needed: raise ValueError("V12 normalizer incomplete")
    return estimator.eval(),ranker.eval(),action.eval(),{k:v.detach().cpu().numpy().astype(np.float32,copy=False) for k,v in normalizer.items()}


def _fresh(context:Path, output:Path) -> None:
    if output.exists() or _model_path(output).exists() or (context.parent/"temporal_ood_v12_decoupled_belief_outcomes.json").exists(): raise FileExistsError("V12 calibration sealed")


def calibrate(source:Path,source_arrays:Path,context:Path,calibration_labels:Path,output:Path,*,device:str="cuda",manifest:Path|None=None,observation_chart:str="latent_history")->dict[str,Any]:
    _fresh(context,output); manifest=manifest or context.parent/"temporal_ood_v6_manifest.json"; v6._validate_manifest(manifest,source,source_arrays,context,calibration_labels); runtime=require_cuda_recurrent(device)
    _m,source_rows=v6._read_panel(source,v6.SOURCE_SCHEMA);cm,rows=v6._read_panel(context,v6.CONTEXT_SCHEMA);v6._validate_builder_contract(source_rows,rows,cm,strict_counts=False)
    calibration=[r for r in rows if str(r["split"])=="calibration"]; labels=v6._calibration_costs(calibration_labels,calibration,require_identity=True); costs={v6._key(r):c for r,c in zip(calibration,labels)}; partition=v6.deterministic_calibration_split(rows); auth=[r for r in calibration if str(r["instance_id"]) in set(partition["authorization"])]
    estimator,ranker,action,fit=_fit(source_rows,np.load(source_arrays,allow_pickle=False),device=device,observation_chart=observation_chart); bc,ac,_oracle,_state=_predict(estimator,ranker,action,fit,auth,observation_chart=observation_chart);actual=np.asarray([costs[v6._key(r)] for r in auth]);b,a=v6._selection(auth,bc,actual),v6._selection(auth,ac,actual);gain=v6._paired(b,a)
    output.parent.mkdir(parents=True,exist_ok=True); model_hash=_write(_model_path(output),estimator,ranker,action,fit)
    report={"schema":FROZEN_SCHEMA,"objective":"fixed_decoupled_visual_state_then_state_conditioned_ranking","observation_chart":observation_chart,"fixed_policy":"decoupled_belief","fixed_hidden":HIDDEN,"fixed_state_epochs":STATE_EPOCHS,"fixed_rank_epochs":RANK_EPOCHS,"source_sha256":v6._sha256(source),"source_arrays_sha256":v6._sha256(source_arrays),"context_sha256":v6._sha256(context),"manifest_sha256":v6._sha256(manifest),"calibration_labels_path":str(calibration_labels.resolve()),"calibration_labels_sha256":v6._sha256(calibration_labels),"calibration_partition":partition,"model_checkpoint_sha256":model_hash,"authorization":{"decoupled_belief":b,"action_only":a,"belief_over_action_only":gain,"permitted":bool(gain["lower"]>0.0),"criterion":"fixed decoupled belief; independent authorization lower CI > 0"},"outcome_labels_opened":False,"device":device,"cuda_preflight":runtime,"runtime_contract":{"uses_privileged_state_at_runtime":False,"source_state_role":"source-only estimator/ranker supervision and posthoc diagnostic"}}
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return report


def score(source:Path,source_arrays:Path,context:Path,frozen_selection:Path,output:Path,*,device:str="cuda",manifest:Path|None=None,observation_chart:str="latent_history")->dict[str,Any]:
    runtime=require_cuda_recurrent(device);frozen=json.loads(frozen_selection.read_text())
    if frozen.get("schema")!=FROZEN_SCHEMA or frozen.get("observation_chart", "latent_history") != observation_chart:raise ValueError("V12 frozen chart/schema required")
    manifest=manifest or context.parent/"temporal_ood_v6_manifest.json";calibration=Path(str(frozen["calibration_labels_path"]));v6._validate_manifest(manifest,source,source_arrays,context,calibration)
    if frozen.get("manifest_sha256")!=v6._sha256(manifest) or (frozen.get("source_sha256"),frozen.get("source_arrays_sha256"),frozen.get("context_sha256"))!=(v6._sha256(source),v6._sha256(source_arrays),v6._sha256(context)):raise ValueError("V12 frozen protocol does not bind panel")
    _m,source_rows=v6._read_panel(source,v6.SOURCE_SCHEMA);cm,rows=v6._read_panel(context,v6.CONTEXT_SCHEMA);v6._validate_builder_contract(source_rows,rows,cm,strict_counts=False);estimator,ranker,action,fit=_load(frozen,frozen_selection,device=device);output.mkdir(parents=True,exist_ok=True);labels_path=output/"temporal_ood_v12_decoupled_belief_outcomes.json"
    if labels_path.exists():raise FileExistsError("V12 outcomes already materialized")
    outcome=[r for r in rows if str(r["split"])=="outcome"];labels={v6._key(r):float(np.linalg.norm(v6._simulate_sequence(np.asarray(r["source_state_raw"],dtype=np.float64),np.asarray(r["action_sequence"],dtype=np.float64))[:2]-np.asarray(r["goal_raw"][:2],dtype=np.float64))) for r in outcome};labels_path.write_text(json.dumps({"schema":"pointmaze_temporal_ood_v12_generated_outcomes.v1","generated_after_frozen_selection_sha256":v6._sha256(frozen_selection),"labels":labels},indent=2,sort_keys=True)+"\n")
    bc,ac,oc,state=_predict(estimator,ranker,action,fit,outcome,observation_chart=observation_chart);cbc,_cac,_coc,cstate=_predict(estimator,ranker,action,fit,outcome,cyclic=True,observation_chart=observation_chart);actual=np.asarray([labels[v6._key(r)] for r in outcome]);b,a,o,c=v6._selection(outcome,bc,actual),v6._selection(outcome,ac,actual),v6._selection(outcome,oc,actual),v6._selection(outcome,cbc,actual);true_state=v6._tensors(outcome)[2]
    report={"schema":SCORE_SCHEMA,"outcome_labels_generated_after_frozen_selection":True,"decoupled_belief":b,"action_only":a,"privileged_state_ceiling":o,"cyclic_decoupled_belief":c,"paired_belief_over_action_only":v6._paired(b,a),"paired_belief_over_cyclic":v6._paired(b,c),"paired_state_ceiling_over_belief":v6._paired(o,b),"belief_readout":{"matched_history":_state_metrics(state,true_state),"cyclic_history":_state_metrics(cstate,true_state)},"active_policy":"decoupled_belief" if frozen["authorization"]["permitted"] else "action_only_fallback","active":b if frozen["authorization"]["permitted"] else a,"device":device,"cuda_preflight":runtime,"runtime_contract":{"uses_privileged_state_at_runtime":False,"privileged_ceiling_diagnostic_only":True}}
    (output/"decoupled_belief_score_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");return report


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    for name in ("calibrate","score"):
        p=sub.add_parser(name);p.add_argument("--source",type=Path,required=True);p.add_argument("--source-arrays",type=Path,required=True);p.add_argument("--context",type=Path,required=True);p.add_argument("--manifest",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",choices=("cuda",),default="cuda");p.add_argument("--observation-chart",choices=("latent_history","rgb_geometry"),default="latent_history")
    sub.choices["calibrate"].add_argument("--calibration-labels",type=Path,required=True);sub.choices["score"].add_argument("--frozen-selection",type=Path,required=True);args=vars(parser.parse_args());command=args.pop("command");print(json.dumps(calibrate(**args) if command=="calibrate" else score(**args),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
