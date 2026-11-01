"""CUDA physical calibration for sealed Cube P1 prefixes (not an outcome)."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability.build_cube_d2p_live_runtime_panel import _restore_state
from experiments.spectral_reachability.cube_d2p_p1_evidence_binding import validate_binding

SCHEMA="cube_d2p_p1_live_physical_calibration.v1"
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def step(env,packet):
    for a in np.asarray(packet,np.float32):
        _,_,term,trunc,_=env.step(a)
        if term or trunc: raise RuntimeError("sealed calibration packet terminated")
def run(env,snapshot,packet):
    _restore_state(env.unwrapped,snapshot); step(env,packet)
    return np.asarray(env.unwrapped.data.qpos,np.float64).copy(),np.asarray(env.unwrapped.data.qvel,np.float64).copy(),np.asarray(env.render(),np.uint8).copy()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--live-panel',type=Path,required=True);p.add_argument('--p1-evidence',type=Path,required=True);p.add_argument('--device',choices=('cuda',),default='cuda');a=p.parse_args();a.output=a.output.resolve();a.live_panel=a.live_panel.resolve();a.p1_evidence=a.p1_evidence.resolve()
 if a.output.exists():raise FileExistsError(a.output)
 panel=json.loads((a.live_panel/'report.json').read_text()); p1=json.loads(a.p1_evidence.read_text())
 binding=validate_binding(a.live_panel,a.p1_evidence,Path(p1['policy']['path']))
 panel=binding['panel'];p1_rows=binding['ordered_model_rows']
 host=require_cuda_allocation(a.device); env=gym.make('swm/OGBCube-v0',mode='data_collection',render_mode='rgb_array',terminate_at_goal=False);rows=[]
 try:
  for root,sel in zip(panel['rows'],p1_rows,strict=True):
   if root['root_id']!=sel['root_id']:raise ValueError('P1 root identity mismatch')
   env.reset(seed=int(root['root_seed'])); packet=np.asarray(sel['selected_raw_packet'],np.float32);zero=np.zeros_like(packet)
   sq,sv,srgb=run(env,root['evaluator_only_root_state'],packet);zq,zv,zrgb=run(env,root['evaluator_only_root_state'],zero);rq,rv,rrgb=run(env,root['evaluator_only_root_state'],packet)
   target=np.asarray(root['heldout_successor']['qpos'],np.float64); rows.append({'root_id':root['root_id'],'selected_qpos_l2_to_sealed_successor':float(np.linalg.norm(sq-target)),'zero_qpos_l2_to_sealed_successor':float(np.linalg.norm(zq-target)),'selected_replay_qpos_max_abs_error':float(np.max(np.abs(sq-rq))),'selected_replay_qvel_max_abs_error':float(np.max(np.abs(sv-rv))),'selected_replay_rgb_max_abs_error':float(np.max(np.abs(srgb.astype(np.int16)-rrgb.astype(np.int16))))})
 finally:env.close()
 g={'cuda_only':True,'same_root_selected_vs_raw_zero':True,'sealed_prefix_identity_bound':True,'sealed_input_and_policy_hashes_verified':True,'selected_physics_replay_exact':all(x['selected_replay_qpos_max_abs_error']==0 and x['selected_replay_qvel_max_abs_error']==0 for x in rows),'evaluator_state_not_model_input':True,'outcome_roots_not_opened':True}
 out={'schema':SCHEMA,'status':'pass' if all(g.values()) else 'fail','scope':'CUDA-only same-root physical calibration; no outcome/effectiveness claim.','cuda_host':{**host,'python':platform.python_version()},'upstream':{'live_panel_sha256':sha(a.live_panel/'report.json'),'p1_sha256':sha(a.p1_evidence),**binding['hashes']},'gates':g,'rows':rows}
 a.output.mkdir(parents=True);(a.output/'report.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'gates':g,'output':str(a.output)},sort_keys=True))
if __name__=='__main__':main()
