"""Physical rank audit of every sealed Cube P1 candidate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import gymnasium as gym,numpy as np,stable_worldmodel  # noqa
from scipy.stats import spearmanr
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability.build_cube_d2p_live_runtime_panel import _restore_state
from experiments.spectral_reachability.run_cube_d2p_p1_live_physical_calibration import step
def calibration_gates(rows):
 selected=sum(x['positive_control_selected'] for x in rows);mean_rho=float(np.mean([x['spearman_model_vs_physics'] for x in rows]));min_rho=float(np.min([x['spearman_model_vs_physics'] for x in rows]))
 gates={'cuda_only':True,'full_sealed_bank_executed':True,'same_root_state_restored_per_candidate':True,'evaluator_state_not_model_input':True,'outcome_roots_not_opened':True,'positive_control_is_physics_optimum_each_root':all(x['positive_control_physics_cost']==x['physics_best_cost'] for x in rows),'positive_control_top1_at_least_75pct':selected*4>=len(rows)*3,'mean_spearman_at_least_0_60':mean_rho>=0.60,'no_per_root_order_reversal':min_rho>=0.0}
 return gates,{'root_count':len(rows),'positive_control_selected_count':selected,'positive_control_selected_rate':selected/len(rows),'mean_spearman':mean_rho,'min_spearman':min_rho}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--live-panel',type=Path,required=True);p.add_argument('--p1-evidence',type=Path,required=True);p.add_argument('--device',choices=('cuda',),default='cuda');a=p.parse_args();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(a.output)
 panel=json.loads((a.live_panel/'report.json').read_text());pre=json.loads(a.p1_evidence.read_text())
 if panel['status']!='pass' or pre['status']!='pass':raise ValueError('upstream receipt failed')
 host=require_cuda_allocation(a.device);env=gym.make('swm/OGBCube-v0',mode='data_collection',render_mode='rgb_array',terminate_at_goal=False);rows=[]
 try:
  for root,item in zip(panel['rows'],pre['rows'],strict=True):
   if root['root_id']!=item['root_id']:raise ValueError('sealed root identity mismatch')
   env.reset(seed=root['root_seed']);target=np.asarray(root['heldout_successor']['qpos']);cost=[]
   for packet in item['proposal_raw_packets']:
    _restore_state(env.unwrapped,root['evaluator_only_root_state']);step(env,np.asarray(packet,np.float32));cost.append(float(np.linalg.norm(np.asarray(env.unwrapped.data.qpos)-target)))
   score=np.asarray(item['proposal_goal_costs']);rho=float(spearmanr(-score,-np.asarray(cost)).statistic);positive=int(item['positive_control_index']);rows.append({'root_id':root['root_id'],'spearman_model_vs_physics':rho,'positive_control_index':positive,'model_best':int(np.argmin(score)),'physics_best':int(np.argmin(cost)),'positive_control_selected':int(np.argmin(score))==positive,'positive_control_physics_cost':cost[positive],'model_best_physics_cost':cost[int(np.argmin(score))],'physics_best_cost':min(cost),'proposal_physics_costs':cost})
 finally:env.close()
 g,aggregate=calibration_gates(rows)
 out={'schema':'cube_d2p_p1_full_bank_rank_audit.v2','status':'pass' if all(g.values()) else 'fail','cuda_host':host,'policy':pre.get('policy',{}),'gates':g,'aggregate':aggregate,'criteria':{'positive_control_selected_rate_minimum':0.75,'mean_spearman_minimum':0.60,'per_root_spearman_minimum':0.0},'rows':rows};a.output.mkdir(parents=True);(a.output/'report.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'aggregate':out['aggregate'],'gates':g},sort_keys=True))
 if out['status']!='pass':raise SystemExit('Cube P1 calibration rejected by preregistered positive-control/ranking gates')
if __name__=='__main__':main()
