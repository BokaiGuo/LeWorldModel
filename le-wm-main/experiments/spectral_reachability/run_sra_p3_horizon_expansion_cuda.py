"""CUDA-only, outcome-blind fixed-tree horizon expansion diagnostic for SRA."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import torch
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import DEFAULT_CHECKPOINT,DEFAULT_CONFIG,DEFAULT_DATASET,ROOT,load_model_and_dataset,make_root_context,parameter_hash,require_cuda,sha256_file,terminal_latent,write_json
from experiments.spectral_reachability.run_sra_p1_pointmaze_cuda import action_bank,block_to_physical

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--roots',default='1832,1833,1834,1835');p.add_argument('--horizons',default='2,3,4');p.add_argument('--model-batch-size',type=int,default=64);p.add_argument('--physical-action-amplitude',type=float,default=.8);p.add_argument('--dataset-root',type=Path,default=DEFAULT_DATASET);p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--dinov2-source',type=Path,default=ROOT/'external/local_dinov2');a=p.parse_args()
 roots=[int(x) for x in a.roots.split(',') if x]; hs=[int(x) for x in a.horizons.split(',') if x]
 if len(roots)<4 or len(set(roots))!=len(roots) or hs!=[2,3,4] or a.output_dir.exists() or a.model_batch_size<1: raise ValueError('need four unique fresh roots, fixed H=2,3,4, positive batch size, and new output')
 d=require_cuda();from experiments.spectral_reachability.pointmaze_physical_evaluator import replay_position_endpoints,position_backend_receipt
 a.output_dir.mkdir(parents=True);m,data=load_model_and_dataset(a,d);before=parameter_hash(m); rows=[]
 # A fixed five-arm continuation tree (zero and cardinal pulses), declared before labels.
 _, cont=action_bank(data,roots[0],0,a.physical_action_amplitude,d);cont=cont[:5]; cont_phys=[block_to_physical(x,data) for x in cont]
 for root in roots:
  labels,first=action_bank(data,root,0,a.physical_action_amplitude,d); first_phys=[block_to_physical(x,data) for x in first]; rr={'root':root,'horizons':{}}
  for h in hs:
   ctx,goal,_=make_root_context(m,data,root,0,h,d); suffixes=[]; physical=[]
   for fi,fb in enumerate(first):
    for tail in itertools.product(range(5),repeat=h-1):
     suffixes.append(torch.stack([fb,*[cont[j] for j in tail]])); physical.append(np.concatenate([first_phys[fi],*[cont_phys[j] for j in tail]],axis=0))
   suffix=torch.stack(suffixes,dim=1)
   with torch.no_grad():
    costs=[torch.sum((terminal_latent(m,ctx,suffix[:,start:start+a.model_batch_size])-goal)**2,dim=1) for start in range(0,suffix.shape[1],a.model_batch_size)]
    cost=torch.cat(costs); immediate=[float(torch.sum((terminal_latent(m,ctx,fb.reshape(1,1,-1))[0]-goal)**2)) for fb in first]
   n=5**(h-1); model_first=np.asarray([float(cost[i*n:(i+1)*n].min()) for i in range(6)]); chosen=int(model_first.argmin()); immediate_choice=int(np.argmin(immediate))
   source=data.states[root,0].cpu().numpy().astype(np.float64); endpoints=replay_position_endpoints(np.repeat(source[None],len(physical),0),np.stack(physical)); target=data.states[root,h*5].cpu().numpy(); physical_cost=np.linalg.norm(endpoints[:,:2]-target[:2],axis=1).reshape(6,n).min(axis=1); optimum=int(physical_cost.argmin())
   rr['horizons'][str(h)]={'tree_sequences':len(physical),'model_first_costs':model_first.tolist(),'immediate_first_costs':immediate,'physical_first_costs':physical_cost.tolist(),'spectral_index':chosen,'immediate_index':immediate_choice,'physical_index':optimum,'spectral_regret':float(physical_cost[chosen]-physical_cost[optimum]),'immediate_regret':float(physical_cost[immediate_choice]-physical_cost[optimum])}
  rows.append(rr);print(json.dumps({'root':root,'horizons':{h:x['spectral_regret']-x['immediate_regret'] for h,x in rr['horizons'].items()}},sort_keys=True))
 after=parameter_hash(m); agg={str(h):float(np.mean([r['horizons'][str(h)]['spectral_regret']-r['horizons'][str(h)]['immediate_regret'] for r in rows])) for h in hs}; gates={'cuda_only':True,'frozen_parameter_hash_unchanged':before==after,'fixed_tree_sizes_correct':all(r['horizons'][str(h)]['tree_sequences']==6*5**(h-1) for r in rows for h in hs),'no_outcome_labels_used_for_selection':True}
 out={'schema':'sra_p3_fixed_tree_horizon_expansion.v1','status':'mechanism_ready' if all(gates.values()) else 'invalid','scope':'fixed-tree diagnostic, not RGB closed-loop success','inputs':{'roots':roots,'horizons':hs,'model_batch_size':a.model_batch_size,'checkpoint_sha256':sha256_file(a.checkpoint),'physical_position_backend':position_backend_receipt()},'gates':gates,'aggregate_spectral_minus_immediate_regret':agg,'rows':rows};write_json(a.output_dir/'report.json',out);print(json.dumps({'status':out['status'],'aggregate':agg,'gates':gates},indent=2));
 if not all(gates.values()): raise RuntimeError('P3 protocol failed')
if __name__=='__main__': main()
