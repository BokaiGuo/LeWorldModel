"""Fixed164 saved-only identity join; no new neural or physical calls."""
import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

NAME='agent_p5_fixed164_physical_model_join_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
FILES={
 'comparison':('results/agent_p5_short_start_comparison_20261029_v1_MAIN01.json','bbafa5ec58dde0f0a12d01aded1f2d49a6a869bfe1c73e0d638eb12a6c57ff24'),
 'short_strict':('results/agent_p5_short_start_full_roll_20261029_v1_consume_MAIN01.json','3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd'),
 'physical':('results/agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json','8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb'),
 'rgb':('results/agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json','d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2'),
}
def need(ok,msg):
 if not ok: raise ValueError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  need(k not in d,'duplicate JSON key'); d[k]=v
 return d
def read(path,expected,cap=128*1024*1024):
 p=Path(path); need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'regular bounded input')
 b=p.read_bytes(); need(sha(b)==expected,'input SHA'); return json.loads(b,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
def load(): return {k:read(ROOT/p,s) for k,(p,s) in FILES.items()}
def number(v):
 need(type(v) in (float,int) and math.isfinite(v),'finite scalar'); return v
def rawhash(commands):
 need(type(commands) is list and len(commands)==10 and all(type(x) is list and len(x)==2 for x in commands),'raw10 shape')
 return sha(b''.join(struct.pack('<f',number(v)) for row in commands for v in row))
def analyze(v):
 c,s,p,r=(v[k] for k in ('comparison','short_strict','physical','rgb'))
 need(c['schema']=='agent_p5_short_start_comparison_20261029_v1.analysis' and c['status']=='all164_and80_common_H2_label_saved_rescored','comparison ABI')
 need(s['schema']=='agent_p5_short_start_full_roll_20261029_v1.consume' and s['status']=='all164_short_start_causal_roll_saved_verified' and s['branches']==164 and s['pairs']==80,'short strict')
 need(c['evidence']['new_complete_strict_sha256']==FILES['short_strict'][1] and c['evidence']['new_report_sha256']==s['report_sha256'],'short strict binding')
 need(p['schema']=='agent_observed_raw_coordinate_physical_20261028_v1.consume' and p['status']=='observed_raw164_physical_recorded_not_admitted' and p['fixed_denominator']==164 and p['known_roots']==2 and p['fresh_roots']==0,'physical strict')
 need(r['schema']=='agent_observed_raw_coordinate_rgb_20261028_v1.consume' and r['status']=='raw164_RGB_verified_not_admitted' and r['fixed_denominator']==164 and r['known_roots']==2 and r['fresh_roots']==0,'RGB strict')
 key=str(ROOT/FILES['physical'][0]); need(c['bindings'][key]==FILES['physical'][1],'comparison physical source binding')
 need(r['physical_input_sha256']['physical_receipt_sha']==FILES['physical'][1] and r['physical_input_sha256']['physical_report_sha']==p['physical_report_sha256'],'RGB physical source binding')
 cr,pr,rr=c['rows'],p['rows'],r['rows']; need(len(cr)==len(pr)==len(rr)==164 and len(c['pairs'])==80,'fixed denominator')
 out=[]
 for i,(a,b,d) in enumerate(zip(cr,pr,rr,strict=True)):
  keys=('index','root_slot','condition','task_id','selection_digest')
  need(all(a[k]==b[k]==d[k] for k in keys) and a['index']==i,'row identity')
  need((a['root_slot'],a['condition'])==((i//41)//2,('off','on')[(i//41)%2]),'fixed four groups')
  plan=b['plan']; raw=rawhash(plan['commands'])
  need(plan['index']==i and raw==plan['raw_float32_sha256']==d['raw_float32_sha256'] and plan['producer_job_sha256']==d['producer_job_sha256'],'raw and producer identity')
  need(plan['model_task_binding']==d['model_task_binding'] and b['initial_exact'] and d['prefix_reference_exact'],'task and physical root')
  need(b['status']=='horizon_completed' and d['physical_status']==b['status'] and not d['latent_missing'],'terminal available')
  cp=[x for x in b['checkpoints'] if x['step']==10]; need(len(cp)==1 and d['source']['step']==10,'actual plan10')
  need(all(cp[0][k]==d['source'][k] for k in ('file_sha256','pixel_sha256','state')),'actual PNG and state')
  need(d['evaluation']['final_metrics']==b['final_metrics'] and d['evaluation']['final_state']==b['final_state'] and d['evaluation']['native_success']==b['native_success'],'native metrics join')
  for val in a['common_MSE'].values(): number(val)
  number(d['actual_Q'])
  out.append(dict(index=i,root_slot=a['root_slot'],condition=a['condition'],task_id=a['task_id'],selection_digest=a['selection_digest'],raw_float32_sha256=raw,producer_job_sha256=d['producer_job_sha256'],plan10_png_sha256=cp[0]['file_sha256'],native_success=b['native_success'],native_final_metrics=b['final_metrics'],lewm_actual_goal_Q=d['actual_Q'],jepa_common_label_MSE=a['common_MSE']))
 pairs=[]
 for i,x in enumerate(c['pairs']):
  panel=i//20; local=i%20; minus=41*panel+1+2*local; plus=minus+1
  need(x['slot']==panel and x['minus_index']==minus and x['plus_index']==plus and x['coord']==local%10 and x['h']==(0.01,0.001)[local//10] and x['root_slot']==out[minus]['root_slot'] and x['condition']==out[minus]['condition'],'pair schedule')
  pairs.append(dict(slot=panel,coord=x['coord'],h=x['h'],minus_index=minus,plus_index=plus))
 return dict(schema=NAME+'.analysis',status='fixed164_saved_identity_join_only',input_sha256={k:x[1] for k,x in FILES.items()},rows=out,pairs=pairs,baseline_indices=[0,41,82,123],known_roots=2,fresh_roots=0,P1_admission=False,P5_admission=False,new_neural_calls=0,new_physics_calls=0,full_strict_rerun=False,interpretation='same physical branch, different model inputs and representations; no cross-model MSE ranking, causal inference or independent confirmation')
def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def main():
 parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
 sub.add_parser('analyze'); p=sub.add_parser('verify'); p.add_argument('--analysis',required=True); p.add_argument('--sha256',required=True)
 a=parser.parse_args(); value=canonical(analyze(load()))
 if a.cmd=='analyze': print(value.decode(),end='')
 else:
  saved=read(a.analysis,a.sha256,8*1024*1024); need(canonical(saved)==value,'whole analysis mismatch')
  print(json.dumps(dict(status='exact_saved_join_verified',analysis_sha256=a.sha256)))
if __name__=='__main__': main()
