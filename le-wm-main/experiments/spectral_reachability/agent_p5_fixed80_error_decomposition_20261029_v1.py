"""Saved-only fixed80 descriptive direction and within-JEPA error decomposition."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

NAME='agent_p5_fixed80_error_decomposition_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
SOURCE='results/agent_p5_fixed164_physical_model_persist_20261029_v1_MAIN01.json'
SOURCE_SHA='25cb6846263027b1ceedd39b9060c3158c6cc6a095d323b012be30728e7511c3'
CAP=8*1024*1024
OWN=(f'experiments/spectral_reachability/{NAME}.py',f'tests/test_{NAME}.py',f'innerdocs/{NAME}.md')
EXPECTED={'baselines':4,'branches':164,'fresh_certified_roots':0,'known_roots':2,'pairs':80}
METRICS=('native_distance','object_position_error','wrapped_object_angle_error')
JEPA=('H1_20','H2_20','short_20','H2_15','short_15')
CONTRASTS=(('H2_20','H1_20'),('short_20','H1_20'),('short_20','H2_20'),('short_15','H2_15'))
def need(value,msg):
 if not value: raise ValueError(msg)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def duplicate(pairs):
 out={}
 for k,v in pairs:
  need(k not in out,'duplicate key'); out[k]=v
 return out
def file(path,expected=None,cap=CAP):
 p=Path(path); need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'regular bounded file'); raw=p.read_bytes()
 if expected is not None: need(sha(raw)==expected,'source SHA')
 return raw
def decode(raw): return json.loads(raw,object_pairs_hook=duplicate,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def closure(): return {name:sha(file(ROOT/name)) for name in OWN}
def numeric(v):
 if v is None: return None
 need(type(v) in (float,int) and math.isfinite(v),'nonfinite or nonnumeric scalar'); return v
def diff(a,b):
 a,b=numeric(a),numeric(b)
 if a is None or b is None: return None
 value=a-b; need(math.isfinite(value),'nonfinite difference'); return value
def sign(x): return None if x is None else ('positive' if x>0 else 'negative' if x<0 else 'zero')
def agreement(a,b):
 if a is None or b is None: return 'missing'
 if a=='zero' and b=='zero': return 'both_zero'
 if a=='zero' or b=='zero': return 'one_zero'
 return 'same' if a==b else 'opposite'
def analyze(x):
 need(x['schema']=='agent_p5_fixed164_physical_model_persist_20261029_v1.result' and x['denominator']==EXPECTED and x['cross_model_spaces_comparable'] is False and x['P1_admission'] is False and x['P5_admission'] is False and x['full_strict_rerun'] is False,'source contract')
 a=x['analysis']; need(a['schema']=='agent_p5_fixed164_physical_model_join_20261029_v1.analysis' and a['status']=='fixed164_saved_identity_join_only' and a['known_roots']==2 and a['fresh_roots']==0 and a['baseline_indices']==[0,41,82,123] and a['P1_admission'] is False and a['P5_admission'] is False,'join contract')
 rows,pairs=a['rows'],a['pairs']; need(len(rows)==164 and len(pairs)==80,'whole denominator')
 need(all(r['index']==i and (r['root_slot'],r['condition'])==((i//41)//2,('off','on')[(i//41)%2]) for i,r in enumerate(rows)),'fixed branch identity')
 out=[]
 for i,p in enumerate(pairs):
  slot=i//20; j=i%20; minus=41*slot+1+2*j; plus=minus+1
  need(p==dict(slot=slot,coord=j%10,h=(.01,.001)[j//10],minus_index=minus,plus_index=plus),'fixed 80 schedule')
  left,right=rows[minus],rows[plus]
  need(left['root_slot']==right['root_slot'] and left['condition']==right['condition'] and left['task_id']==right['task_id'],'pair source identity')
  q=diff(right['lewm_actual_goal_Q'],left['lewm_actual_goal_Q'])
  need(all(k in row['native_final_metrics'] for row in (left,right) for k in METRICS),'native expected metric keys')
  native={k:diff(right['native_final_metrics'][k],left['native_final_metrics'][k]) for k in METRICS}
  js=[r['jepa_common_label_MSE'] for r in (left,right)]
  need(all(set(JEPA)<=set(z) for z in js),'JEPA expected phase keys')
  pair_jepa={k:diff(js[1][k],js[0][k]) for k in JEPA}
  contrast={f'{u}_minus_{v}':dict(minus=diff(js[0][u],js[0][v]),plus=diff(js[1][u],js[1][v])) for u,v in CONTRASTS}
  out.append(dict(**p,root_slot=left['root_slot'],condition=left['condition'],task_id=left['task_id'],native_plus_minus=native,lewm_actual_Q_plus_minus=q,
   direction_agreement={k:agreement(sign(v),sign(q)) for k,v in native.items()},jepa_MSE_plus_minus=pair_jepa,jepa_within_branch_error_contrasts=contrast))
 summary={}
 for metric in METRICS:
  summary[metric]={}
  for group in ('all','slot0','slot1','slot2','slot3'):
   subset=out if group=='all' else [r for r in out if r['slot']==int(group[-1])]
   counts={k:0 for k in ('same','opposite','both_zero','one_zero','missing')}
   for row in subset: counts[row['direction_agreement'][metric]]+=1
   summary[metric][group]=dict(denominator=len(subset),counts=counts)
 return dict(schema=NAME+'.analysis',source_sha256=SOURCE_SHA,source_closure=x['source_closure'],fixed_denominator=80,branch_denominator=164,baseline_indices=[0,41,82,123],known_roots=2,fresh_certified_roots=0,pairs=out,direction_summary=summary,
  interpretation='plus minus within each physical metric and LeWM actual goal-Q; sign comparison only, not numerical cross-space subtraction; JEPA errors compare within same model/time; observational known roots, no causal or independent confirmation',
  P1_admission=False,P5_admission=False,new_neural_calls=0,new_physics_calls=0,full_strict_rerun=False)
def build():
 before=closure(); value=analyze(decode(file(ROOT/SOURCE,SOURCE_SHA))); need(closure()==before,'own source changed'); value['analysis_source_closure']=before; return value
def target(path):
 p=Path(path).absolute(); d=ROOT/'results'; need(d.resolve()==d and not d.is_symlink() and p.parent==d and p.name.startswith(NAME+'_') and p.suffix=='.json','exclusive results namespace'); return p
def write(p,data):
 need(len(data)<=CAP and shutil.disk_usage(p.parent).free>=200*1024*1024+len(data),'output cap/floor')
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
 with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try: os.fsync(fd)
 finally: os.close(fd)
def main():
 parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
 sub.add_parser('analyze'); p=sub.add_parser('write'); p.add_argument('--output',required=True)
 p=sub.add_parser('verify'); p.add_argument('--analysis',required=True); p.add_argument('--sha256',required=True)
 a=parser.parse_args(); data=canonical(build())
 if a.cmd=='analyze': print(data.decode(),end='')
 elif a.cmd=='write':
  p=target(a.output); write(p,data); print(json.dumps(dict(status='saved',path=str(p),sha256=sha(data),bytes=len(data))))
 else:
  need(len(a.sha256)==64 and all(c in '0123456789abcdef' for c in a.sha256),'expected SHA'); raw=file(target(a.analysis),a.sha256); need(raw==data,'whole reconstructed analysis mismatch')
  print(json.dumps(dict(status='exact_saved_analysis_verified',analysis_sha256=a.sha256,full_strict_rerun=False)))
if __name__=='__main__': main()
