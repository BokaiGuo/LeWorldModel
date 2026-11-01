"""Durably persist and separately rebuild the fixed164 saved-only identity join."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from experiments.spectral_reachability import agent_p5_fixed164_physical_model_join_20261029_v1 as join

NAME='agent_p5_fixed164_physical_model_persist_20261029_v1'
ROOT=join.ROOT
CAP=8*1024*1024
FLOOR=200*1024*1024
SEALED={
 'experiments/spectral_reachability/agent_p5_fixed164_physical_model_join_20261029_v1.py':'4cbf388374e5fd3c279ee90ac794dc6a720fc7f6027c3721a95a0201d92d5d3a',
 'tests/test_agent_p5_fixed164_physical_model_join_20261029_v1.py':'1c9490dd6a970c58b84ba7eefb2cd2d03752bc09324ed5b61d828a50b92edb70',
 'innerdocs/agent_p5_fixed164_physical_model_join_20261029_v1.md':'55a7de4b048229d936e7670268fff401ee362f70b0a9d07b99979f3caa826d05',
}
OWN=(f'experiments/spectral_reachability/{NAME}.py',f'tests/test_{NAME}.py',f'innerdocs/{NAME}.md')

def digest(data): return hashlib.sha256(data).hexdigest()
def regular(path,cap=CAP):
 p=Path(path); join.need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'regular bounded file'); return p.read_bytes()
def closure():
 out={}
 for name in (*SEALED,*OWN):
  value=digest(regular(ROOT/name)); join.need(name not in SEALED or value==SEALED[name],'sealed join source changed'); out[name]=value
 return out
def build():
 before=closure(); analysis=join.analyze(join.load())
 join.need(closure()==before,'source changed during rebuild')
 join.need(analysis['schema']==join.NAME+'.analysis' and analysis['status']=='fixed164_saved_identity_join_only' and len(analysis['rows'])==164 and len(analysis['pairs'])==80 and analysis['baseline_indices']==[0,41,82,123] and analysis['known_roots']==2 and analysis['fresh_roots']==0 and analysis['P1_admission'] is False and analysis['P5_admission'] is False,'fixed cohort and gates')
 join.need(analysis['input_sha256']=={k:v[1] for k,v in join.FILES.items()},'fixed source hash bindings')
 return dict(schema=NAME+'.result',source_closure=before,analysis=analysis,denominator=dict(branches=164,pairs=80,baselines=4,known_roots=2,fresh_certified_roots=0),
  missing_policy='no imputation: original join rejects incomplete terminal source',cross_model_spaces_comparable=False,
  interpretation='same saved physical branch; JEPA and LeWM have different feature spaces and input permissions; not a matched model ranking or causal result',
  P1_admission=False,P5_admission=False,new_neural_calls=0,new_physics_calls=0,full_strict_rerun=False)
def target(value):
 p=Path(value).absolute(); directory=ROOT/'results'
 join.need(directory.resolve()==directory and not directory.is_symlink(),'exact results directory')
 join.need(p.parent==directory and p.name.startswith(NAME+'_') and p.suffix=='.json','exclusive result namespace')
 return p
def write(path,data):
 join.need(len(data)<=CAP,'output cap'); join.need(shutil.disk_usage(path.parent).free>=FLOOR+len(data),'disk floor')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
 with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
 dfd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try: os.fsync(dfd)
 finally: os.close(dfd)
def verify(path,expected):
 join.need(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'expected SHA')
 data=regular(path); join.need(digest(data)==expected,'result SHA')
 join.need(data==join.canonical(build()),'whole saved content or closure differs')
 return dict(schema=NAME+'.verify',status='exact_rebuild_verified',result_sha256=expected,full_strict_rerun=False,new_neural_calls=0,new_physics_calls=0)
def main():
 parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
 p=subs.add_parser('write',help='exclusive file, fsync file and directory'); p.add_argument('--output',required=True)
 p=subs.add_parser('verify',help='rebuild complete analysis from original saved inputs'); p.add_argument('--result',required=True); p.add_argument('--sha256',required=True)
 a=parser.parse_args()
 if a.command=='write':
  path=target(a.output); data=join.canonical(build()); write(path,data); result=dict(status='saved',path=str(path),sha256=digest(data),bytes=len(data))
 else: result=verify(target(a.result),a.sha256)
 print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=='__main__': main()
