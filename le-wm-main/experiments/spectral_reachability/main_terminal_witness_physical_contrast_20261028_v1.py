"""Independent native7 arithmetic after fixed terminal-witness dispatch/consume."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='main_terminal_witness_physical_contrast_20261028_v1'
OLD=ROOT/'results/consume_mixedmode_physical_MAIN_20261028_v1_MAIN01.json'
OLD_SHA='eb3c2752dc6ae93d99319439895c99dd136b9847a81cc83dfb8ec71b6ace9e75'
OLD_REPORT=ROOT/'results/agent_mixedmode_physical_20261028_v1_MAIN01/report.json'
OLD_REPORT_SHA='58bd3b4fcb2042e79d796f3eb26e0004f88b098c70ffdb044d1b03a3eedcc965'
NEW_REPORT=ROOT/'results/agent_terminal_witness_physical_20261028_v1_MAIN01/report.json'
NEW_REPORT_SHA='d02868f33f5b4b348c53d3684f30d2681124fc391b4f8d0f9b1a9e8992a04850'


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p,h):
    p=Path(p);require(p.is_file() and not p.is_symlink() and sha(p)==h,'external exact artifact SHA')
    return json.loads(p.read_bytes())


def metrics(state,goal):
    s,g=np.asarray(state,dtype=float),np.asarray(goal,dtype=float)
    require(s.shape==g.shape==(7,) and np.isfinite(s).all() and np.isfinite(g).all(),'finite whole native7')
    d=s-g
    angle=abs(float(d[4]));native_angle=min(angle,2*np.pi-angle)
    joint=float(np.linalg.norm(d[:4]))
    return dict(joint_xy=joint,agent_xy=float(np.linalg.norm(d[:2])),object_xy=float(np.linalg.norm(d[2:4])),
        angle_wrapped=abs(float(np.arctan2(np.sin(d[4]),np.cos(d[4])))),
        native_angle_term=native_angle,velocity=float(np.linalg.norm(d[5:])),
        native_distance=float(np.linalg.norm(d)),native_success=bool(joint<20 and native_angle<np.pi/9))


def pair(old,new,goal):
    a,b=metrics(old,goal),metrics(new,goal)
    return dict(old=a,new=b,delta_new_minus_old={k:b[k]-a[k] for k in a if k!='native_success'})


def build(receipt,receipt_sha):
    old=read(OLD,OLD_SHA);original=read(OLD_REPORT,OLD_REPORT_SHA)
    new=read(NEW_REPORT,NEW_REPORT_SHA);strict=read(receipt,receipt_sha)
    require(old['schema']=='MAIN.mixedmode_physical_receipt.v1' and old['receipt']['rows']==original['rows']
        and old['receipt']['physical_report_sha256']==OLD_REPORT_SHA,'accepted whole old rows including explicit inheritance')
    require(new['status']==strict['status']=='terminal_witness4_physical_recorded_not_admitted'
        and strict['schema']=='agent_terminal_witness_physical_20261028_v1.consume'
        and strict['physical_report_sha256']==NEW_REPORT_SHA and strict['rows']==new['rows']
        and strict['errors']==new['errors']==[] and strict['saved_PNG_verified']==264
        and strict['P1_admission'] is strict['freshness'] is False,'complete new full native strict consumption')
    require(len(original['rows'])==len(new['rows'])==4,'all four fixed conditions')
    rows=[]
    for i,(a,b,prior) in enumerate(zip(original['rows'],new['rows'],old['independent_native_metrics'],strict=True)):
        require(a['index']==b['index']==i and all(a[k]==b[k] for k in ('root_slot','condition','task_id'))
            and all(a['plan'][k]==b['plan'][k] for k in ('task_sha256','blind_sha256','source_report_sha256'))
            and b['status']=='horizon_completed' and b['own_returned']==b['warmup_returned']==10,
            'same original condition/task/blind and whole10 execution')
        goal=b['goal'];result=pair(a['final_state'],b['final_state'],goal)
        require(result['old']['joint_xy']==prior['current']['joint_xy_error']
            and result['old']['native_distance']==a['final_metrics']['native_distance']
            and result['new']['native_distance']==b['final_metrics']['native_distance']
            and result['old']['native_success']==a['native_success']
            and result['new']['native_success']==b['native_success'],'independent native whole-state metrics')
        same=np.asarray(a['plan']['commands'],np.float32).tobytes()==np.asarray(b['plan']['commands'],np.float32).tobytes()
        if same:require(a['final_state']==b['final_state'],'same literal task/action endpoint exact replay control')
        require([c['step'] for c in b['checkpoints']]==[0,5,10],'same time all checkpoints')
        checkpoints={c['step']:metrics(c['state'],goal) for c in b['checkpoints']}
        rows.append(dict(index=i,root_slot=b['root_slot'],condition=b['condition'],
            old_index=a['plan']['selected_index'],new_index=b['plan']['selected_index'],same_raw10=same,
            predicted_Q_delta=b['plan']['nonlinear_terminal_Q']-a['plan']['model_Q'],
            model_prefix_uphill=b['plan']['model_prefix_uphill_from_anchor'],
            actual_checkpoints=checkpoints,**result))
    keys=rows[0]['delta_new_minus_old']
    return dict(schema=NAME,rows=rows,counts={k:dict(better=sum(r['delta_new_minus_old'][k]<0 for r in rows),
        worse=sum(r['delta_new_minus_old'][k]>0 for r in rows),tie=sum(r['delta_new_minus_old'][k]==0 for r in rows)) for k in keys},
        old_success=sum(r['old']['native_success'] for r in rows),new_success=sum(r['new']['native_success'] for r in rows),
        known_roots=2,conditions=4,fresh_roots=0,new_neural_calls=0,new_physics_calls=0,
        full_upstream_rebuilt_here=False,P1_admission=False,
        inputs=dict(old_receipt_sha256=OLD_SHA,old_report_sha256=OLD_REPORT_SHA,
            new_report_sha256=NEW_REPORT_SHA,new_receipt=str(receipt),new_receipt_sha256=receipt_sha),
        caveat='Fixed entire mixed222 panel; post-dispatch descriptive arithmetic, not root/selector tuning or unique-cause proof.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('receipt','receipt-sha','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();result=build(a.receipt,a.receipt_sha);out=Path(a.output).absolute()
    require(out.parent==ROOT/'results' and out.name.startswith(NAME+'_'),'exclusive own output')
    with out.open('xb') as f:f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out),counts=result['counts'],rows=result['rows'])))


if __name__=='__main__':main()
