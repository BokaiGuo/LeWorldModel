"""Independent native metrics for fixed allmode original vs nonlinear suffix4."""
import argparse
import json
from pathlib import Path
from experiments.spectral_reachability import main_terminal_witness_physical_contrast_20261028_v1 as base

ROOT=base.ROOT
NAME='main_nonlinear_suffix_physical_contrast_20261028_v1'
OLD_SHA='3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4'
NEW_SHA='30f5131d6f70abdbe669ce0aef86a430c1ba6593e40149c2402801b6d0c37c8a'
RECEIPT_SHA='2add0915ecd4d1fa16d72d77a44e1db5cc4d41b1f1961e1ad06e216fa76119d4'


def compare(a,b):
    base.require(all(a[k]==b[k] for k in ('index','root_slot','condition','task_id','root_state','goal','goal_pose','root_sha256'))
        and all(a['plan'][k]==b['plan'][k] for k in ('task_sha256','blind_sha256','source_report_sha256'))
        and a['status']==b['status']=='horizon_completed' and a['own_returned']==b['own_returned']==10,
        'same full original task/root/goal/time, distinct bound actions')
    p=base.pair(a['final_state'],b['final_state'],b['goal'])
    base.require(p['old']['native_success']==a['native_success'] and p['new']['native_success']==b['native_success']
        and p['old']['native_distance']==a['final_metrics']['native_distance']
        and p['new']['native_distance']==b['final_metrics']['native_distance'], 'independent native arithmetic matches actual labels')
    return dict(index=b['index'],root_slot=b['root_slot'],condition=b['condition'],
        original_index=a['plan']['selected_index'],new_index=b['plan']['selected_index'],
        predicted_Q_delta=b['plan']['nonlinear_terminal_Q']-a['plan']['model_Q'],**p)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    old=base.read(ROOT/'results/agent_allmode_physical_20261028_v1_MAIN01/report.json',OLD_SHA)
    accepted=base.read(ROOT/'results/consume_allmode_physical_MAIN_20261028_v1_MAIN01.json',
        'c26b2dd15b69a1f3532a5e7fd88ad3f39acb7f84071e34101c5711673d8b4acc')
    new=base.read(ROOT/'results/agent_nonlinear_suffix_physical_20261028_v1_MAIN01/report.json',NEW_SHA)
    strict=base.read(ROOT/'results/agent_nonlinear_suffix_physical_20261028_v1_consume_MAIN01.json',RECEIPT_SHA)
    base.require(accepted['receipt']['rows']==old['rows'] and strict['rows']==new['rows']
        and strict['schema']=='agent_nonlinear_suffix_physical_20261028_v1.consume'
        and strict['physical_report_sha256']==NEW_SHA and strict['errors']==[]
        and strict['saved_PNG_verified']==264 and len(old['rows'])==len(new['rows'])==4,
        'full actual original/new fixed4 strict evidence')
    rows=[compare(x,y) for x,y in zip(old['rows'],new['rows'],strict=True)]
    result=dict(schema=NAME,rows=rows,counts={k:dict(better=sum(r['delta_new_minus_old'][k]<0 for r in rows),
        worse=sum(r['delta_new_minus_old'][k]>0 for r in rows),tie=sum(r['delta_new_minus_old'][k]==0 for r in rows))
        for k in rows[0]['delta_new_minus_old']},known_roots=2,fresh_roots=0,root_conditions=4,
        new_neural_calls=0,new_physics_calls=0,full_upstream_rebuilt_here=False,
        P1_admission=False,scientific_admission=False,
        inputs=dict(old_report_sha256=OLD_SHA,new_report_sha256=NEW_SHA,new_receipt_sha256=RECEIPT_SHA),
        caveat='Same fixed allmode tasks; changed suffix and terminal selector, not isolated causal optimizer effect or P1.')
    path=Path(a.output).absolute();base.require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_'),'own new artifact')
    with path.open('xb') as f:f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(path),sha256=base.sha(path),rows=rows,counts=result['counts'])))


if __name__=='__main__':main()
