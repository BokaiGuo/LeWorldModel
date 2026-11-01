import json
import subprocess
import sys
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_e1_forced_candidate_paired_summary_20261029_v1 as m


def sample(early=False):
    plans=[];rows=[];flat=[]
    for i,(root,arm) in enumerate(m.SLOTS):
        p=dict(root_slot=root,arm=arm,root_sha256=str(root)*64,task_id=str(root)*64,
               task_sha256='a'*64,task_contract_sha256='b'*64,blind_sha256='c'*64,
               target=[0.]*192,commands=[[float(i)/8,0.]]*10,selection_digest=str(i)*64)
        plans.append(p);n=4 if early and i==1 else 10
        cps=[dict(step=s,state=[float(i+s),0.,0.,0.,0.,0.,0.],png=f'{i}-{s}.png',file_sha256='d'*64,pixel_sha256='e'*64)
             for s in m.STEPS if s<=n]
        row=dict(branch_index=i,root_slot=root,arm=arm,selected_plan=p,selection_digest=p['selection_digest'],
                 checkpoints=cps,own_returned=n,status='native_stopped' if n<10 else 'horizon_completed',
                 initial_exact=True,goal=[0.]*7,terminated=False,truncated=n<10)
        rows.append(row);flat.extend(dict(branch=i,root_slot=root,arm=arm,checkpoint=x) for x in cps)
    report=dict(schema=m.PHYSICAL+'.report',status='six_slot_native_recorded',fixed_branch_denominator=6,
                scientific_admission=False,P1_admission=False,rows=rows,counts={'public_step':{'attempted':0,'completed':0}},errors=[])
    receipt=dict(schema=m.PHYSICAL+'.consume',status=report['status'],fixed_branch_denominator=6,
                 scientific_admission=False,P1_admission=False,rows=rows,counts=report['counts'],checkpoints=flat)
    return dict(plans=plans),receipt,report


def test_fixed_six_pairs_and_native_metrics():
    p,r,b=sample();out=m.summarize(p,r,b)
    assert len(out['cells'])==len(out['pairs'])==18
    assert out['pairs'][0]['arms']==['bank_winner','zero_extra']
    assert out['pairs'][0]['delta_b_minus_a']['joint_xy']==1.
    assert not out['pairs'][0]['same_raw10']
    assert out['same_target'] and not out['scientific_admission']


def test_early_stop_retains_missing_not_endpoint_padding():
    p,r,b=sample(True);out=m.summarize(p,r,b)
    assert sum(x['status']=='missing' for x in out['cells'])==2
    assert sum(x['status']=='missing' for x in out['pairs'])==4
    assert next(x for x in out['cells'] if x['branch']==1 and x['step']==5)['value'] is None


@pytest.mark.parametrize('kind',['target','goal','order','checkpoint','receipt'])
def test_join_tampering_rejected(kind):
    p,r,b=sample();p=deepcopy(p);r=deepcopy(r);b=deepcopy(b)
    if kind=='target':
        p['plans'][1]['target'][0]=1;b['rows'][1]['selected_plan']=p['plans'][1];r['rows']=deepcopy(b['rows'])
    if kind=='goal':
        b['rows'][1]['goal'][0]=1;r['rows']=deepcopy(b['rows'])
    if kind=='order':p['plans'].reverse()
    if kind=='checkpoint':b['rows'][0]['checkpoints'].pop();r['rows']=deepcopy(b['rows'])
    if kind=='receipt':r['counts']={}
    with pytest.raises(ValueError):m.summarize(p,r,b)


def test_tmp_cli_full_rebuild_and_tamper(tmp_path):
    p,r,b=sample();source_sha=m.sha(m.SOURCE.read_bytes());p['bindings']={str(m.SOURCE):source_sha}
    prep=tmp_path/'prepare.json';prep.write_bytes(m.canonical(p))
    b['prepared']=str(prep);b['prepare_sha256']=m.sha(prep.read_bytes())
    report=tmp_path/'report.json';report.write_bytes(m.canonical(b));r['report_sha256']=m.sha(report.read_bytes())
    receipt=tmp_path/'receipt.json';receipt.write_bytes(m.canonical(r));output=tmp_path/'analysis.json'
    mod='experiments.spectral_reachability.'+m.NAME
    def cli(*args):return subprocess.run([sys.executable,'-B','-m',mod,*args],capture_output=True,text=True)
    result=cli('analyze','--receipt',str(receipt),'--receipt-sha',m.sha(receipt.read_bytes()),
               '--report',str(report),'--report-sha',m.sha(report.read_bytes()),'--physical-source-sha',source_sha,'--output',str(output))
    assert result.returncode==0,result.stderr
    digest=json.loads(result.stdout)['sha256']
    assert cli('verify','--analysis',str(output),'--analysis-sha',digest).returncode==0
    assert cli('verify','--analysis',str(output),'--analysis-sha','0'*64).returncode!=0
