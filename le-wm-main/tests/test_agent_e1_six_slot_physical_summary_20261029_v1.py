import copy
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_e1_six_slot_physical_summary_20261029_v1 as m


def fixture():
    rows=[]; checkpoints=[]
    for i,(root,arm) in enumerate(m.SLOTS):
        returned=5 if i==0 else 10
        cps=[dict(step=step,state=[float(root),1.,2.,3.,0.25,4.,5.],png=f'{i}_{step}.png',file_sha256='a'*64,pixel_sha256='b'*64)
             for step in m.STEPS if step<=returned]
        row=dict(branch_index=i,root_slot=root,arm=arm,selection_digest=f'{i:064x}',
                 selected_plan=dict(root_slot=root,arm=arm,selection_digest=f'{i:064x}',target=[i]*192),
                 goal=[float(root),0.,0.,0.,0.,0.,0.],own_returned=returned,
                 status='native_success' if i==0 else 'horizon_completed',checkpoints=cps)
        rows.append(row)
        checkpoints.extend(dict(branch=i,root_slot=root,arm=arm,checkpoint=cp) for cp in cps)
    report=dict(schema=m.PHYSICAL+'.report',status='six_slot_native_recorded',fixed_branch_denominator=6,
                scientific_admission=False,P1_admission=False,rows=rows,counts={'example':{'attempted':6,'completed':6}})
    receipt=dict(schema=m.PHYSICAL+'.consume',status=report['status'],fixed_branch_denominator=6,
                 scientific_admission=False,P1_admission=False,rows=rows,counts=report['counts'],checkpoints=checkpoints)
    return receipt,report


def test_fixed_denominator_and_no_intermediate_target_comparison():
    receipt,report=fixture(); result=m.summarize(receipt,report)
    assert (result['slot_denominator'],result['checkpoint_denominator'],result['pair_denominator'])==(6,18,18)
    assert result['rows'][0]['cells'][2]['value'] is None
    assert sum(p['status']=='unknown' for p in result['pairs'])==2
    cell=result['rows'][1]['cells'][0]['value']
    assert cell['axis_squared']==[0.,1.,4.,9.,.0625,16.,25.]
    assert cell['velocity_squared']==41. and cell['native_success'] is True
    assert result['distinct_intermediate_target_quality_comparison'] is False


def test_goal_mismatch_is_unknown_not_cross_goal_comparison():
    r,p=fixture();r=copy.deepcopy(r);p=copy.deepcopy(p)
    r['rows'][1]['goal'][0]=50.;p['rows'][1]['goal'][0]=50.
    result=m.summarize(r,p)
    assert next(x for x in result['pairs'] if x['root']==0 and x['arms']==list(m.ARMS[:2]) and x['step']==0)['reason']=='different native goals'


@pytest.mark.parametrize('mutation', ['wrong_order','wrong_projection','bad_step','bad_state','wrong_report','partial_row','boolean_count'])
def test_malformed_inputs_rejected(mutation):
    r,p=copy.deepcopy(fixture())
    if mutation=='wrong_order':r['rows'].reverse()
    elif mutation=='wrong_projection':r['checkpoints'].pop()
    elif mutation=='bad_step':r['rows'][0]['checkpoints'][0]['step']=7;p['rows'][0]['checkpoints'][0]['step']=7
    elif mutation=='bad_state':r['rows'][0]['checkpoints'][0]['state'][0]=float('nan');p['rows'][0]['checkpoints'][0]['state'][0]=float('nan')
    elif mutation=='wrong_report':p['counts']={}
    elif mutation=='partial_row':r['rows'][1]['status']='failed'
    else:r['rows'][1]['own_returned']=True
    with pytest.raises((ValueError,KeyError)):m.summarize(r,p)


def test_disk_cli_analyze_verify_and_tamper(tmp_path):
    receipt,report=fixture();report_path=tmp_path/'report.json';receipt_path=tmp_path/'receipt.json';out=tmp_path/'out.json'
    report_path.write_bytes(m.canonical(report));report_sha=m.digest(report_path.read_bytes())
    receipt['report_sha256']=report_sha;receipt_path.write_bytes(m.canonical(receipt));receipt_sha=m.digest(receipt_path.read_bytes())
    command=[sys.executable,'-m','experiments.spectral_reachability.'+m.NAME]
    run=subprocess.run(command+['analyze','--receipt',str(receipt_path),'--receipt-sha',receipt_sha,
        '--report',str(report_path),'--report-sha',report_sha,'--output',str(out)],capture_output=True,text=True,check=True)
    output_sha=json.loads(run.stdout)['sha256']
    subprocess.run(command+['verify','--analysis',str(out),'--analysis-sha',output_sha],capture_output=True,text=True,check=True)
    report_path.write_bytes(report_path.read_bytes()+b' ')
    bad=subprocess.run(command+['verify','--analysis',str(out),'--analysis-sha',output_sha],capture_output=True,text=True)
    assert bad.returncode!=0
    again=subprocess.run(command+['analyze','--receipt',str(receipt_path),'--receipt-sha',receipt_sha,
        '--report',str(report_path),'--report-sha',report_sha,'--output',str(out)],capture_output=True,text=True)
    assert again.returncode!=0
