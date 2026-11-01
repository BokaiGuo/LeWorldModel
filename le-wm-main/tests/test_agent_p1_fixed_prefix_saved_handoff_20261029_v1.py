"""Synthetic saved-model ABI only; never authenticates a real producer."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_saved_handoff_20261029_v1 as m
import test_agent_p1_fixed_prefix_saved_audit_20261029_v1 as fixture

inputs = fixture.inputs
complete = fixture.complete
group = fixture.group
saved = fixture.saved


@pytest.fixture
def panel(saved):
    q, result, _, _ = saved
    requests, rows, audits = [], [], []
    for i in range(64):
        ids = dict(index=i, slot=i//16, source_arm=m.host.method.parent.ARMS[(i%16)//8], source_iteration=i%8+1)
        x = deepcopy(q); x.update(ids); requests.append(x)
        y = deepcopy(result); y.update(ids)
        rows.append(dict(index=i,status='sealed',result=y,resources={}))
        audits.append(dict(index=i,native_forwards=19,scalar_reverse=8,
                           scalar_objective=dict(checked_scalar_losses=8),P1_admission=False))
    prepared = dict(schema=m.host.NAME+'.prepare',contract=m.host.contract(),groups=requests)
    receipt = dict(schema=m.host.NAME+'.consume',status='verified_model_only',complete=True,report_sha256='a'*64,
        group_denominator=64,candidate_denominator=576,rows=rows,audits=audits,
        actual_new_compute={k:m.F.caps(1216,512) for k in ('attempted','completed')},
        P1_admission=False,freshness=False,physical_release=False)
    return prepared,receipt


def test_all_no_witness_remains_64_null(panel):
    p,r = panel
    rows = m.select(p,r,'a'*64)
    assert len(rows) == 64 and all(v['status'] == 'abstain' and v['commands'] is None
                                  and v['raw_float32_bits'] is None and v['selected_iteration'] is None for v in rows)
    assert m.contract()['candidates'] == 576 and not m.contract()['physical_release']


def test_explicit_model_witness_is_exact_action(panel):
    p,r = panel; q=p['groups'][0]; result=r['rows'][0]['result']
    root=q['task']['root_context']['latent_history'][-1]; goal=q['task']['goal']
    q['expected_mid']['latent_history'][-1] = [float(g+2*(x-g)+1) for x,g in zip(root,goal,strict=True)]
    result['stage_Q']=m.F.cost(goal,q['expected_mid'])
    assert result['stage_Q'] > q['initial_Q']
    q['baseline_Q']=result['baseline_Q']=max(v['terminal_Q'] for v in result['iterates'])+1
    for v in result['iterates']:
        v.update(uphill_from_initial=True,better_terminal_than_original=True,witness=True)
    best=min(result['iterates'],key=lambda v:(v['terminal_Q'],v['iteration']))
    result.update(selected_iteration=best['iteration'],status='model_witness')
    out=m.select(p,r,'a'*64)[0]
    assert out['status']=='model_witness' and out['selected_iteration']==best['iteration']
    assert out['raw_float32_bits']==m.F.bits(m.F.chart(q['prefix_logits']).tolist()+best['commands'])
    assert not out['physical_release']


@pytest.mark.parametrize('mutation', ['partial','missing','forged_selection','wrong_bits','outcome','bad_audit','wrong_goal','wrong_index'])
def test_fail_closed_mutations(panel,mutation):
    p,r=panel; q=p['groups'][0]; result=r['rows'][0]['result']
    if mutation=='partial': r['complete']=False
    elif mutation=='missing': r['rows'].pop()
    elif mutation=='forged_selection': result['selected_iteration']=0
    elif mutation=='wrong_bits': result['iterates'][0]['raw_bits']='00'
    elif mutation=='outcome': q['physical_success']=True
    elif mutation=='bad_audit': r['audits'][0]['scalar_objective']['checked_scalar_losses']=0
    elif mutation=='wrong_goal': q['task']['goal'][0]+=1
    else: q['index']=1
    with pytest.raises((ValueError, KeyError)):
        m.select(p,r,'a'*64)
