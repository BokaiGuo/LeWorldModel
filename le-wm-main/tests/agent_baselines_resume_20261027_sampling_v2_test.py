"""RAM metadata only; no tmp files, models, simulator or producer execution."""
from copy import deepcopy
import json
import math

import pytest

from experiments.spectral_reachability import agent_baselines_resume_20261027_sampling_v2 as c


def meter():
    where=dict(root_slot=0,arm=c.ARMS[0],branch=0,selection_digest='a'*64)
    outer=dict(operation_index=0,resource='reset',status='pending',**where)
    inner=dict(operation_index=1,resource='physics_step',status='pending',**where)
    close=dict(operation_index=2,resource='environment_close',status='pending',**where)
    events=[dict(kind='operation_dispatch_intent',receipt=outer),
            dict(kind='operation_dispatch_intent',receipt=inner),
            dict(kind='operation_returned',receipt=dict(inner,status='returned')),
            dict(kind='operation_returned',receipt=dict(outer,status='returned')),
            dict(kind='operation_dispatch_intent',receipt=close),
            dict(kind='operation_returned',receipt=dict(close,status='returned'))]
    counts={k:dict(attempted=0,completed=0) for k in c.CAPS}
    for k in ('reset','physics_step','environment_close'):counts[k]=dict(attempted=1,completed=1)
    report=dict(status='partial_or_failed',actual_operation_receipts=[dict(r,status='returned') for r in (outer,inner,close)],
                counts=counts,total_internal_physics_scope='RAM_nested_mock')
    return report,dict(contract=dict(caps=deepcopy(c.CAPS))),events


def count(args):return c.operation_counts(*args,c.ARMS,False)


def test_close_is_real_receipt_not_synthetic_cleanup():
    result=count(meter())
    assert result['actual']['environment_close']==dict(attempted=1,completed=1)
    assert result['operation_receipt_ABI']=='nested_operation_index_including_close'


@pytest.mark.parametrize('kind',['cleanup_intent','cleanup_returned','operation_attempted','operation_completed'])
def test_obsolete_abi_rejected(kind):
    r,p,e=meter();e.append(dict(kind=kind))
    with pytest.raises(ValueError,match='obsolete'):count((r,p,e))


@pytest.mark.parametrize('mutation',range(10))
def test_meter_corruption_rejected(mutation):
    r,p,e=meter()
    if mutation==0:e[0]['receipt']['operation_index']=True
    elif mutation==1:e[1]['receipt']['operation_index']=0
    elif mutation==2:e[2],e[3]=e[3],e[2]
    elif mutation==3:e[2]['receipt']['resource']='reset'
    elif mutation==4:e[2]['receipt']['selection_digest']='b'*64
    elif mutation==5:e[2]['receipt']['status']='failed'
    elif mutation==6:r['actual_operation_receipts'].pop()
    elif mutation==7:r['counts']['environment_close']['completed']=0
    elif mutation==8:e.pop()
    else:e[0]['receipt']['branch']=1
    with pytest.raises(ValueError):count((r,p,e))


def test_failed_close_kept_and_charged_not_success():
    r,p,e=meter();failed=dict(e[-1]['receipt'],status='failed',error_type='RAMFailure',message='close failed')
    e[-1]=dict(kind='operation_failed',receipt=failed);r['actual_operation_receipts'][-1]=failed
    r['counts']['environment_close']['completed']=0
    assert count((r,p,e))['actual']['environment_close']==dict(attempted=1,completed=0)
    r['status']='matched_sampling_physical_recorded_not_admitted'
    with pytest.raises(ValueError,match='complete'):count((r,p,e))


def test_actual_receipt_cannot_be_replaced_by_intent():
    r,p,e=meter();r['actual_operation_receipts'][0]['status']='pending'
    with pytest.raises(ValueError,match='ALL'):count((r,p,e))


def rgb_fixture():
    rows=[];rgb=[]
    for s,a in c.keys(c.ARMS):
        d=c.ident([s,a]);rows.append(dict(root_slot=s,arm=a,physical_selection_digest=d,task_id=str(s)))
        rgb.append(dict(root_slot=s,arm=a,selection_digest=d,task_id=str(s),checkpoints=[
            dict(step=t,status='encoded',Q=float(t+s+1)) for t in (0,5,10)]))
    result=dict(rows=rows,paired_sampling=c.paired([dict(r,terminal10=None) for r in rows],c.ARMS))
    receipt=dict(schema=c.RGB_NAME+'.consume',P1_admission=False,new_independent_samples=0,rows=rgb,
                 counts={k:dict(attempted=14,completed=14) for k in ('encoder','projector')})
    return result,receipt


def test_rgb_stable_identity_not_array_order():
    r,p=rgb_fixture();p['rows'].reverse();c.attach_rgb_rows(r,p)
    assert all(x['actual_latent_Q']['10']==11+x['root_slot'] for x in r['rows'])
    assert r['RGB_evidence']['source_goal_images']==2


@pytest.mark.parametrize('mutation',range(5))
def test_rgb_wrong_selection_or_missing_not_filled(mutation):
    r,p=rgb_fixture()
    if mutation==0:p['rows'][0]['selection_digest']='b'*64
    elif mutation==1:p['rows'][0]['task_id']='wrong'
    elif mutation==2:p['rows'][0]['checkpoints'].pop()
    elif mutation==3:p['rows'][0]['checkpoints'][0]['Q']=float('nan')
    else:p['counts']['encoder']['completed']=12
    with pytest.raises(ValueError):c.attach_rgb_rows(r,p)


@pytest.fixture(scope='module')
def matrix():
    # Existing small saved metadata only: never imports an optimizer or model.
    paper=json.loads((c.ROOT/'results/p2_reported_baseline_table_20261027_v3/export/report.json').read_bytes())
    local=json.loads((c.ROOT/'results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json').read_bytes())
    return c.matrix_from_reported(paper,local['rows'],dict(rows=[]))


def test_all_author_cells_kept_and_separated(matrix):
    assert len(matrix['author_reported_cells'])==238 and len(matrix['prior_local_actual'])==12
    assert sum(x['author_reported']['value'] is None for x in matrix['author_reported_cells'])==2
    assert all(x['local_actual'] is None and x['improvement_over_local'] is None for x in matrix['author_reported_cells'])
    assert all(x['author_reported'] is None for x in matrix['prior_local_actual'])
    assert matrix['eligible_cross_paper_improvements']==0


@pytest.mark.parametrize('paper',['GRASP','Hi-LeWM','LeWM','PlanWAM'])
def test_page_table_version_protocol_and_no_official_reproduction(matrix,paper):
    rows=[r for r in matrix['author_reported_cells'] if r['paper']==paper]
    assert rows and all(r['version'] and r['provenance']['table_id'] and r['provenance']['protocol_quotes'] for r in rows)
    assert all(r['provenance']['pdf_page_1based']=='unknown' for r in rows) if paper=='PlanWAM' else all(
        type(r['provenance']['pdf_page_1based']) is int for r in rows)
    assert matrix['local_official_paper_reproductions'][paper] is None


def test_no_model_success_to_physical_success_promotion(matrix):
    rows=[r for r in matrix['author_reported_cells'] if r['paper']=='GRASP']
    assert all(r['protocol']['supplement']['metric_domain']=='learned_world_model_only' for r in rows)
    assert matrix['official_MPPI'] is matrix['equal_total_FLOPs'] is matrix['P1_admission'] is False


@pytest.mark.parametrize('angle',[-8*math.pi,-2*math.pi,0,2*math.pi,8*math.pi])
def test_wrapped_angle_invariant_native_bug_preserved(angle):
    goal=[0.,0.,0.,0.,angle,0.,0.];state=[0.]*7
    m=c.metrics(goal,state)
    assert abs(m['wrapped_angle_error'])<1e-12


def test_absolute_external_hash_must_match_before_parse():
    with pytest.raises(ValueError,match='external comparison binding'):
        c.material('/RAM/binding','0'*64,read=lambda _:b'{}')


def test_missing_rgb_remains_null():
    result={};c.attach_rgb(result,None,lambda _:None)
    assert result['RGB_evidence']['actual_latent_Q'] is None


def test_fixed_four_identity_denominator():
    with pytest.raises(ValueError):c.index([dict(root_slot=0,arm=c.ARMS[0])]*4,c.ARMS)
    with pytest.raises(ValueError):c.index([],c.ARMS)
