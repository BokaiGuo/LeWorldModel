from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_p3_paired_quota_outcomes_20261028_v1 as m


def fixture():
    rows = []
    for i in range(48):
        value = None if i//2 == 23 else dict(own_returned=5, status='returned5', native_success=False,
                     final_metrics={k:float(i) for k in m.METRICS})
        rows.append(dict(ordinal=i,child_id='c'+str(i//2),mode=('fixed','adaptive')[i%2],
            route='not_reached' if value is None else 'reused_saved', physical_result=value,
            missing_reason='not_reached' if value is None else None, fresh_execution=False,
            start_sha256='s'+str(i//2),goal_sha256='g',command=dict(sha256=str(i)),
            decision=dict(horizon=1 if i%2==0 else 3),execution_quota=dict(allowed=value is not None)))
    return dict(complete=True,rows=rows,fixed_denominator=48,child_denominator=24,census={},
        runtime_actual_resources=dict(fixed=dict(completed=dict(predict_API=634,metric_calls=216)),
                                      adaptive=dict(completed=dict(predict_API=634,metric_calls=72))))


def test_fixed48_and_missing_not_zero():
    result=m.assemble(fixture())
    assert len(result['rows'])==48 and len(result['pairs'])==24
    assert result['physical_pair_count']==23 and result['missing_pair_count']==1
    assert result['pairs'][-1]['native_success']==[None,None]
    assert result['pairs'][-1]['adaptive_minus_fixed_final_metric']['joint_xy_error'] is None


def test_equal_predict_not_equal_resources_or_C3():
    result=m.assemble(fixture())
    assert result['predict_calls_equal'] and not result['resource_vectors_equal']
    assert not result['equal_FLOPs'] and not result['C3_advantage_established']
    assert result['independent_pair_samples'] is result['confidence_interval'] is None


def test_early_success_not_padded_to_same_time():
    value=fixture();value['rows'][0]['physical_result'].update(own_returned=2,native_success=True)
    result=m.assemble(value)
    assert result['native_success_rows']['fixed']==1
    assert result['physical_pair_count']==23 and result['equal_actual5_pair_count']==22
    assert result['pairs'][0]['actual_steps']==[2,5]


@pytest.mark.parametrize('mutation', ['drop','order','start','goal','child'])
def test_wrong_pair_identity_rejected(mutation):
    value=deepcopy(fixture())
    if mutation=='drop':value['rows'].pop()
    elif mutation=='order':value['rows'][0],value['rows'][1]=value['rows'][1],value['rows'][0]
    elif mutation=='start':value['rows'][1]['start_sha256']='other'
    elif mutation=='goal':value['rows'][1]['goal_sha256']='other'
    else:value['rows'][1]['child_id']='other'
    with pytest.raises(ValueError):m.assemble(value)
