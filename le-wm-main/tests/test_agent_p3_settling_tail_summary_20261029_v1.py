import json
import math
import pytest
from experiments.spectral_reachability import agent_p3_settling_tail_summary_20261029_v1 as m


def fixture():
    jobs=[]
    for i in range(10):
        root,role=m.ROLES[i//2]; goal=[1.,2.,3.,4.,.1,5.,-7.]
        ops=[]; cps=[]
        for step in range(1,21):
            s=[float(step+i),2.,3.,4.,.2,0.,0.]; snapshot={'state':s}
            v={'state':s,'snapshot':snapshot,'goal_state':goal,'terminated':False,'truncated':False,
               'reward':-m.metrics(s,goal)['native7_distance']}
            ops.append(dict(resource='public_step',phase='plan',status='returned',step=step,result=v))
            if step in (15,20): cps.append(dict(step=step,state=s,snapshot=snapshot,file_sha256='a',pixel_sha256='b'))
        job=dict(index=i,endpoint=i//2,root_slot=root,source_role=role,probe=('U','Z2')[i%2])
        row=dict(job=job,status='horizon_completed',error=None,operations=ops,checkpoints=cps,
            endpoint15=cps[0],endpoint20=cps[1],fixture_only=False,new_neural_calls=0,**{k:False for k in m.FLAGS})
        jobs.append(dict(row=row))
    return dict(schema=m.RECEIPT_SCHEMA,fixed_denominator=10,new_neural_calls=0,jobs=jobs,**{k:False for k in m.FLAGS})


def test_full_fixed_denominator_and_direction():
    out=m.analyze(fixture())
    assert len(out['branches'])==10 and len(out['pairs'])==10
    assert out['pairs'][0]['axis_gains']['native7_squared']>0
    assert len(out['pairs'][0]['squared_residual7_gains'])==7
    assert out['branches'][0]['cp20_minus_cp15']['native7_distance']>0
    assert out['branches'][0]['checkpoints']['15']['metrics']['goal7'][-2:]==[5.,-7.]
    assert out['branches'][0]['checkpoints']['15']['native_success'] is False


def test_missing_and_early_stop_are_not_zero():
    r=fixture(); row=r['jobs'][0]['row']; row['status']='native_stopped'; row['operations']=row['operations'][:15]
    row['checkpoints']=row['checkpoints'][:1]; row['endpoint20']=None
    out=m.analyze(r)
    assert out['branches'][0]['checkpoints']['20'] is None
    assert out['pairs'][1]['axis_gains'] is None and len(out['pairs'])==10


@pytest.mark.parametrize('mutation', ['identity','goal','state','reward','flag','denominator'])
def test_bad_inputs_fail(mutation):
    r=fixture(); row=r['jobs'][0]['row']
    if mutation=='identity': row['job']['probe']='Z2'
    if mutation=='goal': row['operations'][1]['result']['goal_state']=[0.]*7
    if mutation=='state': row['endpoint15']['state']=[0.]*7
    if mutation=='reward': row['operations'][14]['result']['reward']=0.
    if mutation=='flag': r['P3_admission']=True
    if mutation=='denominator': r['fixed_denominator']=8
    with pytest.raises(ValueError): m.analyze(r)


def test_metrics_velocity_and_wrapped():
    x=m.metrics([0,0,0,0,7,0,0],[0,0,0,0,0,3,4])
    assert x['agent_velocity_error']==5 and x['native_angle_term']<0
    assert math.isclose(x['wrapped_angle_error'],7-2*math.pi)
    assert x['native7_squared']==74


def test_saved_build_verify_and_tamper(tmp_path,monkeypatch):
    r=tmp_path/'receipt.json'; r.write_bytes(m.encode(fixture()))
    monkeypatch.setattr(m,'RECEIPT_SHA',m.sha(r))
    output=tmp_path/'output.json'
    monkeypatch.setattr('sys.argv',['summary','build','--receipt',str(r),'--output',str(output)])
    m.main()
    monkeypatch.setattr('sys.argv',['summary','verify','--receipt',str(r),'--output',str(output)])
    m.main()
    obj=json.loads(output.read_bytes()); obj['fixed_branches']=9; output.write_bytes(m.encode(obj))
    with pytest.raises(ValueError): m.main()
    r.write_bytes(b'{}')
    with pytest.raises(ValueError): m.build(str(r))
