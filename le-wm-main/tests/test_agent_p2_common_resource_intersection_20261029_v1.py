"""Only RAM callbacks and temporary journal files; no numerical model executor."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p2_common_resource_intersection_20261029_v1 as m


class Clock:
    def __init__(self):self.now=0.0
    def __call__(self):return self.now


def ledger(root=0,arm='joint',sink=None):
    clock=Clock();events=[]
    return m.Ledger(root,arm,clock=clock,sink=events.append if sink is None else sink),clock,events


def predict(l,*,capture_fail=False,return_wall=False,clock=None):
    def body():
        for name in ('AE','predictor','lowproj'):
            def callback():
                if return_wall:clock.now=451.0
                return b'RAM_no_tensor'
            def capture(value):
                if capture_fail:raise OSError('fixture capture fails after module return')
                return dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
            l.module(name,callback,capture,shapes=deepcopy(m.MODULE_SHAPES[name]))
    return l.call('predict',body)


def execute_unit(l,label):
    assert l.admit(label)
    wanted=m.envelope(l.arm,label)
    h=(2 if label.startswith('H2:') else 1 if label.startswith('H1:') else 3)
    if label=='replay':
        for part in ('whole','segmented'):
            l.call('replay',lambda:[predict(l) for _ in range(3)],part=part)
    elif l.arm=='grasp':
        for _ in range(3):predict(l)
        l.call('score',lambda:None,trajectories=1,scope='lifted_loss' if label.startswith('joint:') else 'serial_terminal')
        l.call('backward',lambda:None)
        l.auxiliary('Adam_step',lambda:None)
        if label.startswith('joint:'):l.auxiliary('virtual_noise',lambda:None)
    else:
        if wanted['J']:
            l.call('J',lambda:[predict(l) for _ in range(h)],horizon=h,dimension=h*10)
        for _ in range(wanted['predict_API']-(h if wanted['J'] else 0)):predict(l)
        if label=='bank':
            l.auxiliary('root_SVD',lambda:None);l.auxiliary('root_SVD',lambda:None);l.auxiliary('root_ridge',lambda:None)
        else:l.call('score',lambda:None,trajectories=1,scope='root_full_path')
    assert l.commit()


def test_exact_original27_source_no_outcomes():
    pins=m.source_check()
    source=m.ROOT/'experiments/spectral_reachability/agent_p2_six_target_controller_cuda_20261029_v1.py'
    tree=ast.parse(source.read_text())
    axes=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='AXES' for t in n.targets))
    assert m.AXES==axes and len(axes)==27
    assert len(pins)==3 and m.protocol()['production_runner_implemented'] is False


@pytest.mark.parametrize('arm,api,ad,j,bw',[
    ('joint',1902,255,85,0),('composition',1062,129,85,0),('grasp',1902,1896,0,632)])
def test_all_source_schedules_fit_identical_vector_box(arm,api,ad,j,bw):
    total=m.add(*(m.envelope(arm,label) for label in m.schedule(arm)))
    assert (total['predict_API'],total['predictor_samples'],total['AD_primal'],total['J'],total['backward'])==(api,api,ad,j,bw)
    assert total['mean_paid_replays']==0 and total['replay_trajectories']==2
    assert all(total[k]<=m.caps()[k] for k in m.AXES[:-1])
    if arm=='grasp':
        assert sum(x.startswith('joint:') for x in m.schedule(arm))==532
        assert sum(x.startswith('sync:') for x in m.schedule(arm))==100
    assert not m.protocol()['actual_cost_equality']


@pytest.mark.parametrize('root,arm',m.ORDER)
def test_all_six_fixed_slots_real_tempdisk_chain(tmp_path,root,arm):
    path=tmp_path/f'{root}_{arm}.jsonl';journal=m.Journal(str(path),cap_bytes=256*1024)
    l,clock,_=ledger(root,arm,sink=journal)
    execute_unit(l,m.schedule(arm)[0])
    v=m.verify_journal(str(path),cap_bytes=256*1024)
    assert v['used']==l.summary()['used'] and v['events']==len(l.events)
    assert l.summary()['committed']==[m.schedule(arm)[0]]
    assert l.used['FLOPs'] is None and not l.summary()['scientific_admission']
    with pytest.raises(ValueError,match='new bounded'):m.Journal(str(path),cap_bytes=256*1024)


@pytest.mark.parametrize('axis',[k for k in m.AXES if k not in ('FLOPs','J','VJP','AD_primal','backward','mean_paid_replays')])
def test_every_common_axis_is_a_hard_nonexchangeable_gate(axis):
    l,_,_=ledger(arm='grasp')
    # This unit consumes all forward axes plus backward/AD. No padding calls.
    label=l.labels[0];cost=m.envelope(l.arm,label);reserve=m.envelope(l.arm,'replay')
    if cost[axis]+reserve[axis]==0:return
    l.used[axis]=m.caps()[axis]-cost[axis]-reserve[axis]+1
    before=deepcopy(l.used)
    assert not l.admit(label) and l.used==before and l.stop=='resource_cap'
    assert axis in l.events[-1]['over']


@pytest.mark.parametrize('arm,axis',[('joint','J'),('joint','VJP'),('grasp','AD_primal'),('grasp','backward')])
def test_derivative_caps_are_not_a_fungible_predict_currency(arm,axis):
    l,_,_=ledger(arm=arm)
    if arm=='joint':execute_unit(l,'nominal');label='bank'
    else:label=l.labels[0]
    l.used[axis]=m.caps()[axis]
    assert not l.admit(label) and axis in l.events[-1]['over']
    assert l.used['predict_API']<1902


def test_full_replay_is_reserved_and_paid_not_a_free_tail():
    l,_,_=ledger();l.used['predict_API']=1896
    assert not l.admit('nominal') and l.stop=='resource_cap'
    before=l.used['predict_API'];assert l.admit('replay')
    for part in ('whole','segmented'):l.call('replay',lambda:[predict(l) for _ in range(3)],part=part)
    assert l.used['predict_API']==before+6==1902 and l.used['replay_trajectories']==2 and l.commit()
    assert l.used['J']==0 and l.used['VJP']==0


def test_callback_failure_keeps_actual_return_separate_from_capture():
    l,_,_=ledger();assert l.admit('nominal')
    with pytest.raises(OSError):predict(l,capture_fail=True)
    assert l.used['AE_attempts']==l.used['AE_dispatched']==l.used['AE_returned']==1
    assert l.used['AE_captured']==l.used['predictor_samples']==0
    assert l.used['predict_API']==1 and l.stop=='operation_failure' and not l.commit()


def test_source_bank_cpu_and_j_work_paid_separately():
    l,_,_=ledger();execute_unit(l,'nominal');execute_unit(l,'bank')
    assert l.used['J']==1 and l.used['VJP']==192 and l.used['AD_primal']==3
    assert l.side['root_SVD']==2 and l.side['root_ridge']==1 and l.side['Jacobian_elements_requested']==5760
    assert l.used['backward']==0


def test_no_empty_or_partial_unit_commit_and_no_schedule_skip():
    l,_,_=ledger();assert l.admit('nominal');assert not l.commit() and l.stop=='incomplete_unit'
    other,_,_=ledger()
    with pytest.raises(ValueError,match='skipped'):other.admit('bank')


@pytest.mark.parametrize('elapsed',[450.0,451.0])
def test_exact_wall_blocks_new_work_and_reserved_replay(elapsed):
    l,c,_=ledger();c.now=elapsed
    assert not l.admit('nominal') and not l.admit('replay')
    assert l.used==m.zeros() and l.stop=='wall_cap'


def test_inflight_return_after_deadline_is_paid_no_new_capture_or_commit():
    l,c,_=ledger();assert l.admit('nominal')
    with pytest.raises(m.Refused):predict(l,return_wall=True,clock=c)
    assert l.used['predict_API']==l.used['AE_dispatched']==l.used['AE_returned']==1
    assert l.used['AE_captured']==0 and l.stop=='wall_cap' and l.summary()['wall_overrun']
    assert not l.commit()


def test_guard_durable_writer_failure_before_callback_no_dispatch():
    events=[]
    def sink(e):
        if e['kind']=='debit':raise OSError('fixture fsync refusal')
        events.append(e)
    l,_,_=ledger(sink=sink);assert l.admit('nominal');called=[]
    with pytest.raises(OSError):l.call('predict',lambda:called.append(True))
    assert not called and l.failed_transport and l.used==m.zeros()
    assert l.summary()['used'] is None # Not a durable zero-cost certificate.
    with pytest.raises(ValueError,match='cannot resume'):l.admit('nominal')


def test_batch2_rejected_before_module_attempt():
    l,_,_=ledger();assert l.admit('nominal');called=[]
    def body():l.module('AE',lambda:called.append(1),lambda x:{},shapes=[[2,3,10]])
    with pytest.raises(ValueError,match='batch1'):l.call('predict',body)
    assert not called and l.used['predict_API']==1 and l.used['AE_attempts']==0


def test_transport_hashchain_tamper_and_cap(tmp_path):
    p=tmp_path/'ledger.jsonl';journal=m.Journal(str(p),cap_bytes=256*1024)
    l,_,_=ledger(sink=journal);execute_unit(l,'nominal')
    b=p.read_bytes();p.write_bytes(b.replace(b'predict_API',b'predict_BAD',1))
    with pytest.raises(ValueError):m.verify_journal(str(p),cap_bytes=256*1024)
    small=m.Journal(str(tmp_path/'small'),cap_bytes=1)
    with pytest.raises(ValueError,match='byte cap'):ledger(sink=small)


def test_no_model_import_or_outcome_reader():
    tree=ast.parse(Path(m.__file__).read_text());imports=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
        if isinstance(n,ast.ImportFrom):imports.append(n.module)
    assert not any(x and x.split('.')[0] in ('torch','numpy','gymnasium','pickle','experiments') for x in imports)
    assert not any(isinstance(n,ast.Global) for n in ast.walk(tree))
    assert 'results/' not in Path(m.__file__).read_text()
