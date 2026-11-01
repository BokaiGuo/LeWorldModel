"""RAM artificial transitions/PNG bytes only; never native physics or neural."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_p2_contextual_composition_physical_handoff_20261028_v1 as m
from test_p1_moving_overlay_recompile_physical_20261027_v1 import (
    fixture as oldfixture,MemoryStore,Env,fake_place,h)


def fixture():
    p,data,images=oldfixture()
    p.update(schema=m.NAME+'.prepare',contract=m.contract(),strict_receipt_sha256=h('RAM receipt'))
    for i,plan in enumerate(p['plans']):
        plan.update(arm=m.ARMS[i%2],raw_float32_bits=m.bits(plan['commands']),
            raw_sha256=hashlib.sha256(bytes.fromhex(m.bits(plan['commands']))).hexdigest(),
            root_sha256=data['roots'][plan['root_slot']]['root_sha256'],compiler_selection_digest=plan['selection_digest'],
            original_task_file='/RAM_ONLY/task.json.gz',original_task_sha256=h('RAMtask'),
            encoder_identity=dict(model_sha256=h('RAMmodel'),model_config_sha256=h('RAMconfig'),action_mean=[0.,0.],action_std=[1.,1.]))
        plan['selection_digest']=m.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
    return p,data,images


def execute(monkeypatch,*,factory=Env,p=None,data=None,images=None,store=None):
    fp,fd,fi=fixture();p=fp if p is None else p;data=fd if data is None else data;images=fi if images is None else images
    store=MemoryStore() if store is None else store;envs=[]
    monkeypatch.setattr(m.kernel,'place',fake_place)
    monkeypatch.setattr(m.kernel,'validate_env',lambda _:None)
    monkeypatch.setattr(m.kernel,'old_image',lambda ref:images[ref['path']].copy())
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
    def make():
        e=factory();envs.append(e);return e
    binding=dict(prepared='/RAM_ONLY/prepare.json',prepare_sha256=h('RAM prepare'),
        planner_report_sha256=p['planner_report_sha256'],strict_receipt_sha256=p['strict_receipt_sha256'])
    r=m.execute(p,data,store,make,authorization=m.AUTH,binding=binding)
    return r,store,envs,p,data


def assemble(r,store,p,data):
    def image(f):
        b=store.files[f['png']]
        assert hashlib.sha256(b).hexdigest()==f['file_sha256']
        return np.asarray(Image.open(io.BytesIO(b))).copy()
    return m.assemble(r,store.events,p,data,image)


def test_complete_eight_dispatches_full10_rawbits_and_strict_RGB_roll(monkeypatch):
    r,s,envs,p,data=execute(monkeypatch)
    assert r['status']=='contextual_physical_recorded_not_admitted',r['errors']
    assert len(envs)==len(r['rows'])==8 and r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert m.saved.STEM.endswith('_v2')
    c=assemble(r,s,p,data)
    assert c['status']=='engineering_verified_science_blocked' and len(c['rows'])==8 and len(c['PNG_files'])==528
    assert not c['P2_admission'] and not c['physical_release'] and c['new_neural_calls']==0
    for row,plan,e in zip(c['rows'],p['plans'],envs,strict=True):
        assert row['full_horizon'] and e.closed and e.with_target is False
        assert m.bits(np.asarray(e.commands)[10:].tolist())==plan['raw_float32_bits']
        assert e.dispatch_flags==[False]*20
        assert [x['step'] for x in row['rgb_encoding_inputs']]==[0,5,10]
        for cp in row['rgb_encoding_inputs']:
            n=cp['step'];expected=np.concatenate((m.kernel.moving.warmup_commands(),np.asarray(plan['commands'],np.float32)[:n]))[-10:]
            assert cp['raw_action_history_bits']==m.bits(expected.tolist())
            assert not cp['encoding_performed'] and cp['compiler_selection_digest']==plan['compiler_selection_digest']
        assert row['final_metrics']['native_coverage'] is None


@pytest.mark.parametrize('fault',['terminal','truncate'])
def test_native_early_stop_keeps_all8_no_padding(monkeypatch,fault):
    r,s,envs,p,data=execute(monkeypatch,factory=lambda:Env(fault=fault,at=11))
    c=assemble(r,s,p,data)
    assert c['fixed_denominator']==8 and len(c['rows'])==8
    assert r['counts']['public_step']==dict(attempted=88,completed=88)
    for row,e in zip(c['rows'],envs,strict=True):
        assert row['own_returned']==1 and not row['full_horizon'] and row['missing_checkpoints']==[5,10]
        assert len(row['rgb_encoding_inputs'])==1 and len(e.commands)==11


@pytest.mark.parametrize('fault',['substitute','guard','throw'])
def test_incident_denominator_and_bytes_retained_not_success(monkeypatch,fault):
    r,s,envs,p,data=execute(monkeypatch,factory=lambda:Env(fault=fault,at=11,close_error=True))
    assert r['status']=='partial_or_failed' and len(r['rows'])==8 and all(e.closed for e in envs)
    assert all(row['first_error'] and row['secondary_errors'] for row in r['rows'])
    assert not r['P2_admission'] and all(len(e.commands)==11 for e in envs)
    assert all(row['last_observation']['png'] in s.files for row in r['rows'])


@pytest.mark.parametrize('field',['commands','raw_float32_bits','raw_sha256','blind_sha256','root_sha256','selection_digest','condition','arm'])
def test_wrong_plan_rejected_before_factory(field):
    p,data,_=fixture();plan=p['plans'][0]
    if field=='commands':plan[field]=[[.1,0.]]*10  # not lossless float32
    elif field=='condition':plan[field]='on'
    elif field=='arm':plan[field]='OLD_EQUAL144'
    else:plan[field]='0'*64
    with pytest.raises(ValueError):m.validate_plans(p['plans'],data)


def test_auth_rejected_before_IO_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not read prepare'))
    with pytest.raises(ValueError):m.run('/NONE','0'*64,'/NONE',lambda:pytest.fail('constructor'),authorization='OTHER')


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:
        r,s,_,p,data=execute(patch)
    return r,s,p,data


@pytest.mark.parametrize('mutation',[
    'raw','action_dtype','goal','reward','returned_state','omit_return','omit_png','snapshot','goalpose',
    'row_success','row_horizon','missing_checkpoint','history','module_ledger','reserve','branch','terminal','admit','unknown_event'])
def test_saved_mutations_rejected(replay,mutation):
    original,store,p,data=replay;r=deepcopy(original);events=deepcopy(store.events)
    holder=type('SavedRAM',(),dict(files=store.files,events=events))()
    own=[e for e in events if e.get('branch')==0]
    ret=next(e for e in own if e['kind']=='returned_transition' and e['phase']=='plan')
    obs=next(e for e in own if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan')
    if mutation=='raw':ret['actual_bits']='00'*8
    elif mutation=='action_dtype':ret['actual_dtype']='<f8'
    elif mutation=='goal':ret['goal_state'][0]+=1
    elif mutation=='reward':ret['reward']+=1
    elif mutation=='returned_state':ret['state'][0]+=1
    elif mutation=='omit_return':events.remove(ret)
    elif mutation=='omit_png':events.remove(obs)
    elif mutation=='snapshot':obs['post']['snapshot']['state_float64_bits']='WRONG'
    elif mutation=='goalpose':obs['pre']['goal_pose'][0]+=1
    elif mutation=='row_success':r['rows'][0]['native_success']=True
    elif mutation=='row_horizon':r['rows'][0]['own_returned']=9
    elif mutation=='missing_checkpoint':r['rows'][0]['missing_checkpoints']=[5]
    elif mutation=='history':
        data=deepcopy(data);data['roots'][0]['packets']['off']['rgb_history'][1,3,3,0]^=1
    elif mutation=='module_ledger':r['counts']['physics_step']['completed']-=1
    elif mutation=='reserve':
        e=next(e for e in own if e['kind']=='operation_completed' and e['resource']=='public_step');e['raw_bits']='00'*8
    elif mutation=='branch':ret['branch']=99
    elif mutation=='terminal':events.pop()
    elif mutation=='admit':r['P2_admission']=True
    elif mutation=='unknown_event':events.insert(1,dict(kind='UNKNOWN_EVENT',branch=0))
    with pytest.raises((ValueError,AssertionError,KeyError)):assemble(r,holder,p,data)


def test_warmup_old_goal_not_task_goal(replay):
    r,s,p,data=replay
    old=next(e for e in s.events if e['kind']=='returned_transition' and e['phase']=='warmup')
    goal=data['roots'][0]['goal'].tolist()
    assert old['goal_state']!=goal and old['reward']!=m.metrics(old['state'],goal)['native_reward']
    c=assemble(r,s,p,data)
    assert c['rows'][0]['initial_metrics']==m.metrics(r['rows'][0]['root_state'],goal)


def test_native_success_is_agent_PLUS_object_not_object_coverage():
    goal=[100.,100.,150.,150.,0.,0.,0.];state=[200.,200.,150.,150.,0.,0.,0.]
    value=m.metrics(state,goal)
    assert value['object_xy_error']==0 and not value['native_success'] and value['native_coverage'] is None
    assert m.metrics(goal,goal)['native_success']
    assert m.metrics([100.,100.,150.,150.,7.,0.,0.],goal)['native_angle_term']<0  # preserve native formula, no repair


def test_real_constructor_AST_contract_no_instantiation():
    path,expected=m.helper.native_constructor_contract()
    assert m.saved.sha(path)==expected


def test_bad_strict_receipt_rejected_before_loading_source(monkeypatch):
    def reject(*args):raise ValueError('actual receipt rejected')
    monkeypatch.setattr(m.saved,'verify_receipt',reject)
    monkeypatch.setattr(m.kernel,'source_material',lambda:pytest.fail('must not read physical source'))
    with pytest.raises(ValueError,match='actual receipt rejected'):
        m.material('/RAM_ONLY','1'*64,'/RAM_ONLY/receipt','2'*64)


def test_journal_exact_singleLF_and_duplicate_or_tamper_rejected(tmp_path):
    e=dict(index=0,previous_sha256=None,kind='run_finished');e['event_sha256']=m.kernel.source.digest(e)
    path=tmp_path/'RAM_journal.jsonl';path.write_bytes(m.kernel.raw(e))
    assert m.journal(path,e['event_sha256'])==[e]
    for bad in (m.kernel.raw(e)+b'\n',m.kernel.raw(e).replace(b'run_finished',b'run_malformed'),b'{"x":1,"x":2}\n'):
        path.write_bytes(bad)
        with pytest.raises((ValueError,KeyError)):m.journal(path,e['event_sha256'])


def test_consume_receipt_exclusive_small_stdout(tmp_path):
    path=tmp_path/(m.NAME+'_RAM.json');value=dict(status='RAM_ONLY',rows=[{}]*8)
    summary=m.write_receipt(path,value)
    assert summary['rows']==8 and 'rows' not in value.get('summary',{}) and summary['receipt_sha256']==m.saved.sha(path)
    assert path.read_bytes()==m.raw(value)
    with pytest.raises(ValueError):m.write_receipt(path,value)
