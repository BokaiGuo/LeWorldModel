"""Saved literal arithmetic and RAM native-interface tests; no real physics/model."""
from copy import deepcopy
import os
from pathlib import Path
import time
import pytest
from experiments.spectral_reachability import agent_p2_safeguard_factorial_physical_20261028_v1 as m
import test_agent_p2_contextual_composition_physical_handoff_20261028_v1 as native


@pytest.fixture(scope='module')
def literal():
    cells=m.physical.read(m.ROOT/'results/agent_p2_safeguard_factorial_cuda_20261028_v1_consume_MAIN01.json')['cells']
    old=m.physical.read(m.ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v2_prepare_MAIN01.json')
    return cells,old['plans'],old['branches']


def test_actual_saved_schema_all32_to_eight_and_guard4(literal):
    cells,parents,branches=literal
    plans,reuse=m.zero_plans(cells,parents,branches,'a'*64)
    assert len(plans)==len(reuse)==8
    assert [r['source']=='original_incumbent_control' for r in reuse]==[False]*4+[True]+[False]*3
    for i,p in enumerate(plans):
        assert p['commands']==cells[4*i]['full10']
        assert p['planner_report_sha256']=='a'*64
        assert p['parent_planner_report_sha256']==parents[i]['planner_report_sha256']
        assert 'predicted_Q' not in p and 'witness' not in p
        assert p['new_suffix_Q']==cells[4*i]['selection']['model_Q']
        assert p['selection_digest']==m.physical.saved.digest({k:v for k,v in p.items() if k!='selection_digest'})


@pytest.mark.parametrize('field',['input_context_sha256','goal_sha256','prefix_raw_bits','full10_raw_bits','selection_digest','branch','guard'])
def test_cell_mutations_failclosed(literal,field):
    cells,parents,branches=literal;cells=deepcopy(cells)
    cells[0][field]=False if field=='guard' else 'BAD'
    if field=='guard':cells[0][field]=True
    with pytest.raises(ValueError):m.zero_plans(cells,parents,branches,'a'*64)


def test_missing_candidate_denominator_refused(literal):
    c,p,b=literal
    with pytest.raises(ValueError):m.zero_plans(c[:-1],p,b,'a'*64)


def test_metadata_file_fsync_open_then_closed_directory(monkeypatch,tmp_path):
    path=tmp_path/(m.NAME+'_prepare_RAM.json');calls=[];real=m.os.fsync
    def sync(fd):
        assert os.fstat(fd).st_size==len(m.raw({'x':1}))
        assert path.read_bytes()==m.raw({'x':1})
        calls.append(('file',fd));real(fd)
    def directory(p):
        assert p==tmp_path and calls[0][0]=='file'
        with pytest.raises(OSError):os.fstat(calls[0][1])
        calls.append(('dir',p))
    monkeypatch.setattr(m.os,'fsync',sync);monkeypatch.setattr(m.kernel.paired,'fsync_dir',directory)
    result=m.write(path,{'x':1})
    assert [c[0] for c in calls]==['file','dir'] and result['sha256']==m.sha(path)
    with pytest.raises(ValueError):m.write(path,{'x':1})


class GateRAM(m.Store,native.MemoryStore):
    def __init__(self,midpoints):
        native.MemoryStore.__init__(self);self.midpoints=midpoints;self.matched=set();self.started=time.monotonic()
    def check(self,size=4096,final=False):
        return native.MemoryStore.check(self,size,final)
    def emit(self,kind,**fields):
        if kind=='run_finished':return native.MemoryStore.emit(self,kind,**fields)
        return super().emit(kind,**fields)


@pytest.fixture(scope='module')
def baseline():
    with pytest.MonkeyPatch.context() as patch:
        r,s,_,p,data=native.execute(patch);strict=native.assemble(r,s,p,data)
    return p,data,[r['rgb_encoding_inputs'][1] for r in strict['rows']]


def test_unchanged_native_execute_assemble_and_exact_mid_gate(monkeypatch,baseline):
    p,data,mid=baseline;s=GateRAM(mid)
    r,s,envs,p,data=native.execute(monkeypatch,p=deepcopy(p),data=data,store=s)
    result=native.assemble(r,s,p,data)
    assert len(result['rows'])==8 and s.matched==set(range(8))
    assert r['counts']['public_step']==dict(attempted=160,completed=160)
    assert r['counts']['physics_step']==dict(attempted=1632,completed=1632)
    assert all(len(e.commands)==20 for e in envs)


@pytest.mark.parametrize('field',['file_sha256','pixel_sha256','state','snapshot'])
def test_bad_mid_blocks_before_suffix_keeps_eight(monkeypatch,baseline,field):
    p,data,mid=baseline;mid=deepcopy(mid)
    for v in mid:
        if field=='state':v[field][0]+=1e-7
        elif field=='snapshot':v[field]['state_float64_bits']='BAD'
        else:v[field]='0'*64
    r,s,envs,_,_=native.execute(monkeypatch,p=deepcopy(p),data=data,store=GateRAM(mid))
    assert r['status']=='partial_or_failed' and len(r['rows'])==8
    assert all(row['own_returned']==5 for row in r['rows'])
    assert all(len(e.commands)==15 and e.closed for e in envs)


def test_step6_before_gate_rejected(baseline):
    with pytest.raises(ValueError):GateRAM(baseline[2]).emit('operation_attempted',resource='public_step',phase='plan',step=6,branch=0)


def test_authorization_before_io(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unexpected IO'))
    with pytest.raises(ValueError):m.run('/NONE','0'*64,'/NONE',lambda:pytest.fail('factory'),'BAD')


def test_timeout_final_uses_native_journal_ABI(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_RAM'),[],time.monotonic()-1000)
    with pytest.raises(ValueError):s.check()
    s.emit('run_finished',status='partial_or_failed')
    events=m.physical.journal(s.path/'journal.jsonl',s.previous)
    assert len(events)==1 and events[0]['kind']=='run_finished' and events[0]['status']=='partial_or_failed'


def test_run_dispatches_unchanged_native_executor(monkeypatch):
    p=dict(native_prepare={'RAM':1},midpoints=[],runtime_report_sha256='b'*64)
    monkeypatch.setattr(m,'verify',lambda *a:(p,{'RAM':2}))
    class S:
        def __init__(self,*a):pass
        def check(self):pass
    monkeypatch.setattr(m,'Store',S)
    def execute(pp,data,store,factory,**kw):
        assert pp is p['native_prepare'] and data=={'RAM':2}
        assert kw['authorization']==m.physical.AUTH
        assert kw['binding']['strict_receipt_sha256']==m.STRICT_SHA
        return {'status':'RAM_only'}
    monkeypatch.setattr(m.physical,'execute',execute)
    assert m.run('/RAM','a'*64,'/RAM',None,m.AUTH)['status']=='RAM_only'


def test_budget_and_science_contract():
    c=m.contract()
    assert c['plans']==8 and c['new_neural']==0 and c['total_FLOPs'] is None
    assert not c['P1_admission'] and not c['P2_admission'] and not c['equal_FLOPs']
    assert c['final_reserve_bytes']==2*1024**2


def test_consume_exports_full_plans_native_rows_and_report_SHA(monkeypatch,tmp_path,literal):
    c,parents,branches=literal;plans,reuse=m.zero_plans(c,parents,branches,'b'*64)
    p=dict(native_prepare={'plans':plans},midpoints=[],bindings={},runtime_report_sha256='b'*64,
        guarded_reuse=reuse,inherited_model_compute={'predict':32})
    b=m.execution_binding('/RAM','a'*64,p)
    (tmp_path/'execution_binding.json').write_bytes(m.raw(b))
    (tmp_path/'journal.jsonl').write_bytes(b'RAM')
    report=dict(execution_binding=b,execution_binding_sha256=m.sha(tmp_path/'execution_binding.json'),
        journal_sha256=m.sha(tmp_path/'journal.jsonl'),journal_chain_tail='RAM')
    (tmp_path/'report.json').write_bytes(m.raw(report));h=m.sha(tmp_path/'report.json')
    rows=[dict(index=i,final_metrics={'native_success':False}) for i in range(8)]
    monkeypatch.setattr(m,'verify',lambda *a:(p,{}))
    monkeypatch.setattr(m.physical,'journal',lambda *a:[])
    monkeypatch.setattr(m.physical,'assemble',lambda *a:dict(status='engineering_verified_science_blocked',rows=rows,PNG_files=[],counts={}))
    monkeypatch.setattr(m.bridge,'verify_bindings',lambda *a:None)
    result=m.consume(tmp_path,h)
    assert result['physical_report_sha256']==h and result['plans']==plans and result['rows']==rows
    assert all(len(plan['commands'])==10 for plan in result['plans'])
    with pytest.raises(ValueError):m.consume(tmp_path,'0'*64)


def test_wrong_pinned_receipt_refused_before_upstream(monkeypatch,tmp_path):
    receipt=tmp_path/'receipt.json';receipt.write_bytes(b'{}')
    monkeypatch.setattr(m.runtime,'material',lambda *a:pytest.fail('must reject before material'))
    with pytest.raises(ValueError,match='bound actual saved authority'):m.build(tmp_path,receipt)
