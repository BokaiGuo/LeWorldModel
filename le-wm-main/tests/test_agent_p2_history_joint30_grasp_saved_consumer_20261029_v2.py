"""Saved original error fixture and RAM exactness; no NN, AD or physics."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import subprocess
import sys
import builtins
import symtable
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_saved_consumer_20261029_v2 as m


@pytest.fixture(scope='module')
def actual_first_optimizer():
    output=m.ACTUAL_OUTPUT
    assert m.filehash(output/'report.json')==m.ACTUAL_REPORT_SHA
    r=m.readjson(output/'report.json')
    assert m.filehash(output/'events.jsonl.gz')==r['artifact_sha256']['events.jsonl.gz']
    c=m.Cursor(m.journal_lines(output/'events.jsonl.gz'));initialized=None
    # Read only bounded prefix, maintaining canonical chain verification.
    for line in m.journal_lines(output/'events.jsonl.gz'):
        payload=m.inputs.parse(line)['payload'];e=payload.get('event',payload)
        if payload.get('root_slot')==0:c.root=0
        c.get(e['kind'])
        if e['kind']=='initialized':initialized=e
        if e['kind']=='optimizer_return':
            assert c.n==41 and initialized is not None
            return initialized,e
    pytest.fail('missing original optimizer_return')


def test_exact_real_failure_signed_zero_not_nonzero(actual_first_optimizer):
    init,op=actual_first_optimizer;v=m.unpack(init['v'],(1,2,192),np.float32)
    assert op['v_before']['step']==0
    for key in ('m','q'):
        a=m.unpack(op['v_before'][key],(1,2,192),np.float32)
        assert np.count_nonzero(a)==0 and np.count_nonzero(np.signbit(a))==185
        assert op['v_before'][key]['sha256']=='0b9c761b27a2e0b287c3fedf401af232fba0fae3783087b1f5da0b86568c17bc'
    # Reproduce original failure without changing source or re-running its model.
    with pytest.raises(ValueError,match='persisted virtual Adam memory'):
        m.equal(op['v_before'],m.v1.blank((1,2,192),np.float32),'persisted virtual Adam memory')
    m.equal(op['v_before'],m.initial_memory(v),'v2 exact original parameter-times-zero')
    x=m.unpack(init['x'],(30,),np.float64)
    m.equal(op['x_before'],m.initial_memory(x),'v2 exact action initialization')


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_resets_use_current_parameter_signs_not_global_canonicalization(dtype):
    first=np.array([-2.,-0.,0.,3.],dtype);later=np.array([4.,-3.,-2.,-0.],dtype)
    a=m.initial_memory(first);b=m.initial_memory(later)
    assert m.unpack(a['m']).tobytes()==(first*dtype(0)).tobytes()
    assert m.unpack(b['m']).tobytes()==(later*dtype(0)).tobytes()
    assert a!=b and np.array_equal(m.unpack(a['m']),m.unpack(b['m']))


@pytest.mark.parametrize('field',['m','q','step'])
def test_zero_snapshot_mutations_still_fail_exactly(actual_first_optimizer,field):
    init,op=actual_first_optimizer;want=m.initial_memory(m.unpack(init['v']));bad=deepcopy(op['v_before'])
    if field=='step':bad['step']=1
    else:bad[field]=m.pack(np.zeros((1,2,192),np.float32))
    with pytest.raises(ValueError):m.equal(bad,want,'strict empty snapshot signs/dtype/step')


def test_initialized_moment_signedzeros_and_tiny_nonzero_remain_byte_strict():
    after=dict(step=1,m=m.pack(np.array([-0.,.1],np.float32)),q=m.pack(np.array([0.,.01],np.float32)))
    for key in ('m','q'):
        bad=deepcopy(after);a=m.unpack(bad[key]).copy();a[0]=-a[0];bad[key]=m.pack(a)
        with pytest.raises(ValueError):m.equal(bad,after,'persisted initialized memory')
        bad=deepcopy(after);a=m.unpack(bad[key]).copy();a[0]=np.nextafter(np.float32(0),np.float32(1));bad[key]=m.pack(a)
        with pytest.raises(ValueError):m.equal(bad,after,'no small-moment tolerance')


def test_validate_root_only_delta_is_exact_fresh_memory_construction():
    old=inspect.getsource(m.v1.validate_root).strip();new=inspect.getsource(m.validate_root).strip()
    expected=old.replace('mx=blank((30,),np.float64)','mx=initial_memory(x)').replace(
        'mv=blank((1,2,192),np.float32)','mv=initial_memory(v)')
    assert new==expected
    # Old arithmetic bounds/census/native arrays and source binding unchanged.
    assert m.Saved is m.v1.Saved
    assert m.loss_check is m.v1.loss_check and m.close_arithmetic is m.v1.close_arithmetic
    assert m.verify is m.v1.verify and m.Cursor is m.v1.Cursor


def test_sealed_producer_and_new_triple_closed():
    b=m.consumer_bindings()
    assert len(b)==8 and all(m.filehash(p)==h for p,h in b.items())
    for p,h in m.V1_PINS.items():assert b[str(m.ROOT/p)]==h


def test_own_consumer_no_run_prepare_or_old_globals_patch():
    tree=ast.parse(Path(m.__file__).read_text());functions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert functions=={'consumer_bindings','initial_memory','native_lerp','adam_check','validate_root','consume','main'}
    s=Path(m.__file__).read_text();assert 'v1.blank=' not in s and 'v1.validate_root=' not in s
    assert "schema=NAME+'.consume'" in s and "r['schema']==v1.NAME+'.report'" in s


def test_all_saved_function_global_dependencies_resolved():
    table=symtable.symtable(Path(m.__file__).read_text(),m.__file__,'exec')
    def visit(t):
        for symbol in t.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                assert symbol.get_name() in vars(m) or hasattr(builtins,symbol.get_name()),symbol.get_name()
        for child in t.get_children():visit(child)
    for child in table.get_children():visit(child)


def test_adam_only_reference_formula_changed_not_any_tolerance():
    old=inspect.getsource(m.v1.adam_check).strip()
    expected=old.replace('want_m=m+.1*(grad-m);want_q=.999*q+.001*grad*grad',
        'want_m=native_lerp(m,grad);want_q=.999*q+(1-.999)*grad*grad')
    assert inspect.getsource(m.adam_check).strip()==expected


def test_real_float64_beta_complement_cancellation_fixture():
    # Actual root0 joint2 coordinate28, step3: direct .1 != installed1-.9.
    old=np.array([.0014992054751739488],np.float64)
    grad=np.array([-.01341571158008321],np.float64)
    actual=np.array([7.713769648233314e-06],np.float64)
    wrong=old+.1*(grad-old)
    with pytest.raises(ValueError):m.close_arithmetic(actual,wrong,eps=np.finfo(np.float64).eps,ops=64)
    assert m.native_lerp(old,grad).tobytes()==actual.tobytes()


def test_real_float32_cuda_fma_cancellation_fixture():
    # Actual root0 joint10 coordinate176, step11: FMA not two rounded ops.
    old=np.array([.00025037897285073996],np.float32)
    grad=np.array([-.002266007475554943],np.float32)
    actual=np.array([-1.2596873375514406e-06],np.float32)
    wrong=old+(1-.9)*(grad-old)
    with pytest.raises(ValueError):m.close_arithmetic(actual,wrong,eps=np.finfo(np.float32).eps,ops=64)
    assert m.native_lerp(old,grad).tobytes()==actual.tobytes()


def test_real_fma_fixtures_have_actual_journal_origin():
    assert m.filehash(m.ACTUAL_OUTPUT/'report.json')==m.ACTUAL_REPORT_SHA
    request=grad=None;seen=set()
    for line in m.journal_lines(m.ACTUAL_OUTPUT/'events.jsonl.gz'):
        p=m.inputs.parse(line)['payload']
        if p.get('root_slot')!=0:continue
        e=p['event']
        if e['kind']=='optimization_attempt':request=e
        elif e['kind']=='backward_return':grad=e
        elif e['kind']=='optimizer_return' and request['phase']=='joint' and request['iteration'] in (2,10):
            k=request['iteration'];key,index=('x',28) if k==2 else ('v',176)
            before=m.unpack(e[key+'_before']['m']);g=m.unpack(grad['g'+key]);after=m.unpack(e[key+'_after']['m'])
            expected=m.native_lerp(before,g)
            assert after.flat[index].tobytes()==expected.flat[index].tobytes()
            seen.add(k)
            if seen=={2,10}:break
    assert seen=={2,10}


def test_cold_saved_only_import_refuses_neural_foreign_network():
    code='''import builtins,sys
original=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in ('torch','transformers','tensorflow','gymnasium','grasp_solver'):raise AssertionError(name)
 return original(name,*a,**k)
builtins.__import__=guard
def audit(e,args):
 if e in ('socket.__new__','socket.connect','socket.getaddrinfo'):raise AssertionError(e)
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_saved_consumer_20261029_v2 as m
assert m.initial_memory(m.np.array([-1.],m.np.float32))['step']==0
assert 'torch' not in sys.modules
'''
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
