"""RAM/metadata-only tests. Mock run callbacks are NOT execution evidence."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_p4_runtime_assets_20261028_v1 as m


def metadata():
    contract=dict(layers=[list(x) for x in m.LAYERS],modes=['O','I','R'],episodes=12,warmup=10,
        horizon=10,feedback_step=5,with_target=False,P1_admission=False,P4_admission=False,
        independent_confirmation=False,physical_caps=dict(public_step=240),neural_caps=dict(predict=40),
        wall_seconds=600,output_cap_bytes=64*1024**2,disk_floor_bytes=200*1024**2,
        final_reserve_bytes=256*1024,cuda_allocated_cap=3*1024**3,cuda_reserved_cap=4*1024**3)
    p=dict(schema=m.NATIVE+'.prepare',contract=contract,layers=[dict(source_slot=s,arm=a) for s,a in m.LAYERS])
    report=dict(schema=m.NATIVE+'.report',status='live_feedback_calibration_recorded',fixed_denominator=12,
        independent_tasks=2,errors=[],contract=deepcopy(contract),rows=[dict(source_slot=s,prefix_arm=a,mode=mode)
            for s,a in m.LAYERS for mode in ('O','I','R')])
    return p,report


def test_fixed_actual_callback_inventory_not_new_algorithm():
    p,report=metadata();v=m.inspect_saved(p,report)
    assert v['callbacks']['dispatch']['symbol']=='run'
    assert v['callbacks']['encode_observation']['symbol']=='Handler.encode_observation'
    assert v['callbacks']['native_metrics']['symbol']=='native_transition'
    assert v['denominator']==dict(independent_tasks=2,layers=4,episodes=12,mode_slots=4)
    assert v['budget']['actual_new_work'] is v['budget']['flops'] is None
    assert not v['execution_authorized'] and not v['P4_admission'] and v['formal_efficacy_blockers']


@pytest.mark.parametrize('damage',['missing_row','wrong_order','subset','modes','feedback','target','error','admit','tasks'])
def test_saved_contract_mutations_rejected(damage):
    p,report=metadata()
    if damage=='missing_row':report['rows'].pop()
    elif damage=='wrong_order':report['rows'].reverse()
    elif damage=='subset':p['layers'].pop()
    elif damage=='modes':p['contract']['modes']=['R']
    elif damage=='feedback':p['contract']['feedback_step']=10
    elif damage=='target':p['contract']['with_target']=True
    elif damage=='error':report['errors']=['preserved']
    elif damage=='admit':p['contract']['P4_admission']=True
    else:report['independent_tasks']=12
    with pytest.raises(ValueError):m.inspect_saved(p,report)


def dispatch_fixture(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    value=dict(native_prepared='/RAM_only/prepared.json',native_prepare_sha256='a'*64)
    expected=m.c.digest(value);output=tmp_path/'results'/(m.NAME+'_RAM')
    monkeypatch.setattr(m,'verify',lambda v,h: m.require(v==value and h==expected,'input'))
    auth=dict(schema=m.NAME+'.MAIN_authority',reviewer='MAIN',input_sha256=expected,output=str(output),
        scope='known_root_single_feedback_engineering',host_cuda_and_physics_authorized=True,confirmatory=False)
    path=tmp_path/'authority.json'
    def write():path.write_bytes(m.c.encode(auth));return m.sha(path)
    calls=[];native=SimpleNamespace(__file__=str(m.DRIVER),AUTH=m.AUTH)
    def verify(*a):calls.append(('native_verify',a));return {}
    def run(*a,**kw):calls.append(('native_run',a,kw));return dict(status='RAM_ONLY',P4_admission=False)
    native.verify=verify;native.run=run
    def loader(name):calls.append(('import',name));return native
    monkeypatch.setattr(m.importlib,'import_module',loader)
    return value,expected,output,path,auth,write,calls,native


def test_dispatch_really_calls_original_entry_once_after_durable_intent(tmp_path,monkeypatch):
    v,h,out,path,a,write,calls,native=dispatch_fixture(tmp_path,monkeypatch)
    result=m.dispatch_known(v,h,out,path,write())
    assert result==dict(status='RAM_ONLY',P4_admission=False)
    assert calls[-1]==('native_run',(v['native_prepared'],v['native_prepare_sha256'],str(out)),dict(authorization=m.AUTH))
    launch=out.with_name(out.name+'.launch.json');record=m.c.bound(launch,m.sha(launch))
    assert record['actual_computation'] is None and record['episode_denominator']==12
    before=deepcopy(calls)
    with pytest.raises(FileExistsError):m.dispatch_known(v,h,out,path,write())
    assert calls==before  # no second import or execution, no retry


@pytest.mark.parametrize('key,value',[('confirmatory',True),('scope','formal_P4'),('reviewer','worker'),
    ('host_cuda_and_physics_authorized',False),('host_cuda_and_physics_authorized',1),('input_sha256','b'*64)])
def test_no_authority_no_import_or_dispatch(tmp_path,monkeypatch,key,value):
    v,h,out,path,a,write,calls,_=dispatch_fixture(tmp_path,monkeypatch);a[key]=value
    with pytest.raises(ValueError):m.dispatch_known(v,h,out,path,write())
    assert calls==[] and not out.with_name(out.name+'.launch.json').exists()


def test_failed_native_verify_keeps_launch_prevents_retry(tmp_path,monkeypatch):
    v,h,out,path,a,write,calls,native=dispatch_fixture(tmp_path,monkeypatch)
    def broken(*args):raise ValueError('original closure failure')
    native.verify=broken
    with pytest.raises(ValueError,match='original closure failure'):m.dispatch_known(v,h,out,path,write())
    assert not any(x[0]=='native_run' for x in calls)
    assert out.with_name(out.name+'.launch.json').exists()


def test_native_exception_not_replaced_by_mock_success(tmp_path,monkeypatch):
    v,h,out,path,a,write,calls,native=dispatch_fixture(tmp_path,monkeypatch)
    def broken(*args,**kwargs):raise RuntimeError('preserved partial native run')
    native.run=broken
    with pytest.raises(RuntimeError,match='preserved partial'):m.dispatch_known(v,h,out,path,write())
    assert out.with_name(out.name+'.launch.json').exists()


def test_fsync_failure_cannot_dispatch(tmp_path,monkeypatch):
    v,h,out,path,a,write,calls,_=dispatch_fixture(tmp_path,monkeypatch);auth_sha=write()
    def fail(fd):raise OSError('durability failed')
    monkeypatch.setattr(m.os,'fsync',fail)
    with pytest.raises(OSError):m.dispatch_known(v,h,out,path,auth_sha)
    assert calls==[]


def test_outside_or_existing_output_refused(tmp_path,monkeypatch):
    v,h,out,path,a,write,calls,_=dispatch_fixture(tmp_path,monkeypatch);out.mkdir()
    with pytest.raises(ValueError):m.dispatch_known(v,h,out,path,write())
    with pytest.raises(ValueError):m.dispatch_known(v,h,tmp_path/'bad',path,write())
    assert calls==[]
