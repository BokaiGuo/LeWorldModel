"""RAM saved fixtures only: no real model, environment, physics or outcome."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_common_resource_physical_summary_20261029_v1 as m
from test_agent_p2_common_resource_physical_20261029_v1 import complete, execute, reseal


def literal_strict(report, prepared):
    # An explicitly synthetic return-shaped fixture; production never accepts
    # this seam in lieu of invoking physical.consume and comparing all fields.
    return dict(schema=m.physical.NAME+'.consume',fixed_denominator=6,known_roots=2,fresh_roots=0,inherited=0,
        plans=deepcopy(prepared['plans']),planner_authority=deepcopy(prepared['planner_authority']),
        rows=deepcopy(report['rows']),counts=deepcopy(report['counts']),errors=deepcopy(report['errors']),
        new_neural_calls=0,new_physics_calls=0,**{k:v for k,v in m.FLAGS.items() if k!='selection'})


@pytest.fixture(scope='module')
def full_panel():
    report, _, prepared, _, _ = complete()
    return m.panel(prepared,literal_strict(report,prepared))


def test_all_six_eighteen_same_time_records_and_all_pairs(full_panel):
    p=full_panel
    assert p['fixed_branch_denominator']==6
    assert p['fixed_checkpoint_denominator']==p['fixed_pair_denominator']==18
    assert len(p['rows'])==len(p['pairs'])==18 and len(p['same_time_aggregates'])==9
    assert all(r['metrics'] is not None and not r['missing'] for r in p['rows'])
    assert all(r['known'] for r in p['pairs'])
    for pair in p['pairs']:
        a,b=[next(r for r in p['rows'] if (r['root_slot'],r['arm'],r['step'])==(pair['root_slot'],arm,pair['step'])) for arm in (pair['a'],pair['b'])]
        assert a['identity']==b['identity']
        for k in m.SCALARS: assert pair['delta_b_minus_a'][k]==b['metrics'][k]-a['metrics'][k]
    assert not p['scientific_admission'] and not p['equal_FLOPs'] and p['known_roots']==2


def test_independent_arithmetic_known_components_and_native_wrap():
    s=[3.,4.,6.,8.,2*np.pi+.5,5.,12.]
    q=m.scalars(s,[0.]*7)
    assert q['agent_xy_squared']==25 and q['block_xy_squared']==100
    assert q['joint_xy_squared']==125 and q['agent_velocity_squared']==169
    assert q['raw_angle_squared']==s[4]**2
    assert q['native7_squared']==sum(v*v for v in s)
    assert q['native_angle_term']==-.5 and q['negative_native_angle']
    assert q['native_success'] and not q['wrapped_success_diagnostic']
    assert q['agent_velocity_error']==13 and q['native7_reward']==-q['native7_distance']


def test_arithmetic_not_producer_metric_or_endpoint_labels(monkeypatch,full_panel):
    r,_,p,_,_=complete();u=literal_strict(r,p)
    for row in u['rows']:
        row['final_metrics']={'fabricated':True}
        row['final_state']=[999.]*7
        for response in row['responses']:
            response['native_metrics']={'fabricated':True};response['reward']=123456.;response['native_success']=None
    monkeypatch.setattr(m.physical.native,'metrics',lambda *a:pytest.fail('must not call producer metrics in panel'))
    assert m.panel(p,u)==full_panel


@pytest.mark.parametrize('bad',[[0.]*6,[0.]*6+[True],[0.]*6+[float('nan')],[0.]*6+[float('inf')]])
def test_bad_native_arrays_failclosed(bad):
    with pytest.raises(ValueError): m.scalars(bad,[0.]*7)


@pytest.mark.parametrize('at,known',[ (5,0),(15,6),(20,12),(25,18) ])
def test_earlystop_full_denominator_never_endpoint_padding(at,known):
    report,_,p,_,_=execute(fault='truncate',at=at)
    panel=m.panel(p,literal_strict(report,p))
    assert panel['known_checkpoint_scalars']==known and len(panel['rows'])==18
    for row in panel['rows']:
        if row['step']>max(0,at-10):
            assert row['metrics'] is row['error_reduction_from_original_root'] is None
            assert row['native7_reward_gain'] is None and 'native_stop_no_padding' in row['missing']
    assert sum(x['known'] for x in panel['pairs'])==known


def test_all_missing_exports_retained_not_zero_gain():
    r,_,p,_,envs=execute(missing=tuple(range(6)))
    panel=m.panel(p,literal_strict(r,p))
    assert not envs and panel['known_checkpoint_scalars']==0
    assert all(x['metrics'] is None and 'export_missing' in x['missing'] for x in panel['rows'])
    assert all(x['delta_b_minus_a'] is None for x in panel['pairs'])
    assert all(v['known_mean'] is None and v['known_sum'] is None for x in panel['same_time_aggregates'] for v in x['delta_b_minus_a'].values())


def test_engineering_return_without_cp_render_keeps_scalar_separately(full_panel):
    r,_,p,_,_=complete();u=literal_strict(r,p);row=u['rows'][0]
    row['checkpoints']=[f for f in row['checkpoints'] if f['step']!=15]
    row['missing_checkpoints']=[15];row['status']='engineering_failed';row['first_error']={'stage':'RAM_render_incident'}
    panel=m.panel(p,u);last=panel['rows'][2]
    assert last['metrics']==full_panel['rows'][2]['metrics'] and last['checkpoint'] is None
    assert last['missing']==['checkpoint_RGB_missing'] and last['first_error'] is not None


@pytest.mark.parametrize('mutation',['order','goal','task','checkpoint_state','missing_list','duplicate','source_root','flags'])
def test_identity_same_time_and_flags_mutations_rejected(mutation):
    r,_,original,_,_=complete();p=deepcopy(original);u=literal_strict(r,p)
    if mutation=='order': u['rows'].reverse()
    elif mutation=='goal':u['rows'][0]['goal'][0]+=1
    elif mutation=='task':u['rows'][0]['task_id']='wrong'
    elif mutation=='checkpoint_state':u['rows'][0]['checkpoints'][1]['state'][0]+=1
    elif mutation=='missing_list':u['rows'][0]['missing_checkpoints']=[15]
    elif mutation=='duplicate':u['rows'][0]['responses'].append(deepcopy(u['rows'][0]['responses'][-1]))
    elif mutation=='source_root':p['source_roots'][0]['root_state'][0]+=1
    else:u['P2_admission']=True
    with pytest.raises((ValueError,KeyError)):m.panel(p,u)


@pytest.fixture(scope='module')
def disk_fixture(tmp_path_factory):
    # The upstream mock producer's actual loop creates its full journal and PNG
    # bytes. Only source-material reconstruction is injected; full saved
    # lifecycle/hash/action/PNG strict below is real code, not a status stub.
    base=tmp_path_factory.mktemp('RAM_saved_common6')
    r0,store,p0,data,_=complete();r,p,events=deepcopy(r0),deepcopy(p0),deepcopy(store.events)
    prep=base/'RAM_prepare.json';prep.write_bytes(m.raw(p))
    b=r['execution_binding'];b['prepared']=str(prep);b['prepare_sha256']=m.sha(prep)
    events[0]['prepare_sha256']=m.sha(prep);reseal(events,r)
    files=dict(store.files)
    files['execution_binding.json']=m.raw(b)
    # Owner raw() already appends exactly one newline, like Store.emit.
    files['journal.jsonl']=b''.join(m.raw(e) for e in events)
    r['execution_binding_sha256']=hashlib.sha256(files['execution_binding.json']).hexdigest()
    r['journal_sha256']=hashlib.sha256(files['journal.jsonl']).hexdigest()
    files['report.json']=m.raw(r)
    output=base/'physical';output.mkdir()
    for name,body in files.items(): (output/name).write_bytes(body)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.physical,'verify',lambda *args:(deepcopy(p),data))
        strict=m.physical.consume(output,m.sha(output/'report.json'))
    receipt=base/'RAM_full_strict.json';receipt.write_bytes(m.raw(strict))
    return base,output,prep,receipt,p,data


@pytest.fixture
def authority(disk_fixture,monkeypatch):
    base,output,prep,receipt,p,data=disk_fixture
    monkeypatch.setattr(m.physical,'verify',lambda *args:(deepcopy(p),data))
    return dict(output=str(output),report_sha=m.sha(output/'report.json'),receipt=str(receipt),receipt_sha=m.sha(receipt))


def test_real_six_saved_fixture_complete_strict_build_verify_and_writer(authority,tmp_path,monkeypatch):
    called=[];original=m.physical.consume
    def spy(*a):called.append(a);return original(*a)
    monkeypatch.setattr(m.physical,'consume',spy)
    result=m.build_summary(**authority)
    assert len(called)==1 and result['summary']['known_checkpoint_scalars']==18
    assert result['new_physics_calls']==result['new_neural_calls']==0
    dest=tmp_path/(m.NAME+'_RAM.json')
    info=m.write_new(dest,result)
    assert m.verify(dest,info['sha256'])['verified'] and len(called)==2
    with pytest.raises(ValueError):m.write_new(dest,result)
    changed=deepcopy(result);changed['summary']['rows'][0]['metrics']['native7_distance']+=1
    altered=tmp_path/'RAM_tamper.json';altered.write_bytes(m.raw(changed))
    with pytest.raises(ValueError,match='entire summary'):m.verify(altered,m.sha(altered))


@pytest.mark.parametrize('kind',['status_only','changed_count','metric','wrong_sha'])
def test_external_whole_receipt_not_status_only(authority,tmp_path,kind):
    saved=m.read_bound(authority['receipt'],authority['receipt_sha'])
    if kind=='status_only':saved={'schema':saved['schema'],'status':saved['status']}
    elif kind=='changed_count':saved['counts']['physics_step']['completed']-=1
    elif kind=='metric':saved['metric_breakdown'][0]['metrics']['native7_distance']=123.
    file=tmp_path/'RAM_external.json';file.write_bytes(m.raw(saved))
    args=dict(authority,receipt=str(file),receipt_sha='0'*64 if kind=='wrong_sha' else m.sha(file))
    with pytest.raises(ValueError):m.build_summary(**args)


def test_actual_cli_in_process_with_real_saved_bytes(authority,tmp_path,capsys):
    dest=tmp_path/(m.NAME+'_CLI.json')
    m.main(['build','--physical-output',authority['output'],'--report-sha',authority['report_sha'],
        '--strict-receipt',authority['receipt'],'--strict-sha',authority['receipt_sha'],'--output',str(dest)])
    m.main(['verify','--summary',str(dest),'--summary-sha',m.sha(dest)])
    assert '"verified": true' in capsys.readouterr().out


def test_cold_import_no_torch_physics_or_network():
    code='''
import builtins
import socket
import urllib.request
def forbidden(*args,**kwargs):
    raise AssertionError('network operation forbidden')
socket.socket.connect=forbidden
socket.socket.connect_ex=forbidden
socket.create_connection=forbidden
socket.getaddrinfo=forbidden
urllib.request.urlopen=forbidden
original=builtins.__import__
def guarded(name,*args,**kwargs):
    assert name.split('.')[0] not in ('torch','gymnasium','pymunk','pygame','requests'),name
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from experiments.spectral_reachability import agent_p2_common_resource_physical_summary_20261029_v1
'''
    # The pinned parent's official-contract dependency imports urllib.request
    # inertly. Test network OPERATIONS, not importing a standard-library client;
    # neural/physical imports remain forbidden, including cold transitive ones.
    run=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_own_triple_and_frozen_upstream_pins():
    assert all(p.is_file() for p in m.own_files())
    assert all(m.sha(p)==h for p,h in m.PINS.items())
