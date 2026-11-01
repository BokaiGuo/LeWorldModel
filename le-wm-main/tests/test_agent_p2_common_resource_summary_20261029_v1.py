"""RAM/saved array fixtures, never CUDA/neural/physics or production outputs."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_common_resource_summary_20261029_v1 as m
from test_agent_p2_common_resource_grasp_cuda_20261029_v1 import saved_six


def literal_panel():
    """Synthetic receipt-shaped data: not a runtime/gradient certification."""
    rt=m.runtime;slots=[]
    for root in (0,1):
        slots.append(dict(root_slot=root,condition='off',task_id='RAM_task_'+str(root),task_file='/RAM/task'+str(root),
            task_sha256=str(root)*64,q=np.eye(192).tolist(),warm_raw10=[[.1,0]]*5+[[0,.1]]*5,
            packet=dict(root_context=rt.grasp.contexts(np.zeros((3,192),np.float32),np.zeros((2,10),np.float32)),goal=[1.]*192)))
    p=dict(slots=slots,expected_model_hash='RAM_MODEL')
    report=dict(model_before='RAM_MODEL',model_after='RAM_MODEL')
    receipt=dict(schema=rt.NAME+'.consume',status='all6_common_cap_records_verified_not_admitted',fixed_denominator=6,
        contract=rt.contract(),report_sha256='a'*64,plans=[],accounts=[],resources=[])
    traces=[];x=np.zeros(30,np.float64)
    for i,(root,arm) in enumerate(rt.ORDER):
        s=slots[root];contexts=[rt.grasp.contexts(np.full((3,192),v,np.float32),np.zeros((2,10),np.float32)) for v in (.25,.5,.75)]
        q=m.scalar_q64(contexts[-1]['latent_history'][-1],s['packet']['goal'])
        plan=dict(schema=rt.NAME+'.plan',root_slot=root,arm=arm,task_id=s['task_id'],task_file=s['task_file'],task_sha256=s['task_sha256'],
            root_context=deepcopy(s['packet']['root_context']),goal=deepcopy(s['packet']['goal']),warm_raw10=deepcopy(s['warm_raw10']),
            model_hash=p['expected_model_hash'],report_sha256=receipt['report_sha256'],commands=np.zeros((15,2),np.float32).tolist(),
            raw_float32_bits=np.zeros((15,2),dtype='<f4').tobytes().hex(),contexts=contexts,model_Q=q,selection_digest='b'*64)
        a=dict(elapsed_seconds=100.,stop='natural_completion',pending=[])
        t=dict(event_counts=Counter({'native:export_complete':1}),cpu_attempt=Counter(),cpu_return=Counter(),refusals=[],error=None,
            commits=0,accepted_commits=0,last_complete=None,last_accepted=None,initialized=None)
        if arm=='grasp':
            a.update(completed_units=632,incomplete_unit=False,completed=dict(packet=1902))
            t.update(commits=632,accepted_commits=632,last_complete=dict(x=rt.grasp.pack(x),journal_ordinal=120,accepted=True),
                last_accepted=dict(x=rt.grasp.pack(x),journal_ordinal=120,accepted=True))
        else:
            a.update(root=root,arm=arm,finished_units=rt.wall.protocol.schedule(arm),active_unit=None,export_complete=True,errors=[],
                incumbent=dict(witness=dict(logits=x.tolist(),contexts=contexts),cost=q,tag='RAM_retained_incumbent'))
        number=1062 if arm=='composition' else 1902
        resources={field:0 for field in m.FIELDS if field!='AD_primal_packet_steps_requested'}
        resources.update(packet=number,predict_API=number,predict_samples=number,action_encoder=number,predictor=number,pred_proj=number,
            predictor_context_tokens=3*number,pred_proj_rows=3*number,scalar_backward=632 if arm=='grasp' else 0,
            AD_primal_packet_steps=1896 if arm=='grasp' else 129 if arm=='composition' else 255,
            Jacobian=0 if arm=='grasp' else 85,mathematical_VJP_requested=0 if arm=='grasp' else 16320,
            root_SVD=0 if arm=='grasp' else 2)
        res=dict(attempted=deepcopy(resources),completed=deepcopy(resources),engine_internal_backward_calls=None,independent_JVP_calls=None,total_FLOPs=None)
        receipt['plans'].append(plan);receipt['accounts'].append(a);receipt['resources'].append(res);traces.append(t)
    return p,report,receipt,traces


def test_pure_panel_all_six_and_independent_native192_Q():
    p,r,c,t=literal_panel();out=m.panel(p,r,c,t)
    assert len(out['rows'])==6 and len(out['paired_roots'])==2
    assert all(v['model']['packet_goal_Q64']==[.5625,.25,.0625] for v in out['rows'])
    assert out['rows'][0]['boxes']['predict_API']['at_cap']
    assert not out['rows'][1]['boxes']['predict_API']['at_cap']
    assert all(v['CPU_capture']['seconds'] is None for v in out['rows'])
    assert out['rows'][2]['resources']['AD_primal_packet_steps_requested']['attempted'] is None


@pytest.mark.parametrize('change',['task','source','model','context','goal','warm','arm','raw','raw_shape','Q','logit_domain','count'])
def test_join_or_metric_mutations_never_silently_pass(change):
    p,r,c,t=literal_panel();v=c['plans'][0]
    if change=='task':v['task_id']='other'
    elif change=='source':v['task_sha256']='f'*64
    elif change=='model':v['model_hash']='other'
    elif change=='context':v['root_context']['latent_history'][0][0]=.01
    elif change=='goal':v['goal'][0]=.5
    elif change=='warm':v['warm_raw10'][0][0]=.3
    elif change=='arm':v['arm']='composition'
    elif change=='raw':v['raw_float32_bits']='00'
    elif change=='raw_shape':v['commands']=[[0,0]]
    elif change=='Q':v['model_Q']+=.01
    elif change=='logit_domain':c['accounts'][0]['incumbent']['witness']['logits']=[10.]*30
    else:c['plans']=c['plans'][:-1]
    with pytest.raises(ValueError):m.panel(p,r,c,t)


@pytest.mark.parametrize('which',['joint','grasp'])
def test_missing_not_replaced_by_last_complete_or_zero(which):
    p,r,c,t=literal_panel();i=0 if which=='joint' else 2
    c['plans'][i]=None;a=c['accounts'][i];a['stop']='wall_cap';a['elapsed_seconds']=451.
    if which=='joint':a['export_complete']=False;a['active_unit']='joint:0:1';a['pending']=[7]
    else:
        t[i]['event_counts']['native:export_complete']=0;a['incomplete_unit']=True;a['pending']=[['optimizer_attempt','Adam_step']]
        t[i]['error']=dict(type='RuntimeError',message='RAM partial failure',journal_ordinal=100)
    row=m.panel(p,r,c,t)['rows'][i]
    assert row['model'] is None and row['last_accepted_or_incumbent'] is not None
    assert 'no_strict_exported_plan' in row['missing'] and 'no_complete_export_replay' in row['missing']
    assert row['boxes']['search_wall']['overrun'] and row['pending']
    if which=='grasp':assert row['actual_error']['message']=='RAM partial failure'


def test_resource_box_refusal_is_distinct_from_reached_predict_cap():
    p,r,c,t=literal_panel();a=c['accounts'][0];a['stop']='resource_cap';c['plans'][0]=None
    c['resources'][0]['attempted']['predict_API']=1894;c['resources'][0]['completed']['predict_API']=1894
    row=m.panel(p,r,c,t)['rows'][0]
    assert row['boxes']['resource_cap_stop'] and not row['boxes']['predict_API']['at_cap']
    assert row['boxes']['predict_samples']['at_cap'] # Separate axes, not their sum.


def test_float64_not_float32_reduction_and_invalid_shape():
    terminal=np.linspace(-10000,10000,192,dtype=np.float32);goal=np.linspace(-.1,.1,192,dtype=np.float32)
    score=m.scalar_q64(terminal,goal)
    assert score==float(np.mean((terminal.astype(np.float64)-goal.astype(np.float64))**2))
    assert score!=float(np.mean((terminal-goal)**2,dtype=np.float32))
    with pytest.raises(ValueError):m.scalar_q64([0.]*191,[0.]*192)
    with pytest.raises(ValueError):m.scalar_q64([float('nan')]*192,[0.]*192)


def saved_authority(tmp_path,monkeypatch):
    """Use actual frozen runtime strict on its RAM six-file producer fixture."""
    rt=m.runtime;output=saved_six(tmp_path,monkeypatch)
    report=rt.old.parent.overlay.read(output/'report.json');prepared=Path(report['prepared'])
    # The parent fixture uses an inert prepare sentinel. This fixture supplies
    # real JSON authority bytes, still behind an explicitly mocked source build.
    p=rt.verify(None,None);prepared.write_bytes(rt.old.raw(p));ph=rt.old.sha(prepared)
    report['prepare_sha256']=ph
    loader=rt.old.parent.overlay.read(output/'loader.json');loader['prepare_sha256']=ph
    (output/'loader.json').write_bytes(rt.old.raw(loader))
    (output/'binding.json').write_bytes(rt.old.raw(dict(prepared=str(prepared),prepare_sha256=ph)))
    report['artifact_sha256']={name:rt.old.sha(output/name) for name in report['artifact_sha256']}
    (output/'report.json').write_bytes(rt.old.raw(report));rh=rt.old.sha(output/'report.json')
    actual=rt.consume(output,rh);receipt=tmp_path/'exact_receipt.json';receipt.write_bytes(rt.old.raw(actual))
    return output,rh,receipt,rt.old.sha(receipt)


def test_full_runtime_strict_rebuilt_receipt_exact_six_partial_files(tmp_path,monkeypatch):
    args=saved_authority(tmp_path,monkeypatch)
    actual=m.runtime.consume;calls=[]
    def spy(output,sha):calls.append((output,sha));return actual(output,sha)
    monkeypatch.setattr(m.runtime,'consume',spy)
    result=m.build_summary(*args)
    assert len(calls)==1 and result['fixed_denominator']==6
    assert all(v['model'] is None for v in result['rows'])
    assert result['rows'][2]['actual_error']['type']=='RuntimeError'
    assert result['rows'][2]['resources']['Adam_step']==dict(attempted=1,completed=0)
    assert result['authority']['full_imported_runtime_strict_rebuilt']
    assert result['authority']['independent_gradient_verification'] is False
    assert result['selected_winner'] is None and result['equal_FLOPs'] is False
    assert result['inherited_encoding_cohort']['encoder']==16


@pytest.mark.parametrize('change',['receipt_status_only','receipt_resources','wrong_report_sha','receipt_sha','root_plan','post_hash_file'])
def test_external_authorities_fail_closed(tmp_path,monkeypatch,change):
    args=list(saved_authority(tmp_path,monkeypatch));rt=m.runtime
    if change in ('receipt_status_only','receipt_resources','root_plan'):
        value=rt.old.parent.overlay.read(args[2])
        if change=='receipt_status_only':value={'status':value['status']}
        elif change=='receipt_resources':value['resources'][0]['attempted']['predict_API']+=1
        else:value['plans'][0]={'root_slot':1}
        args[2].write_bytes(rt.old.raw(value));args[3]=rt.old.sha(args[2])
    elif change=='wrong_report_sha':args[1]='0'*64
    elif change=='receipt_sha':args[3]='0'*64
    else:(args[0]/'binding.json').write_bytes(b'{}')
    with pytest.raises(ValueError):m.build_summary(*args)


def test_write_verify_cli_full_saved_rebuild_and_no_overwrite(tmp_path,monkeypatch,capsys):
    output,rh,receipt,sh=saved_authority(tmp_path,monkeypatch)
    path=tmp_path/(m.NAME+'_RAM.json')
    assert m.main(['build','--runtime-output',str(output),'--report-sha',rh,'--strict-receipt',str(receipt),'--strict-sha',sh,'--output',str(path)])==0
    expected=m.runtime.old.sha(path)
    assert m.main(['verify','--summary',str(path),'--summary-sha',expected])==0
    assert '"verified":true' in capsys.readouterr().out
    with pytest.raises(ValueError):m.write_new(path,{})
    value=m.read_bound(path,expected);value['rows'][0]['resources']['predict_API']['attempted']+=1
    path.write_bytes(m.runtime.old.raw(value))
    with pytest.raises(ValueError):m.verify(path,m.runtime.old.sha(path))


def test_inert_cold_import_and_cli_no_torch_physics_network():
    code='''import builtins,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','transformers','gymnasium','pymunk','pygame'):raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guard
def audit(event,args):
 if event in ('socket.__new__','socket.connect','socket.getaddrinfo'):raise AssertionError(event)
sys.addaudithook(audit)
from experiments.spectral_reachability import agent_p2_common_resource_summary_20261029_v1 as m
assert 'torch' not in sys.modules
assert all(m.runtime.old.sha(path)==sha for path,sha in m.PINS.items())
'''
    completed=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    help_=subprocess.run([sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME,'--help'],capture_output=True,text=True,check=True)
    assert 'build,verify' in help_.stdout


def test_own_triple_and_frozen_runtime_pins():
    assert all(p.is_file() for p in m.own_files())
    assert all(m.runtime.old.sha(p)==h for p,h in m.PINS.items())
