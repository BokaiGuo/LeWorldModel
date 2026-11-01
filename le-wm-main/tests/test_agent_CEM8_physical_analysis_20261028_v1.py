"""Synthetic saved dictionaries only: no environment/model/producer execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_CEM8_physical_analysis_20261028_v1 as m


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def native(goal, state):
    d = np.asarray(state, np.float64)-np.asarray(goal, np.float64)
    a = min(float(abs(d[4])), 2*np.pi-float(abs(d[4])))
    w = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    return dict(native_success=bool(np.linalg.norm(d[:4]) < 20 and a < np.pi/9),
                native_distance=float(np.linalg.norm(d)), native_angle_term=a,
                negative_native_angle=bool(a < 0), wrapped_allxy_success=bool(np.linalg.norm(d[:4]) < 20 and w < np.pi/9),
                wrapped_object_angle_error=w, object_position_error=float(np.linalg.norm(d[2:4])),
                object_pose=list(state[2:5]), object_success=None)


def fixture():
    plans, anchors, cr, sr, cs, ss = [], [], [], [], [], []
    for i, (slot, condition, arm) in enumerate(m.ORDER):
        commands = np.full((10, 2), np.float32(.1+i/100), np.float32).tolist()
        original = dict(root_slot=slot, condition=condition, arm=arm, task_id=h(('task', i)),
                        selection_digest=h(('selection', i)), blind_path='/RAM/blind'+str(i), blind_sha256=h(('blind', i)),
                        source_report_sha256=h('source'), planner_report_sha256=h('planner'), commands=commands)
        goal = [0.]*7
        root = [60., 60., 30., 30., .2, 1., 2.]
        start = [50., 50., 30., 30., .2, 1., 2.]
        def cp(step, state):
            return dict(step=step, png=f'b{i}_{step}.png', file_sha256=h((i,step,'file')),
                        pixel_sha256=h((i,step,'pixel')), state=state[:], snapshot=dict(state=state[:]))
        anchor = dict(index=i, checkpoint=cp(5, start), trace=[dict(ordinal=n) for n in range(51)])
        def row(index, cem, variant=None):
            suffix = commands[5:]
            plan = (dict(anchor=i, source_slot=slot, root_slot=slot, condition=condition, arm='teacher_full_CEM64',
                         prefix_arm=arm, task_id=original['task_id'], blind_sha256=original['blind_sha256'],
                         original_selection_digest=original['selection_digest'], prefix_commands=commands[:5],
                         suffix_commands=suffix, commands=commands, whole_run_report_sha256=h('CEMcompiler')) if cem else
                    dict(index=index, anchor=i, root_slot=slot, condition=condition, arm=arm, variant=variant,
                         task_id=original['task_id'], selection_digest=original['selection_digest'],
                         original_plan=deepcopy(original), prefix=commands[:5], suffix=suffix, commands=commands))
            responses=[]
            for n in range(1,6):
                state=[50.-n, 50., 30., 30., .2, 1., 2.]; metrics=native(goal,state)
                delta=np.asarray(state)-start
                responses.append(dict(step=n, state=state, delta_native7=delta.tolist(),
                    delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),
                    reward=-metrics['native_distance'], native_success=metrics['native_success'],
                    terminated=False, truncated=False, native_metrics=metrics))
            checkpoints=[cp(0,root), cp(5,start), cp(10,responses[-1]['state'])]
            r=dict(index=index, anchor_index=i, root_slot=slot, condition=condition,
                   arm='teacher_full_CEM64' if cem else arm, task_id=original['task_id'],
                   selection_digest=original['selection_digest'], selected_suffix=dict(plan=plan),
                   status='suffix_completed',first_error=None,secondary_errors=[],start_verified=True,
                   start_state=start[:],root_state=root[:],goal=goal[:],goal_pose=[1.,2.,3.],
                   root_sha256=hashlib.sha256(np.asarray(root+goal,np.float64).tobytes()).hexdigest(),
                   goal_pixel_sha256=h(('goal',i)),responses=responses,checkpoints=checkpoints,
                   suffix_returned=5,full_suffix=True,missing_checkpoints=[], final_state=responses[-1]['state'][:],
                   final_metrics=deepcopy(responses[-1]['native_metrics']),native_success=False,terminated=False,truncated=False)
            if cem:r['prefix_arm']=arm
            else:r['variant']=variant
            return r
        plans.append(original); anchors.append(anchor)
        c=row(i,True);cr.append(c);cs.append(deepcopy(c['selected_suffix']))
        for j,v in enumerate(m.VARIANTS):
            r=row(i*3+j,False,v);sr.append(r);ss.append(deepcopy(r['selected_suffix']))
    def receipt(cem, rows):
        return dict(schema=(m.CEM if cem else m.SPECTRAL)+'.consume', status='partial_or_failed',
                    rows=rows,fixed_denominator=8 if cem else 24, physical_report_sha256=h('cem' if cem else 'spectral'),
                    compiler_report_sha256=h('CEMcompiler' if cem else 'spectralcompiler'),consumer_report_sha256=h('accept'),
                    P1_admission=False, scientific_admission=False, independent_confirmation=False,new_neural_calls=0,counts={})
    def prepared(cem, suffix):
        return dict(schema=(m.CEM if cem else m.SPECTRAL)+'.prepare', plans=deepcopy(plans),anchors=deepcopy(anchors),suffix_plans=suffix,
                    compiler_report_sha256=h('CEMcompiler' if cem else 'spectralcompiler'),consumer_report_sha256=h('accept'),
                    physical_report_sha256=h('original'),bindings={})
    return receipt(True,cr),receipt(False,sr),prepared(True,cs),prepared(False,ss)


def test_fixed8_teacher_only_and_full_denominator():
    r=m.assemble(*fixture())
    assert len(r['rows'])==r['fixed_denominator']==8
    assert [r['spectral_ordinal'] for r in r['rows']]==list(range(1,24,3))
    assert r['summaries']['all']['cem']['endpoint']['denominator']==8
    assert r['summaries']['all']['cem']['endpoint']['available']==8
    assert r['summaries']['all']['cem']['endpoint']['native_success_count']==0
    assert set(r['summaries']['by_prefix_arm'])==set(m.ARMS)
    assert r['known_roots']==2 and r['fresh_roots']==0 and r['P1_admission'] is False
    assert r.get('independent_root_count') is None and r['new_independent_roots']==0
    assert r['consumes']['spectral24']['rows']==fixture()[1]['rows']


@pytest.mark.parametrize('mutation',['duplicate','order','observed4','variant','ordinal','condition','task','selection',
    'original_source','anchor','trace','goal','goal_pixel','root','cp0','cp5','prefixULP','rawsuffix','start','reward',
    'metric','success','chronology','delta','endpoint','cp10','missing','flags','compiler'])
def test_failclosed_mutations(mutation):
    c,s,p,q=fixture();r=c['rows'][0];t=s['rows'][1]
    if mutation=='duplicate':s['rows'][1]=deepcopy(s['rows'][4])
    elif mutation=='order':s['rows'].reverse()
    elif mutation=='observed4':c['rows']=c['rows'][:4];c['fixed_denominator']=4
    elif mutation=='variant':t['variant']='open_full'
    elif mutation=='ordinal':r['anchor_index']=1
    elif mutation=='condition':r['condition']='on'
    elif mutation=='task':r['task_id']=h('wrong')
    elif mutation=='selection':r['selection_digest']=h('wrong')
    elif mutation=='original_source':q['plans'][0]['source_report_sha256']=h('wrong')
    elif mutation=='anchor':q['anchors'][0]['checkpoint']['snapshot']['state'][0]+=1
    elif mutation=='trace':q['anchors'][0]['trace'][0]['ordinal']=1
    elif mutation=='goal':t['goal'][0]+=1
    elif mutation=='goal_pixel':t['goal_pixel_sha256']=h('wrong')
    elif mutation=='root':r['root_state'][0]+=1
    elif mutation=='cp0':t['checkpoints'][0]['pixel_sha256']=h('wrong')
    elif mutation=='cp5':r['checkpoints'][1]['state'][0]+=1
    elif mutation=='prefixULP':r['selected_suffix']['plan']['prefix_commands']=deepcopy(r['selected_suffix']['plan']['prefix_commands']);r['selected_suffix']['plan']['prefix_commands'][0][0]=float(np.nextafter(np.float32(.1),np.float32(1)))
    elif mutation=='rawsuffix':r['selected_suffix']['plan']['commands']=deepcopy(r['selected_suffix']['plan']['commands']);r['selected_suffix']['plan']['commands'][9][0]=0.
    elif mutation=='start':r['start_state'][0]+=1
    elif mutation=='reward':r['responses'][0]['reward']=0.
    elif mutation=='metric':r['responses'][0]['native_metrics']['native_distance']=0.
    elif mutation=='success':r['responses'][0]['native_success']=True
    elif mutation=='chronology':r['responses'][0]['step']=2
    elif mutation=='delta':r['responses'][0]['delta_native7'][0]+=1
    elif mutation=='endpoint':r['final_state'][0]+=1
    elif mutation=='cp10':r['checkpoints'][-1]['state'][0]+=1
    elif mutation=='missing':r['missing_checkpoints']=[10]
    elif mutation=='flags':c['P1_admission']=True
    else:c['compiler_report_sha256']=h('wrong')
    with pytest.raises((ValueError,KeyError)):m.assemble(c,s,p,q)


def test_earlystop_has_endpoint_but_no_padded_horizon():
    args=fixture();r=args[0]['rows'][0]
    r['responses']=r['responses'][:2];last=r['responses'][-1];last['truncated']=True
    r.update(status='native_stopped',suffix_returned=2,full_suffix=False,missing_checkpoints=[10],
             checkpoints=r['checkpoints'][:2],final_state=last['state'][:],final_metrics=deepcopy(last['native_metrics']),truncated=True)
    out=m.assemble(*args);row=out['rows'][0]['cem']
    assert row['endpoint'] is not None and row['horizon10'] is None and row['returned_suffix_steps']==2
    summary=out['summaries']['all']['cem']
    assert summary['endpoint']['available']==8
    assert summary['horizon10']['available']==7 and summary['horizon10']['mean'] is None
    assert summary['horizon10']['native_success_count'] is None


def test_failure_unknown_does_not_become_zero_or_false():
    args=fixture();r=args[0]['rows'][0]
    r.update(status='engineering_failed',first_error={'message':'RAM failure'})
    out=m.assemble(*args)
    assert out['rows'][0]['cem']['endpoint'] is None
    assert out['rows'][0]['cem']['observed_endpoint'] is not None
    summary=out['summaries']['all']['cem']['endpoint']
    assert summary['available']==7 and summary['mean'] is summary['native_success_count'] is None


def test_independent_native_quirk_and_decomposition():
    state=[0.,0.,0.,0.,3*np.pi,3.,4.];d=m.metrics([0.]*7,state)
    assert d['native']['native_success'] is True and d['native']['negative_native_angle'] is True
    assert d['native']['wrapped_allxy_success'] is False
    assert d['velocity_error']==5. and d['joint_xy_error']==0.
    assert d['native7_reward']==-np.linalg.norm(state)


@pytest.mark.parametrize('value',[None,[0]*6,[0]*8,[True]*7,[float('nan')]*7,[float('inf')]*7])
def test_bad_native_vectors_rejected(value):
    with pytest.raises(ValueError):m.metrics([0.]*7,value)


def test_output_exclusive_bounded_and_finite(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'free_bytes',lambda p:m.FLOOR+m.OUTPUT_CAP)
    path=tmp_path/'new.json';m.write_new(path,{'ok':True})
    assert json.loads(path.read_bytes())=={'ok':True}
    with pytest.raises(FileExistsError):m.write_new(path,{'ok':False})
    with pytest.raises(ValueError):m.write_new(tmp_path/'big.json',{'a':'a'*m.OUTPUT_CAP})
    with pytest.raises(ValueError):m.write_new(tmp_path/'nan.json',{'a':float('nan')})


@pytest.mark.parametrize('value',['0'*64,'x'*64,'a'*63,None])
def test_external_hash_required(value):
    with pytest.raises(ValueError):m.actual_hash(value)


def test_real_ABI_keeps_extra_original_plan_fields_without_dropping_source_checks():
    args=fixture()
    for row,selected in zip(args[1]['rows'],args[3]['suffix_plans'],strict=True):
        for plan in (row['selected_suffix']['plan'],selected['plan']):
            plan['original_plan'].update(schema='original.model_plan',prediction=[.1]*192,with_target=True)
    assert len(m.assemble(*args)['rows'])==8
    args[1]['rows'][1]['selected_suffix']['plan']['original_plan']['source_report_sha256']=h('bad')
    with pytest.raises(ValueError):m.assemble(*args)


def saved_boundary(monkeypatch,tmp_path):
    c,s,p,q=fixture();calls=[]; hashes={};docs={};modules={}; outputs=[]
    for name,receipt,prep in ((m.CEM,c,p),(m.SPECTRAL,s,q)):
        directory=tmp_path/name;directory.mkdir();outputs.append(str(directory))
        pp=str(tmp_path/(name+'_prepare.json'));rp=str(directory/'report.json')
        source=str(Path(m.__file__).resolve().with_name(name+'.py'))
        ph=h(pp);rh=receipt['physical_report_sha256']
        hashes.update({source:m.SOURCE_SHA[name],pp:ph,rp:rh})
        docs[pp]=deepcopy(prep)
        docs[rp]=dict(schema=name+'.report',status=receipt['status'],rows=deepcopy(receipt['rows']),
                      execution_binding=dict(prepared=pp,prepare_sha256=ph),errors=[],counts={})
        def consume(output,expected,receipt=receipt,name=name):
            calls.append((name,output,expected));return deepcopy(receipt)
        modules['experiments.spectral_reachability.'+name]=SimpleNamespace(__file__=source,consume=consume)
    hashes[str(Path(m.__file__).resolve())]=h('analyzer')
    monkeypatch.setattr(m,'sha',lambda path:hashes[str(path)])
    def read(path,expected,cap):
        if hashes[str(path)]!=expected:raise ValueError('mock SHA mismatch')
        return deepcopy(docs[str(path)])
    monkeypatch.setattr(m,'read_bound',read)
    monkeypatch.setattr(m.importlib,'import_module',lambda name:modules[name])
    return outputs,c,s,calls,hashes,docs,modules


def test_complete_both_consumers_called_once_and_preserved(monkeypatch,tmp_path):
    outputs,c,s,calls,hashes,docs,modules=saved_boundary(monkeypatch,tmp_path)
    r=m.analyze(outputs[0],c['physical_report_sha256'],outputs[1],s['physical_report_sha256'])
    assert [v[0] for v in calls]==[m.CEM,m.SPECTRAL]
    assert r['consumes']==dict(cem8=c,spectral24=s)
    assert r['input_sha256']==hashes
    assert r['physical_report_diagnostics']['cem8']['errors']==[]


@pytest.mark.parametrize('mutation',['cem_source','spectral_source','reportSHA','consume_raises','rows','post_drift','binding_conflict'])
def test_saved_boundary_tamper_and_real_consume_failure_propagate(monkeypatch,tmp_path,mutation):
    outputs,c,s,calls,hashes,docs,modules=saved_boundary(monkeypatch,tmp_path)
    if mutation in ('cem_source','spectral_source'):
        name=m.CEM if mutation=='cem_source' else m.SPECTRAL
        hashes[str(Path(m.__file__).resolve().with_name(name+'.py'))]=h('tamper')
    elif mutation=='reportSHA':c['physical_report_sha256']=h('wrong')
    elif mutation=='consume_raises':
        def fail(*a):raise ValueError('actual consume failed')
        modules['experiments.spectral_reachability.'+m.CEM].consume=fail
    elif mutation=='rows':docs[str(Path(outputs[0])/'report.json')]['rows'][0]['selection_digest']=h('wrong')
    elif mutation=='binding_conflict':
        doc=next(v for k,v in docs.items() if k.endswith(m.SPECTRAL+'_prepare.json'))
        doc['bindings']={str(Path(m.__file__).resolve()):h('wrong')}
    else:
        original=modules['experiments.spectral_reachability.'+m.SPECTRAL].consume
        def drift(*a):
            r=original(*a);hashes[str(Path(outputs[0])/'report.json')]=h('changed');return r
        modules['experiments.spectral_reachability.'+m.SPECTRAL].consume=drift
    with pytest.raises(ValueError):m.analyze(outputs[0],c['physical_report_sha256'],outputs[1],s['physical_report_sha256'])
    if mutation in ('cem_source','spectral_source','reportSHA'):assert calls==[]


def test_reentry_refused_before_consume(monkeypatch,tmp_path):
    path=tmp_path/'exists.json';path.touch()
    monkeypatch.setattr(m,'analyze',lambda *a:pytest.fail('must not consume'))
    with pytest.raises(ValueError):m.main(['--cem-output','/a','--cem-report-sha',h('a'),
        '--spectral-output','/b','--spectral-report-sha',h('b'),'--output',str(path)])


def test_free_floor_and_input_directory_writes_refused(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'free_bytes',lambda p:m.FLOOR-1)
    with pytest.raises(ValueError):m.write_new(tmp_path/'low.json',{})
    monkeypatch.setattr(m,'free_bytes',lambda p:m.FLOOR+m.OUTPUT_CAP)
    with pytest.raises(ValueError):m.write_new(tmp_path/'producer.json',{'input_directories':[str(tmp_path)]})
    assert list(tmp_path.iterdir())==[]


def test_truly_not_started_keeps_missing_identity_and_outcome_unknown():
    args=fixture();r=args[0]['rows'][0]
    r.update(status='not_started',first_error=None,responses=[],checkpoints=[],missing_checkpoints=[0,5,10],
             final_state=None,final_metrics=None,native_success=None,start_state=None,start_verified=False,
             suffix_returned=0,full_suffix=False)
    for field in ('goal','goal_pose','root_state','root_sha256','goal_pixel_sha256'):r.pop(field)
    out=m.assemble(*args);row=out['rows'][0]
    assert row['cem']['endpoint'] is row['cem']['horizon10'] is None
    assert row['spectral_teacher_full']['endpoint'] is None
    assert 'goal' in row['identity_unknown']
    assert out['consumes']['cem8']['rows'][0]['native_success'] is None


def test_bounded_metadata_parser_duplicate_keys_nonfinite_sha_and_symlink(tmp_path):
    path=tmp_path/'metadata.json'
    # apply_patch is for project edits; test fixture bytes are RAM/test-temp only.
    for body in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        path.write_bytes(body)
        with pytest.raises(ValueError):m.read_bound(path,hashlib.sha256(body).hexdigest(),1024)
    body=b'{"x":1}';path.write_bytes(body);expected=hashlib.sha256(body).hexdigest()
    assert m.read_bound(path,expected,1024)=={'x':1}
    with pytest.raises(ValueError):m.read_bound(path,expected,1)
    with pytest.raises(ValueError):m.read_bound(path,h('wrong'),1024)
    link=tmp_path/'link.json';link.symlink_to(path)
    with pytest.raises(ValueError):m.read_bound(link,expected,1024)


def test_cli_exact_arguments_routes_only_analyze_then_exclusive_write(monkeypatch,tmp_path,capsys):
    calls=[]
    def analyze(*args):calls.append(args);return {'status':'saved_not_admitted'}
    monkeypatch.setattr(m,'analyze',analyze)
    monkeypatch.setattr(m,'write_new',lambda p,r:h('receipt'))
    out=tmp_path/'new.json'
    m.main(['--cem-output','/cem','--cem-report-sha',h('cem'),
            '--spectral-output','/spectral','--spectral-report-sha',h('spectral'),'--output',str(out)])
    assert calls==[('/cem',h('cem'),'/spectral',h('spectral'))]
    assert json.loads(capsys.readouterr().out)['P1_admission'] is False


def test_write_seal_checks_final_input_drift_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'free_bytes',lambda _:m.FLOOR+m.OUTPUT_CAP)
    monkeypatch.setattr(m,'sha',lambda _:h('changed'))
    path=tmp_path/'new.json'
    with pytest.raises(ValueError):m.write_new(path,{'input_sha256':{'/RAM/source':h('old')}})
    assert not path.exists()


def test_successful_early_stop_is_native_success_not_H10_and_not_velocity_thresholded():
    args=fixture();r=args[0]['rows'][0];state=[1.,1.,1.,1.,.1,3.,4.]
    nm=native(r['goal'],state);delta=np.asarray(state)-r['start_state']
    response=dict(step=1,state=state,delta_native7=delta.tolist(),
                  delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),
                  native_metrics=nm,native_success=True,reward=-nm['native_distance'],terminated=True,truncated=False)
    r.update(status='native_stopped',responses=[response],suffix_returned=1,full_suffix=False,
             final_state=state[:],final_metrics=nm,native_success=True,terminated=True,truncated=False,
             missing_checkpoints=[10],checkpoints=r['checkpoints'][:2])
    out=m.assemble(*args);row=out['rows'][0]['cem'];summary=out['summaries']['all']['cem']
    assert row['endpoint']['native']['native_success'] is True and row['endpoint']['velocity_error']==5.
    assert row['horizon10'] is None and summary['endpoint']['native_success_count']==1
    assert summary['endpoint']['native_success_rate']==1/8 and summary['horizon10']['mean'] is None
