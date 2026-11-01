"""Only synthetic RAM/saved PNG metadata; no env/model/physics execution."""
from copy import deepcopy
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_p2_composition_execution_20261028_v1 as m

spec = importlib.util.spec_from_file_location('p2_execution_ram_plans', m.ROOT/'tests/test_agent_p2_execution_assets_20261028_v1.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)


def fixture(tmp_path, *, stop=None, partial=False):
    """Manufacture metadata, not a simulated or executed research trajectory."""
    plans = m.assets.make_plans(*previous.fixture()); output = tmp_path/'RAM_saved'; output.mkdir()
    refs = {x['root_sha256']: dict(history_rgb_sha256='a'*64, goal_rgb_sha256='b'*64) for x in plans}
    p = dict(schema=m.assets.NAME+'.prepare', plans=plans, contract=m.assets.contract(), references=refs,
        formal_execution_authorized=False, P1_admission=False, P2_admission=False,
        inherited_resources=dict(fixture='RAM, not actual resource evidence'), bindings={})
    counters = {key: {phase: dict(attempts=0, returns=0) for phase in m.PHASES} for key in ('steps', 'resets')}
    budget = []; rows = []; review = 'c'*64
    def call(name, phase, plan=None, command=None):
        count = counters['steps' if name == 'env_step' else 'resets'][phase]
        before = dict(kind=name, phase=phase, index=count['attempts'], completed=False)
        if plan is not None: before.update({k: plan[k] for k in ('root_sha256', 'method', 'selection_sha256')})
        if command is not None: before['command'] = deepcopy(command)
        budget.extend([before, dict(before, completed=True)]); count['attempts'] += 1; count['returns'] += 1
    call('env_reset', 'root_collection')
    for i, asset in enumerate(plans):
        plan = asset['execution_plan']; stem = f'root_{plan["root_index"]}_{plan["method"]}'
        outer = dict(root_slot=asset['root_slot'], arm=asset['arm'], handoff_sha256=asset['handoff_sha256'], status='returned', physical=None)
        if partial and i > 0:
            outer['status'] = 'not_started'; rows.append(outer); continue
        call('env_reset', 'warmup', plan)
        mean = np.asarray(m.physical.helpers.NORMALIZATION['mean'], np.float32)
        raw_history = np.repeat(mean[None], 3, axis=0).tolist()
        for command in np.repeat(np.asarray(raw_history), 5, axis=0).tolist(): call('env_step', 'warmup', plan, command)
        identity = dict(refs[asset['root_sha256']], state=np.ones(7).tolist(), goal=np.ones(7).tolist(),
            **m.physical.helpers.validate_reset_history(raw_history, plan['predictor_context']['action_history']))
        events = []; trace = []; guard = m.physical.durable.RgbObservabilityGuard(root_id=plan['root_sha256'], agent_rgb=(65,105,225))
        def image_event(value, absent=False):
            rgb = np.zeros((224,224,3), np.uint8)
            if not absent: rgb[1:4,1:4] = [65,105,225]
            step = value.get('primitive_step',0); path = output/(stem+f'_rgb_{step}.png')
            Image.fromarray(rgb).save(path)
            events.append(dict(value, rgb_path=str(path), rgb_file_sha256=m.sha(path)))
            try: guard.observe(rgb, primitive_step=step)
            except m.physical.durable.ObservationUnavailable: assert absent
        absent_initial = stop == 'initial_guard' and i == 0
        image_event(dict(kind='initial', state=identity['state']), absent_initial)
        n = 0 if absent_initial else 2 if stop in ('native', 'guard') and i == 0 else 10
        state = identity['state']
        for step in range(1,n+1):
            command = plan['commands'][step-1]; guard.dispatch(command)
            events.append(dict(kind='attempt', primitive_step=step, command=command)); call('env_step', 'selected', plan, command)
            events.append(dict(kind='step_returned', primitive_step=step))
            state = deepcopy(identity['state']); state[0] += float(step)+(30. if asset['arm'] == m.assets.ARMS[1] else 0.)
            item = dict(kind='observation', primitive_step=step, command=command, state=state, reward=0.,
                terminated=stop == 'native' and i == 0 and step == n, truncated=False)
            trace.append(item); image_event(item, stop == 'guard' and i == 0 and step == n)
        status = ('observation_unavailable' if stop in ('guard','initial_guard') else 'environment_terminated') if stop and i == 0 else 'completed'
        result = {k: deepcopy(plan[k]) for k in ('root_index','root_sha256','method','selection_sha256','planned_horizon_tokens',
            'planned_commands','metric_label','schedule','active_metric_id')}
        journal = output/(stem+'.events.jsonl'); journal.write_bytes(b''.join(m.canonical(e)+b'\n' for e in events))
        goal = np.asarray(identity['goal']); arr = np.asarray(state)
        xy = float(np.linalg.norm(arr[2:4]-goal[2:4])); angle = float(abs(np.arctan2(np.sin(arr[4]-goal[4]),np.cos(arr[4]-goal[4]))))
        result.update(status=status, failures=[] if status != 'observation_unavailable' else [dict(type='ObservationUnavailable', phase='guard', message='RAM absent RGB')],
            trace=trace, attempted_commands=n, completed_commands=n, physical_env_step_attempts=n, physical_env_step_returns=n,
            full_horizon_completed=n == 10, terminal_rgb_path=events[-1]['rgb_path'], terminal_rgb_sha256=events[-1]['rgb_file_sha256'],
            endpoint_fresh=True, endpoint_scope=('full_H2' if n == 10 else 'partial_H2'), journal_path=str(journal),journal_sha256=m.sha(journal),
            context_identity=identity, final_state=state, guard=guard.receipt(), failure=None if status != 'observation_unavailable' else result.get('failures'),
            remaining_commands_not_executed=10-n, actual_endpoint_primitive_steps=n,
            official=m.labels(state,identity['goal']),pose=dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20.*angle))
        (output/(stem+'.physical.json')).write_bytes(m.canonical(result)); outer['physical'] = result; rows.append(outer)
    receipt = dict(counters, journal_errors=[],
        total_public_env_step_attempts=sum(x['attempts'] for x in counters['steps'].values()),
        total_public_env_step_returns=sum(x['returns'] for x in counters['steps'].values()),
        total_env_reset_attempts=sum(x['attempts'] for x in counters['resets'].values()),
        total_env_reset_returns=sum(x['returns'] for x in counters['resets'].values()),
        scope='public Gym step/reset calls; internal simulator substeps and reset-internal work unmeasured')
    (output/'physical_budget.jsonl').write_bytes(b''.join(m.canonical(e)+b'\n' for e in budget))
    prepared = tmp_path/'prepare.json'; prepared.write_bytes(m.canonical(p)); expected = m.sha(prepared)
    execution = dict(schema=m.assets.NAME+'.execution',prepared=str(prepared),prepare_sha256=expected,
        main_review_sha256=review,planner_report_sha256=m.assets.REPORT_SHA,known_root_calibration=True,P1_admission=False,P2_admission=False)
    (output/'execution.json').write_bytes(m.canonical(execution))
    errors = [dict(stage='execution',type='RuntimeError',message='RAM retained failure')] if partial else []
    report = dict(schema=m.assets.NAME+'.report',status='partial_or_failed' if partial else 'calibration_recorded',
        prepared=str(prepared),prepare_sha256=expected,rows=rows,errors=errors,first_error=errors[0] if errors else None,
        artifact_sha256={f.name:m.sha(f) for f in output.iterdir()},physical_calls=receipt,contract=m.assets.contract(),
        root_denominator=4,arm_denominator=8,model_loaded=False,new_neural_calls=0,
        native_success_count=sum(r['physical']['official']['official_success'] for r in rows if r['physical']),
        P1_admission=False,P2_admission=False,known_root_calibration=True)
    (output/'report.json').write_bytes(m.canonical(report))
    return p,report,execution,budget,output,prepared,expected,review


def test_complete_saved_numpy_png_consumer(tmp_path):
    p,r,e,b,out,*_ = fixture(tmp_path); value = m.validate_saved(p,r,e,b,out)
    assert value['status'] == 'saved_calibration_verified_not_admitted'
    assert len(value['rows']) == 8 and value['complete_branches'] == 8
    assert value['public_resources']['total_public_env_step_returns'] == 200
    assert [x['native_success'] for x in value['rows']].count(False) == 4  # retain negatives, no root selection
    assert all(value[k] is False for k in m.FLAGS)
    assert all(value['public_resources'][k] is None for k in ('internal_physics_substeps','render_calls','wall_seconds','constructor_calls'))


@pytest.mark.parametrize('stop',['native','guard','initial_guard'])
def test_realistic_saved_early_stop_no_padding(tmp_path, stop):
    p,r,e,b,out,*_ = fixture(tmp_path,stop=stop); value = m.validate_saved(p,r,e,b,out)
    assert len(value['rows']) == 8 and value['complete_branches'] == 7
    assert value['rows'][0]['returned_commands'] == (0 if stop == 'initial_guard' else 2)
    assert value['rows'][0]['full_horizon_completed'] is False
    assert value['status'] == 'saved_calibration_verified_not_admitted'


def test_partial_retains_all_denominators_and_null_unknowns(tmp_path):
    p,r,e,b,out,*_ = fixture(tmp_path,partial=True); value = m.validate_saved(p,r,e,b,out)
    assert value['status'] == 'partial_saved_verified_not_complete' and value['arm_denominator'] == 8
    assert value['complete_branches'] == 1
    assert all(x['native_success'] is x['returned_commands'] is x['physical'] is None for x in value['rows'][1:])


@pytest.mark.parametrize('fault',['root','missing_arm','reordered','commands','meter_command','meter_missing','meter_index',
    'meter_extra','meter_count','internal_zero','warmup','report_science','report_schema','physical_count','fake_complete',
    'initial_state','goal','official','object_only','state5','normalized','paired','png','PNG_mode','orphan','review_binding'])
def test_mutation_rejection(tmp_path,fault):
    p,r,e,b,out,*_ = fixture(tmp_path); row = r['rows'][0]; result = row['physical']
    if fault=='root':row['root_slot']=1
    elif fault=='missing_arm':r['rows'].pop()
    elif fault=='reordered':r['rows'].reverse()
    elif fault=='commands':p['plans'][0]['execution_plan']['commands'][0][0]=.3
    elif fault=='meter_command':next(x for x in b if x['phase']=='selected')['command']=[.4,.5]
    elif fault=='meter_missing':b.pop()
    elif fault=='meter_index':b[0]['index']=1
    elif fault=='meter_extra':b[0]['internal_physics_substeps']=0
    elif fault=='meter_count':r['physical_calls']['total_public_env_step_attempts']+=1
    elif fault=='internal_zero':r['physical_calls']['internal_physics_substeps']=0
    elif fault=='warmup':next(x for x in b if x['phase']=='warmup' and x['kind']=='env_step')['command']=[.5,.5]
    elif fault=='report_science':r['P1_admission']=True
    elif fault=='report_schema':r['schema']='Jason_two_moving_roots'
    elif fault=='physical_count':result['physical_env_step_returns']=9
    elif fault=='fake_complete':r['status']='partial_or_failed'
    elif fault=='initial_state':result['context_identity']['state'][0]+=1
    elif fault=='goal':result['context_identity']['goal'][0]+=50.
    elif fault=='official':result['official']['official_state_distance']=0.
    elif fault=='object_only':
        target=r['rows'][1]['physical']['official']
        assert target['official_success'] is False and target['object_only_success'] is True
        target['official_success']=target['object_only_success']
    elif fault=='state5':result['final_state']=result['final_state'][:5]
    elif fault=='normalized':result['context_identity']['normalized_between_history'][0][0]=1.
    elif fault=='paired':r['rows'][1]['physical']['context_identity']['goal']=[2.]*7
    elif fault=='png':Path(result['terminal_rgb_path']).write_bytes(b'bad')
    elif fault=='PNG_mode':Image.fromarray(np.zeros((224,224),np.uint8)).save(result['terminal_rgb_path'])
    elif fault=='orphan':r['artifact_sha256']['borrowed.json']='a'*64
    else:e['planner_report_sha256']='a'*64
    # Keep redundant branch JSON consistent so semantic mutations must be
    # rejected by actual saved arithmetic/identity checks, not only disk equality.
    for x in r['rows']:
        if x['physical'] is not None:
            value=x['physical'];path=out/f'root_{value["root_index"]}_{value["method"]}.physical.json'
            if path.exists():path.write_bytes(m.canonical(value))
    with pytest.raises((ValueError,KeyError)):m.validate_saved(p,r,e,b,out)


def patch_verify(monkeypatch,p,prepared,expected):
    def verify(path,h):
        m.require(str(path)==str(prepared) and h==expected and m.sha(path)==h,'RAM external prepare seal');return deepcopy(p)
    monkeypatch.setattr(m,'verify',verify)  # NEW module only, no frozen globals


def test_full_disk_consume_roundtrip(tmp_path,monkeypatch):
    p,r,e,b,out,prepared,expected,review = fixture(tmp_path);patch_verify(monkeypatch,p,prepared,expected)
    value=m.consume(out,m.sha(out/'report.json'),prepared,expected,review)
    assert value['report_sha256']==m.sha(out/'report.json') and len(value['rows'])==8
    assert value['consumer_sources']==m.source_check()


@pytest.mark.parametrize('fault',['report_sha','prepare_sha','review_sha','artifact_sha','extra_file','symlink','source','journal_line','duplicate_json'])
def test_external_seal_rejections(tmp_path,monkeypatch,fault):
    p,r,e,b,out,prepared,expected,review=fixture(tmp_path);patch_verify(monkeypatch,p,prepared,expected);reportsha=m.sha(out/'report.json')
    if fault=='report_sha':reportsha='0'*64
    elif fault=='prepare_sha':expected='0'*64
    elif fault=='review_sha':review='0'*64
    elif fault=='artifact_sha':r['artifact_sha256']['execution.json']='0'*64
    elif fault=='extra_file':(out/'extra').write_bytes(b'')
    elif fault=='symlink':(out/'extra_link').symlink_to(out/'execution.json')
    elif fault=='source':p['bindings'][str(prepared)]='0'*64
    elif fault=='journal_line':
        with (out/'physical_budget.jsonl').open('ab') as f:f.write(b'{}')
        r['artifact_sha256']['physical_budget.jsonl']=m.sha(out/'physical_budget.jsonl')
    else:
        (out/'execution.json').write_bytes(b'{"schema":1,"schema":2}')
        r['artifact_sha256']['execution.json']=m.sha(out/'execution.json')
    if fault in ('artifact_sha','journal_line','duplicate_json'):
        (out/'report.json').write_bytes(m.canonical(r));reportsha=m.sha(out/'report.json')
    with pytest.raises((ValueError,KeyError)):m.consume(out,reportsha,prepared,expected,review)


def test_MAIN_guard_before_io_or_runtime_import(monkeypatch):
    old=builtins.__import__;seen=[]
    def guarded(name,*args,**kw):
        if name.split('.')[0] in ('torch','gymnasium','pymunk'):seen.append(name);raise AssertionError('forbidden neural/physics import')
        return old(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='MAIN-only'):m.run_calibration(None,None,None,None,'not_authorized')
    assert seen==[]


def test_only_existing_runner_dispatch_and_no_new_manifest(tmp_path,monkeypatch):
    seen=[];monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m,'verify',lambda *a:seen.append('verify'))
    monkeypatch.setattr(m,'assets',type('RamRunner',(),dict(execute=staticmethod(lambda *a:seen.append(a) or 'RAM delegated'))))
    output=tmp_path/'results'/(m.NAME+'_RAM')
    assert m.run_calibration('sealed','a'*64,output,'b'*64,m.AUTH)=='RAM delegated'
    assert seen==['verify',('sealed','a'*64,output,'b'*64)] and not output.exists()


def test_receipt_writer_exclusive(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir();path=tmp_path/'results'/(m.NAME+'_RAM.json')
    assert m.write_receipt(path,{'RAM':True})['sha256']==m.sha(path)
    with pytest.raises(ValueError):m.write_receipt(path,{})
    with pytest.raises(ValueError):m.write_receipt(tmp_path/'results'/'other.json',{})


def test_cold_import_no_neural_or_physics_modules():
    script='import sys; from experiments.spectral_reachability import agent_p2_composition_execution_20261028_v1; assert not any(x.split(".")[0] in ("torch","gymnasium","pymunk") for x in sys.modules)'
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
