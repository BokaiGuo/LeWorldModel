"""No physics/neural work. Array fakes plus optional real saved evidence checks."""
from copy import deepcopy
import gzip
import inspect
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_sampling_physical_20261027_v1 as m

# Reuse frozen array-only fake ENV/packet fixtures, never the actual factory.
fakes=runpy.run_path(str(Path(__file__).with_name('test_p1_witness_physical_20261027_v1.py')))
class Env(fakes['Env']):
    """Local source-authenticatable array fake, no simulator inheritance."""


def inputs(monkeypatch,tmp_path):
    plan, sp, packet=fakes['fixture'](monkeypatch,tmp_path)
    sp.update(source_closure={str(Path(inspect.getsourcefile(Env)).resolve()):'e'*64},
        environment_sha256='e'*64,reset_semantics=dict(internal_physics_steps_per_reset=2,internal_renders_per_reset=1))
    plans=[]
    for root in range(2):
        for arm in m.ARMS:
            p=deepcopy(plan);p['arm']=arm;p['candidate_id']=root
            # Fake state is identical deliberately; run counts, not cohort identity,
            # are tested here. Production collect enforces unique actual roots.
            p['selection_sha256']=m.source.digest(dict(root=root,arm=arm))
            plans.append(p)
    sp['slots']=[dict(sp['slots'][0],candidate_id=i) for i in range(2)]
    return dict(contract=m.contract(),inputs=dict(plans=plans,source_plan=sp,inherited_cost={},sampling_cost={}))


def test_fixed_four_branch_mock_census_and_full_guard(monkeypatch,tmp_path):
    p=inputs(monkeypatch,tmp_path);monkeypatch.setattr(m,'verify',lambda *a:p)
    envs=[]
    def factory():
        env=Env();envs.append(env);return env
    out=tmp_path/'four'
    r=m.run('mock','a'*64,out,factory,authorize_physics=True)
    assert r['status']=='physical_calibration_recorded',r['errors']
    assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=8,step=40,render=44).items()}
    assert len(list(out.glob('*.png')))==12
    assert [x['arm'] for x in r['rows']]==['cem','iid_random']*2
    assert all(e.closed and len(e.reset_calls)==2 and len(e.step_calls)==10 for e in envs)
    assert all(len(x['guard_receipt']['observations'])==11 for x in r['rows'])
    assert all([c['step'] for c in x['checkpoints']]==[0,5,10] for x in r['rows'])
    assert r['reset_internal_physics_steps']==dict(completed_reset_implied=16,attempted_upper_bound=16,actual_observed=None)
    assert r['internal_goal_renders_completed_reset_implied']==8
    assert not r['P1_admission'] and r['neural_calls']==r['new_independent_samples']==0
    assert m.sha(out/'journal.jsonl')==r['journal_sha256']
    assert m.source.read_journal(out/'journal.jsonl')[-1]['kind']=='run_finished'


def test_native_terminal_no_replacement_or_completion(monkeypatch,tmp_path):
    p=inputs(monkeypatch,tmp_path);monkeypatch.setattr(m,'verify',lambda *a:p)
    envs=[]
    def factory():
        env=Env(term=3);envs.append(env);return env
    r=m.run('mock','a'*64,tmp_path/'terminal',factory,authorize_physics=True)
    assert r['status']=='partial_or_failed' and not r['errors']
    assert len(envs)==4 and all(len(e.step_calls)==3 for e in envs)
    assert r['counts']['step']==dict(attempted=12,completed=12)
    assert all(x['status']=='environment_terminated' and not x['full_horizon'] for x in r['rows'])


def test_primary_step_error_survives_close_error(monkeypatch,tmp_path):
    p=inputs(monkeypatch,tmp_path);monkeypatch.setattr(m,'verify',lambda *a:p)
    # Class source authentication remains the frozen fake's exact file.
    def factory():
        e=Env(fail_step=3)
        def close():raise RuntimeError('secondary_close')
        e.close=close;return e
    r=m.run('mock','a'*64,tmp_path/'errors',factory,authorize_physics=True)
    assert r['status']=='partial_or_failed'
    assert all(x['first_error']['message']=='step primary' for x in r['rows'])
    assert all(x['secondary_errors'][-1]['message']=='secondary_close' for x in r['rows'])
    assert r['counts']['step']==dict(attempted=12,completed=8)


def test_unauthorized_run_never_constructs_factory(tmp_path):
    def factory():raise AssertionError('must not execute')
    with pytest.raises(ValueError,match='authorization'):
        m.run('absent','a'*64,tmp_path/'absent',factory)
    assert not (tmp_path/'absent').exists()


def test_output_floor_rejects_before_environment_factory(monkeypatch,tmp_path):
    p=inputs(monkeypatch,tmp_path);monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m.physical.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR+m.CAP-1))
    def factory():raise AssertionError('no environment before disk gate')
    out=tmp_path/'blocked'
    with pytest.raises(ValueError,match='reserve'):
        m.run('mock','a'*64,out,factory,authorize_physics=True)
    assert not out.exists()


def test_global_cap_no_large_allocation_or_extra_dispatch(tmp_path):
    store=m.physical.Store(tmp_path/'cap')
    store.used=m.CAP-1
    with pytest.raises(ValueError,match='cap'):store.put('too_large',b'ab')
    assert not (store.path/'too_large').exists()


@pytest.mark.parametrize('kind',['truncated','concatenated','junk'])
def test_partial_or_extra_gzip_rejected(tmp_path,kind):
    raw=gzip.compress(b'{"kind":"mock"}\n')
    if kind=='truncated':raw=raw[:-5]
    elif kind=='concatenated':raw+=raw
    else:raw+=b'extra'
    path=tmp_path/'journal.gz';path.write_bytes(raw)
    with pytest.raises(ValueError,match='gzip'):m.journal(path)


def actual_arguments():
    return dict(sampling_prepared=str(m.ROOT/'results/p1_matched_sampling_cuda_20261027_v1_prepared01'),
        sampling_prepare_sha='1c4ee801b9e67b2529e23c51fae94ce0c7dd421290c8fd3edc2de6786f2ab0b3',
        sampling_report=str(m.ROOT/'results/p1_matched_sampling_cuda_20261027_v1_cuda_main01/report.json'),
        sampling_report_sha='cfd13ca478603c90c56b81c70f0ed161ef56d6a8de87da4c0e317388bd138e4e')


@pytest.fixture(scope='module')
def actual():
    args=actual_arguments()
    if not Path(args['sampling_report']).is_file():pytest.skip('actual saved evidence absent')
    return m.collect(**args)


def test_actual_saved_four_selections_exact_source_and_model_only(actual):
    plans=actual['plans'];assert len(plans)==4
    assert len({p['root_sha256'] for p in plans})==2
    assert [p['arm'] for p in plans]==['cem','iid_random']*2
    assert actual['sampling_cost']['predict']==dict(attempted=504,completed=504)
    for p in plans:
        assert p['commands']==p['prediction']['commands']
        assert p['prediction_sha256']==m.source.digest(p['prediction'])
        assert p['source_metadata']['root_sha256']==p['root_sha256']
        assert m.sha(p['raw_source_file'])==p['raw_source_sha256']


@pytest.mark.parametrize('defect',['commands','winner','cost','seed'])
def test_actual_saved_selection_rejects_tamper(actual,defect):
    plan=actual['plans'][0];saved=m.read(plan['origin'])
    top=m.read(actual['arguments']['sampling_report'])
    row=next(x for x in top['arms'] if x['root_sha256']==plan['root_sha256'] and x['method']==plan['arm'])
    events=m.journal(Path(plan['origin']).parent/row['journal_file'])
    protocol=m.sampling.verify(actual['arguments']['sampling_prepared'],actual['arguments']['sampling_prepare_sha'])
    task=next(x['task'] for x in protocol['inputs']['tasks'] if x['root_sha256']==plan['root_sha256'])
    seed=saved['seed']
    if defect=='commands':saved['witness']['commands'][0][0]+=.01
    elif defect=='winner':saved['winner']['index']+=1
    elif defect=='cost':saved['resources']['sampling']['predict']['completed']-=1
    else:saved['seed']+=1
    with pytest.raises(ValueError):m.selection(task,plan['arm'],saved,events,seed)


def test_new_closure_prepare_reverify_drift(tmp_path,monkeypatch):
    sentinel=tmp_path/'source.py';sentinel.write_text('# immutable mock')
    data=dict(arguments={})
    monkeypatch.setattr(m,'collect',lambda **kw: data)
    monkeypatch.setattr(m,'own_sources',lambda:{str(sentinel):m.sha(sentinel)})
    # Supply binding surface needed by verify; no real sourcepack/model calls.
    data['bindings']={}
    out=tmp_path/'prepared';m.prepare({},out);h=m.sha(out/'prepare.json')
    assert m.verify(out,h)['schema']==m.NAME+'.prepare'
    sentinel.write_text('# changed mock')
    with pytest.raises(ValueError,match='closure'):m.verify(out,h)
