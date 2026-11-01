from types import SimpleNamespace
import pytest
import numpy as np
from experiments.spectral_reachability import p1_direct_cuda_calibration_main_20261027_v2 as m


@pytest.mark.parametrize('method',m.METHODS)
def test_exact_config_budget(method):
    c=m.config(method);budget=m.screen.budget(c)
    assert budget['predict']==30 and budget['action_encoder']==30
    assert budget['jacobian']==3 and budget['vjp_directions']==576
    assert c['B1']==1 and c['B2']==3 and c['extra_line_search_trials']==0
    assert np.array_equal(c['initial_logits'],np.zeros((1,2,10)))


def test_no_unrecognized_method():
    with pytest.raises(ValueError):m.config('fake_GN')


def test_write_exclusive_small_json(tmp_path):
    path=tmp_path/'x.json';m.write(path,dict(test=True))
    with pytest.raises(FileExistsError):m.write(path,dict(test=False))


def test_no_sourcepack_no_neural_import(tmp_path):
    with pytest.raises(ValueError):m.material(tmp_path,'0'*64)


def test_space_guard_precedes_claim_and_model(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:{})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=200*1024**2))
    with pytest.raises(ValueError):m.run(tmp_path/'prepared','fake',tmp_path/'out')
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope='module')
def real_source():
    from copy import deepcopy
    pack=m.ROOT/'results/p1_conditional_calibration2_main_20261027_v1'
    r=m.io.read(pack/'sourcepack.json');events=m.source.read_journal(pack/'journal.jsonl')
    return r,r['plan'],events[0]['exclusions'],events,pack


def test_actual_preflight_without_task_ids(real_source):
    assert m.preflight_source(*real_source)['placeholder_task_ids'] is False


@pytest.mark.parametrize('damage',['terminal','slot','coordinates','selected','raw_state'])
def test_corrupt_source_rejected_before_neural(real_source,damage):
    from copy import deepcopy
    r,p,e,events,pack=deepcopy(real_source)
    if damage=='terminal':events[-1]['kind']='not_finished'
    elif damage=='slot':r['slots'][0]['slot']['candidate_id']=99
    elif damage=='coordinates':
        next(x for x in events if x['kind']=='operation_completed')['random_reset_seed']+=1
    elif damage=='selected':
        next(a for q in r['slots'][0]['proposals'] for a in q['attempts'] if a['selected'])['selected']=False
    else:r['slots'][0]['proposals'][0]['attempts'][0]['state'][0]+=1
    with pytest.raises(ValueError):m.preflight_source(r,p,e,events,pack)


def test_no_old_collect_or_report_dependency():
    import inspect
    code=inspect.getsource(m.material)
    assert 'io.collect(' not in code and 'scheduler_20261027' not in code
    metadata,norm=m.native_metadata()
    assert norm['training_provenance_verified'] is False
    assert metadata['policy'].endswith('/pusht/lewm')

