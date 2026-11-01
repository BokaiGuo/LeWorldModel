from pathlib import Path
import runpy
import pytest
from experiments.spectral_reachability import p2_h15_sra_host_candidate_20261029_v1 as m


@pytest.fixture(scope='module')
def native_fixture():
    f=runpy.run_path(str(Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')))
    return f, f['material']()['tasks'][0]


def test_real_h15_ram_native_40_candidate_and_saved_consumer(tmp_path,native_fixture):
    fixture,task=native_fixture
    result=m.run_fixture(tmp_path/'case',task,lambda t,s:fixture['NumpyNative'](t,s))
    checked=m.consume(tmp_path/'case',result['report_sha256'],task,permit_fixture=True)
    assert checked['candidate_count']==40 and checked['resources']['attempted']['predict_samples']==132
    assert checked['resources']['attempted']['VJP_directions']==192
    assert not checked['ready_to_execute'] and not checked['physical_release']
    with pytest.raises(ValueError):m.consume(tmp_path/'case',result['report_sha256'],task)


def test_failure_record_does_not_create_complete_plan(tmp_path,native_fixture):
    fixture,task=native_fixture
    result=m.run_fixture(tmp_path/'case',task,lambda t,s:fixture['NumpyNative'](t,s,fail=True))
    assert result['status']=='failed_partial'
    with pytest.raises(ValueError,match='partial'):
        m.consume(tmp_path/'case',result['report_sha256'],task,permit_fixture=True)


def test_fixed_budget_and_authorization():
    assert m.caps()['predict_samples']==132
    assert m.caps()['VJP_directions']==192
    with pytest.raises(ValueError,match='not authorized'):
        m.host_backend(None,None,None,None,lambda:None)


def test_prepared_case_binding_rejects_wrong_root_and_context(native_fixture):
    _,task=native_fixture
    case=dict(task_id=task['binding']['task_id'],task_sha256=task['binding']['task_sha256'],
        native_root={'root_sha256':task['binding']['root_sha256']},root_context=task['root_context'],
        goal=task['goal'],q=task['projection'],action_normalizer=dict(mean=task['action_mean'],std=task['action_std']),
        model_sha256=task['binding']['model_sha256'])
    m.validate_case(case,task)
    bad=dict(case,native_root={'root_sha256':'f'*64})
    with pytest.raises(ValueError,match='source'):m.validate_case(bad,task)


def test_saved_journal_tampering_rejected(tmp_path,native_fixture):
    fixture,task=native_fixture
    result=m.run_fixture(tmp_path/'case',task,lambda t,s:fixture['NumpyNative'](t,s))
    path=tmp_path/'case'/'events.jsonl'
    with path.open('ab') as f: f.write(b'{}\n')
    with pytest.raises(ValueError,match='journal SHA'):
        m.consume(tmp_path/'case',result['report_sha256'],task,permit_fixture=True)
