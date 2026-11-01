"""Saved-only v3 hardening; no producer/model/native execution."""
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v3 as m


@pytest.mark.parametrize('position',[0,1,2,3])
def test_unknown_event_rejected_anywhere_in_full_unprojected_stream(position):
    events=[dict(kind='run_started'),dict(kind='replay20_ready'),dict(kind='run_finished')]
    events.insert(position,dict(kind='unreleased_hidden_callback'))
    with pytest.raises(ValueError,match='unreleased event'):m.validate_eventset(events)


def test_other_producer_operation_completed_not_accepted_as_v2_event():
    with pytest.raises(ValueError,match='unreleased event'):
        m.validate_eventset([dict(kind='operation_completed')])


def test_all_reachable_success_and_failure_event_kinds_are_declared():
    assert m.validate_eventset([dict(kind=k) for k in sorted(m.ALLOWED_EVENTS)])=={k:1 for k in m.ALLOWED_EVENTS}
    assert {'placement_recipe','goal_attached','invalid_observation','alternate_encode_failed','operation_failed'}<=m.ALLOWED_EVENTS


def image_fixture(tmp_path):
    s=m.v2.Store(tmp_path/(m.v2.NAME+'_v3_fixture'))
    rgb=np.arange(224*224*3,dtype=np.uint8).reshape(224,224,3)
    pre=m.pixels.preprocess_pixels(rgb)
    image=dict(index=0,completed=True,pixels=s.array('pixels',rgb),preprocessed=s.array('pre',pre),
        modules=[dict(resource='encoder',inputs=[s.array('encoder_input',pre.reshape(1,3,224,224))])])
    return s,rgb,pre,image


def test_original224_preprocess_exactly_matches_saved_bridge(tmp_path):
    s,rgb,pre,image=image_fixture(tmp_path)
    assert m.validate_preprocessing(s.path,[image])==dict(checked=1,partial_without_preprocessed=0)


def test_coordinated_preprocess_encoder_mutation_no_longer_passes(tmp_path):
    s,rgb,pre,image=image_fixture(tmp_path)
    wrong=pre.copy();wrong[0,0,0,0,0]=np.nextafter(wrong[0,0,0,0,0],np.float32(np.inf))
    image['preprocessed']=s.array('wrong_pre',wrong)
    image['modules'][0]['inputs'][0]=s.array('wrong_encoder',wrong.reshape(1,3,224,224))
    assert m.v2.exact_array(m.v2.load_array(s.path,image['preprocessed']).reshape(1,3,224,224),
                            m.v2.load_array(s.path,image['modules'][0]['inputs'][0]))
    with pytest.raises(ValueError,match='pixels -> released preprocessing'):
        m.validate_preprocessing(s.path,[image])


def test_rehashed_saved_pixel_mutation_rejected(tmp_path):
    s,rgb,pre,image=image_fixture(tmp_path)
    changed=rgb.copy();changed[0,0,0]^=np.uint8(1)
    image['pixels']=s.array('different_pixels',changed)
    with pytest.raises(ValueError,match='pixels -> released preprocessing'):
        m.validate_preprocessing(s.path,[image])


def test_partial_image_with_saved_preprocessing_is_also_checked(tmp_path):
    s,rgb,pre,image=image_fixture(tmp_path);image['completed']=False
    image['preprocessed']=s.array('wrong_partial',np.zeros_like(pre))
    with pytest.raises(ValueError):m.validate_preprocessing(s.path,[image])


def test_missing_preprocessing_allowed_only_for_incomplete_attempt(tmp_path):
    s,rgb,pre,image=image_fixture(tmp_path);del image['preprocessed'];image['completed']=False
    assert m.validate_preprocessing(s.path,[image])==dict(checked=0,partial_without_preprocessed=1)
    image['completed']=True
    with pytest.raises(ValueError):m.validate_preprocessing(s.path,[image])


def test_wrapper_never_bypasses_original_consumer(tmp_path,monkeypatch):
    s=m.v2.Store(tmp_path/(m.v2.NAME+'_v3_fixture'))
    s.emit('run_started');s.emit('run_finished')
    report=dict(artifacts=s.census(),images=[]);s.put('report.json',report,final=True)
    calls=[]
    def base(output,report_sha):
        calls.append((output,report_sha));return dict(status='partial_saved_audit',fixed_pairs=80,P1_admission=False)
    monkeypatch.setattr(m.v2,'consume',base)
    result=m.consume(s.path,m.v2.sha(s.path/'report.json'))
    assert len(calls)==1 and result['v2_consume']['fixed_pairs']==80
    assert result['P1_admission'] is False and result['schema']==m.NAME+'.consume'


def test_wrapper_rejects_unknown_even_when_old_consumer_accepts(tmp_path,monkeypatch):
    s=m.v2.Store(tmp_path/(m.v2.NAME+'_v3_fixture'))
    s.emit('run_started');s.emit('replay20_ready');s.emit('unreleased_hidden_callback');s.emit('run_finished')
    s.put('report.json',dict(artifacts=s.census(),images=[]),final=True)
    monkeypatch.setattr(m.v2,'consume',lambda *args:dict(status='partial_saved_audit',P1_admission=False))
    with pytest.raises(ValueError,match='unreleased event'):m.consume(s.path,m.v2.sha(s.path/'report.json'))


def test_wrapper_rejects_pixel_bridge_mutation_even_if_original_consumer_accepts(tmp_path,monkeypatch):
    s,rgb,pre,image=image_fixture(tmp_path)
    wrong=pre.copy();wrong[0,0,0,0,0]=np.nextafter(wrong[0,0,0,0,0],np.float32(np.inf))
    image['preprocessed']=s.array('wrong_pre',wrong)
    image['modules'][0]['inputs'][0]=s.array('wrong_encoder',wrong.reshape(1,3,224,224))
    s.emit('run_started');s.emit('run_finished')
    s.put('report.json',dict(artifacts=s.census(),images=[image]),final=True)
    monkeypatch.setattr(m.v2,'consume',lambda *args:dict(status='partial_saved_audit',P1_admission=False))
    with pytest.raises(ValueError,match='pixels -> released preprocessing'):
        m.consume(s.path,m.v2.sha(s.path/'report.json'))


def test_original_saved_native40_event_kinds_are_covered_without_physics():
    path=m.ROOT/'results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02/journal.jsonl'
    counts=m.validate_eventset(m.v2.read_events(path))
    assert {'placement_recipe','goal_attached','fresh_observation','raw_return','teacher_history_verified'}<=counts.keys()


def actual_terminal_fixture():
    """Real saved schema/values, no environment/model construction."""
    from copy import deepcopy
    d=m.ROOT/'results/agent_p1_prospective_action_effect_20261028_v2_MAIN01'
    r=m.v2.teacher.read(d/'report.json','2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c',m.v2.phys.REPORT_CAP)
    row=deepcopy(r['rows'][0]);body=deepcopy(row['terminal_triplet'])
    events=[]
    for i,e in enumerate(body):
        full=dict(e,index=100+i,previous_sha256='a'*64);full['event_sha256']=m.v2.digest(full);events.append(full)
    return dict(rows=[row]),events


def test_actual_shape_envelope_projection_leaves_all_payload_fields_exact():
    r,events=actual_terminal_fixture()
    assert m.v2.canonical(events)!=m.v2.canonical(r['rows'][0]['terminal_triplet'])  # Original v2 failure.
    mapping=m.terminal_projection(r,events)
    assert len(mapping)==1
    target,index=mapping[m.v2.canonical(events)]
    assert target==m.v2.canonical(r['rows'][0]['terminal_triplet']) and index==0


@pytest.mark.parametrize('field',['state','snapshot','pixel_sha256','step'])
def test_actual_shape_projection_rejects_rehashed_payload_mutation(field):
    from copy import deepcopy
    r,events=actual_terminal_fixture();events=deepcopy(events)
    receipt=events[-1]['receipt']
    if field=='state':receipt[field][0]+=1
    elif field=='snapshot':receipt[field]['state'][0]+=1
    elif field=='pixel_sha256':receipt[field]='0'*64
    else:receipt[field]=9
    events[-1]['event_sha256']=m.v2.digest({k:v for k,v in events[-1].items() if k!='event_sha256'})
    with pytest.raises(ValueError):m.terminal_projection(r,events)


def test_projection_does_not_accept_extra_metadata_or_invalid_hash():
    r,events=actual_terminal_fixture();events[-1]['unexpected_metadata']='do not drop'
    with pytest.raises(ValueError):m.terminal_projection(r,events)
    r,events=actual_terminal_fixture();events[-1]['event_sha256']='0'*64
    with pytest.raises(ValueError):m.terminal_projection(r,events)


def test_consumer_projection_is_local_and_applied_exactly_once(monkeypatch):
    r,events=actual_terminal_fixture();mapping=m.terminal_projection(r,events)
    original=m.v2.canonical
    # FunctionType clone resolves canonical locally; never monkeypatch v2.canonical.
    def frozen_consumer(output,report_sha):
        assert canonical(events)==canonical(r['rows'][0]['terminal_triplet'])
        return dict(status='partial_saved_audit',P1_admission=False)
    monkeypatch.setattr(m.v2,'consume',frozen_consumer)
    base,usage=m.consume_with_projection('unused','unused',mapping)
    assert usage=={0:1} and base['P1_admission'] is False
    assert m.v2.canonical is original
    assert original(events)!=original(r['rows'][0]['terminal_triplet'])
