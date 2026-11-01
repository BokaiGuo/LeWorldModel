"""RAM metadata only; never checkpoints, reserved roots, model or physics."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p4_fresh_schema_projection_20261028_v1 as m


def raw(v):return json.dumps(v,allow_nan=False).encode()


def conditional():
    slots=[]
    for i in range(2):
        lo=100+4*i;proposals=[]
        for j in range(2):
            attempts=[dict(candidate_id=i,proposal_index=j,attempt_index=k,random_reset_seed=lo+2*j+k,
                           placement_reset_seed=0,state=[float(i),float(j),float(k),2.,0.,0.,0.],root_sha256='a'*64,
                           selected=False,status='failed',goal=['DO_NOT_DECODE_GOAL'],cost='DO_NOT_DECODE_COST') for k in range(2)]
            proposals.append(dict(proposal_index=j,attempts=attempts))
        slots.append(dict(slot=dict(candidate_id=i,root_seed=50+i,reset_seed_span=[lo,lo+4]),proposals=proposals))
    return dict(schema='p1_conditional_procedural_source_20261027_v1.sourcepack',slots=slots)


def observed():
    return dict(schema='p1_observed_history_source_20261027_v1.sourcepack',rows=[dict(candidate_id=i,
        random_reset_seed=100+4*i,placement_reset_seed=0,last_state=[999.]*7,placement_requested_state=[888.]*7,
        metadata=dict(candidate_id=i,root_sha256='a'*64,parent_root_sha256='a'*64,
                      parent_source=dict(root_seed=50+i,reset_seed_span=[100+4*i,104+4*i])),
        checkpoints=[dict(file=f'checkpoint_{i}_{j}.npz',sha256='b'*64,rgb_sha256='c'*64,public_step=j) for j in (0,5,10)]) for i in range(2)])


def capture():
    files=[]
    for i in range(618):
        refs=[dict(parent=parent,pointer=f'/exclusions/unknown_sources/{i}/source',
                   original_reason='historical_reset_attempt_bound_not_attested' if i<8 else 'old_parse_failure',
                   historical_sha256=None) for parent in ('moving_prepare','reachable_prepare')] if i<480 else []
        files.append(dict(path=f'/RAM/history/{i}.json',sha256='d'*64,inherited_refs=refs,blockers=['KEEP_ORIGINAL_UNKNOWN']))
    return dict(schema='root_exclusion_authority_resolution_20261027_v1.resolution_request',history_file_requests=files,
                summary={'inherited_reset_bound_unattested_refs':16},authority_requests=[dict(family='original_four',status='blocked')])


def payloads():return dict(conditional=raw(conditional()),observed=raw(observed()),capture=raw(capture()))


def test_all_eight_attempts_even_failed_unselected_and_all_original_unknowns():
    r=m.build(payloads());c=r['conditional']
    assert (c['slot_denominator'],c['proposal_denominator'],c['attempt_denominator'])==(2,4,8)
    assert len(c['attempts'])==8 and [a['reset_seed'] for a in c['attempts']]==list(range(100,108))
    assert all(a['root_state'] is not None and not a['identity_complete'] for a in c['attempts'])
    assert len(r['inherited']['sources'])==618 and r['inherited']['inherited_unknown_occurrences']==960
    assert sum(len(s['inherited_refs']) for s in r['inherited']['sources'])==960
    assert r['inherited']['sources']==[{k:f[k] for k in ('path','sha256','inherited_refs','blockers')} for f in capture()['history_file_requests']]
    assert r['inherited']['original_unknowns_discharged']==0 and r['inherited']['reset_bound_unattested_refs']==16
    assert not r['source_run_preconditions_met'] and not r['freshness_certified'] and not r['physical_release']


def test_observed_own_state_unknown_despite_matching_parent_hash_and_last_state():
    r=m.observed(raw(observed()))
    assert len(r['rows'])==2
    for x in r['rows']:
        assert x['root_state'] is x['root_state_sha256'] is x['physical_pose_key'] is x['root_seed'] is None
        assert x['root_snapshot_sha256'] is x['history_pixels_sha256'] is x['raw_history_sha256'] is None
        assert len(x['checkpoints'])==3 and x['checkpoint_payloads_opened'] is False and x['maps_to']==[]


def test_projection_never_decodes_goal_status_selected_cost_or_last_state(monkeypatch):
    original=m.old.Projection.string
    def string(self,decode):
        value=original(self,decode)
        if decode:assert value not in ('DO_NOT_DECODE_GOAL','DO_NOT_DECODE_COST')
        return value
    monkeypatch.setattr(m.old.Projection,'string',string)
    assert len(m.conditional(raw(conditional()))['attempts'])==8
    p=m.project(raw(observed()),'observed')
    assert not any('last_state' in k or 'placement_requested_state' in k for k in p.leaves)


@pytest.mark.parametrize('change',['slot','proposal','attempt','reset','placement','state_short','state_bool','state_bad','hash','index','schema'])
def test_conditional_schema_mutations_fail_closed(change):
    v=conditional();a=v['slots'][0]['proposals'][0]['attempts'][0]
    if change=='slot':v['slots'].pop()
    elif change=='proposal':v['slots'][0]['proposals'].pop()
    elif change=='attempt':v['slots'][0]['proposals'][0]['attempts'].pop()
    elif change=='reset':a['random_reset_seed']+=1
    elif change=='placement':a['placement_reset_seed']=1
    elif change=='state_short':a['state'].pop()
    elif change=='state_bool':a['state'][0]=True
    elif change=='state_bad':a['state']='not_state'
    elif change=='hash':a['root_sha256']='bad'
    elif change=='index':a['attempt_index']=True
    else:v['schema']='wrong'
    with pytest.raises((ValueError,KeyError)):m.conditional(raw(v))


def test_missing_actual_attempt_state_is_unknown_not_zero_or_drop():
    v=conditional();del v['slots'][0]['proposals'][0]['attempts'][0]['state']
    r=m.conditional(raw(v))
    assert len(r['attempts'])==8 and r['attempts'][0]['root_state'] is None
    assert 'root_state' in r['attempts'][0]['missing']


@pytest.mark.parametrize('change',['drop_source','drop_ref','duplicate_ref','reason','drop_row','checkpoint_step','checkpoint_hash','row_index'])
def test_denominator_and_observed_metadata_mutations(change):
    v=capture();o=observed()
    if change=='drop_source':v['history_file_requests'].pop()
    elif change=='drop_ref':v['history_file_requests'][0]['inherited_refs'].pop()
    elif change=='duplicate_ref':v['history_file_requests'][0]['inherited_refs'][1]=deepcopy(v['history_file_requests'][0]['inherited_refs'][0])
    elif change=='reason':v['history_file_requests'][0]['inherited_refs'][0]['original_reason']='fake_cleared'
    elif change=='drop_row':o['rows'].pop()
    elif change=='checkpoint_step':o['rows'][0]['checkpoints'][0]['public_step']=10
    elif change=='checkpoint_hash':o['rows'][0]['checkpoints'][0]['sha256']='bad'
    else:o['rows'][0]['metadata']['candidate_id']=1
    with pytest.raises((ValueError,KeyError)):
        m.inherited(raw(v));m.observed(raw(o))


def test_observed_partial_checkpoint_does_not_remove_row_or_imply_actual_state():
    v=observed();v['rows'][0]['checkpoints']=[]
    r=m.observed(raw(v));assert len(r['rows'])==2 and r['rows'][0]['root_state'] is None


def test_state_hash_matches_original_float64_bits_including_signed_zero():
    v=conditional();s=v['slots'][0]['proposals'][0]['attempts'][0]['state'];s[0]=-0.
    r=m.conditional(raw(v))['attempts'][0]
    assert r['root_state_sha256']==m.sha(np.asarray(s,dtype='<f8').tobytes())


def test_exact_input_set_and_byte_caps():
    p=payloads();p['checkpoint']=b'NOT_ALLOWED'
    with pytest.raises(ValueError):m.build(p)
    with pytest.raises(ValueError):m.conditional(b' '* (m.CAPS['conditional']+1))


def test_duplicate_projected_keys_and_nonfinite_state_refused():
    with pytest.raises(ValueError):m.conditional(b'{"schema":"x","schema":"y"}')
    v=conditional();data=raw(v).replace(b'0.0',b'NaN',1)
    with pytest.raises(ValueError):m.conditional(data)
