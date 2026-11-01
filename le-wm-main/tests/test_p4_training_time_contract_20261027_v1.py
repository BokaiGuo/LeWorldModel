"""Metadata-only temporal tests; never construct a model or an environment."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p4_training_time_contract_20261027_v1 as m


def fixture(arm='actual'):
    past=np.arange(20,dtype=float).reshape(2,5,2)/100
    commands=np.arange(40,dtype=float).reshape(20,2)/100
    p=dict(prefix=commands[:5].tolist(),between_raw=past.tolist())
    hashes={t:str(t) for t in (-10,-5,0,*range(1,21))}
    events=[]
    def emit(kind,**fields):
        events.append(dict(kind=kind,arm=arm,event_index=len(events),**fields))
    def ctx(b,offset):
        return dict(latent_history=[([t+offset]*192) for t in (b-10,b-5,b)],
            action_history=np.zeros((2,10)).tolist())
    emit('initial',pixel_sha256=hashes[0])
    for step in range(1,21):
        emit('step_attempt',primitive_step=step,command=commands[step-1].tolist())
        emit('step_returned',primitive_step=step)
        emit('observation',primitive_step=step,pixel_sha256=hashes[step])
        if step not in (5,10,15):
            continue
        b=step
        raw=np.concatenate([past,commands.reshape(4,5,2)])
        between=raw[b//5:b//5+2].tolist()
        emit('boundary_begin',boundary=b,between_raw=between,
            frames_pixel_sha256=[hashes[t] for t in (b-10,b-5,b)])
        observed,predicted=ctx(b,0),ctx(b,100)
        chosen=observed if arm=='actual' else predicted
        emit('planning_context',boundary=b,observed_context=observed,
            predicted_context=predicted,chosen_context=chosen)
        emit('selection',boundary=b,incoming_between_raw=between,
            commands=commands[b:b+5].tolist(),compiler_plan=dict(context=chosen),selection_sha256=str(b))
        # Real packet_complete is after the following five steps; insert later.
    for b in (5,10,15):
        # Build the correct ordering, including packet completion before next boundary.
        after=next(e for e in events if e['kind']=='observation' and e['primitive_step']==b+5)
        index=events.index(after)+1
        events.insert(index,dict(kind='packet_complete',arm=arm,boundary=b,next_boundary=b+5,
            next_between_raw=raw[b//5+1:b//5+3].tolist(),predicted_context=ctx(b+5,100)))
    for i,e in enumerate(events):
        e['event_index']=i
    return p,events,arm,[hashes[t] for t in (-10,-5,0)]


@pytest.mark.parametrize('arm',['actual','imagined'])
def test_valid_temporal_fixture(arm):
    rows=m.audit_arm(*fixture(arm))
    assert len(rows)==3
    assert [r['predicted_state_step'] for r in rows]==[10,15,20]
    assert rows[0]['rgb_state_steps']==[-5,0,5]
    assert rows[1]['between_action_intervals']==[[0,5],[5,10]]
    assert rows[2]['proposed_action_interval']==[15,20]


@pytest.mark.parametrize('boundary',[5,10,15])
@pytest.mark.parametrize('fault',['rgb','past_packet','new_packet','next_packet','observed_shift','condition','dispatch_order'])
def test_temporal_misalignments_rejected(boundary,fault):
    p,events,arm,roots=deepcopy(fixture())
    find=lambda kind: next(e for e in events if e['kind']==kind and e.get('boundary')==boundary)
    if fault=='rgb':
        find('boundary_begin')['frames_pixel_sha256'][0]='wrong-time'
    elif fault=='past_packet':
        # Corrupt BOTH receipts: cross-check with actual dispatch must still reject.
        find('boundary_begin')['between_raw'][0][0][0]+=1
    elif fault=='new_packet':
        find('selection')['commands'][0][0]+=1
    elif fault=='next_packet':
        find('packet_complete')['next_between_raw'][1][0][0]+=1
    elif fault=='observed_shift':
        if boundary==5:
            find('planning_context')['observed_context']['latent_history'].pop()
        else:
            find('planning_context')['observed_context']['latent_history'][0][0]+=1
    elif fault=='condition':
        c=find('planning_context');c['chosen_context']=c['predicted_context']
    else:
        find('selection')['event_index']=9999
    with pytest.raises(ValueError):
        m.audit_arm(p,events,arm,roots)


@pytest.mark.parametrize('fault',['gap','duplicate_observation','prefix','root','missing_boundary','imagined_recurrence'])
def test_saved_trace_integrity_rejected(fault):
    p,events,arm,roots=deepcopy(fixture())
    if fault=='gap':
        events.remove(next(e for e in events if e['kind']=='step_attempt'))
    elif fault=='duplicate_observation':
        events.append(deepcopy(next(e for e in events if e['kind']=='observation')))
    elif fault=='prefix':
        p['prefix'][0][0]+=1
    elif fault=='root':
        roots[-1]='wrong'
    elif fault=='missing_boundary':
        events.remove(next(e for e in events if e['kind']=='planning_context'))
    else:
        next(e for e in events if e['kind']=='packet_complete')['predicted_context']['latent_history'][2][0]+=1
    with pytest.raises(ValueError):
        m.audit_arm(p,events,arm,roots)


def test_reviewed_actual_training_dataset_encoder_sources_without_torch():
    inputs=m.saved.Inputs()
    result=m.marker_probes(inputs)
    assert result['training']['stopped_before_loss']
    assert result['sample_global_observation_rows']==[99,104,109,114]
    assert result['training']['context']==[[[99],[104],[109]]]
    assert result['training']['target']==[[[104],[109],[114]]]
    assert result['sample_flat_action_tokens'][0]==[1099,2099,1100,2100,1101,2101,1102,2102,1103,2103]
    assert result['action_encoder']['conv_input_shape']==[1,10,3]
    assert result['neural_calls']==0


def test_changed_training_target_source_is_not_executed(tmp_path,monkeypatch):
    original=m.PROBES['train']
    # Use pytest's own file writer for the fixture; production files untouched.
    changed=tmp_path/'train.py'
    changed.write_text(original[0].read_text().replace('emb[:, n_preds:]','emb[:, n_preds+1:]'))
    monkeypatch.setitem(m.PROBES,'train',(changed,*original[1:]))
    with pytest.raises(ValueError,match='unreviewed probe'):
        m.marker_probes(m.saved.Inputs())


def test_actual_static_source_anchors():
    evidence=m.source_contract(m.saved.Inputs())
    assert len(evidence)==17
    assert any('kernel_size=1, stride=1' in e['anchors'] for e in evidence)


@pytest.mark.parametrize('boundary',[0,1,20,True,5.0])
def test_reject_other_temporal_protocols(boundary):
    with pytest.raises(ValueError):
        m.timing(boundary)


def test_bounded_hdf5_witness(tmp_path):
    import h5py
    path=tmp_path/'source.h5'
    with h5py.File(path,'w') as f:
        f['ep_offset']=[0,30];f['ep_len']=[30,30]
        f['state']=np.arange(60*7).reshape(60,7)
        f['action']=np.zeros((60,2))
        f['step_idx']=np.tile(np.arange(30),2)
        f['episode_idx']=np.repeat([0,1],30)
    result=m.dataset_witness(path)
    assert [r['global_start'] for r in result['records']]==[2,32]
    assert result['records'][1]['observation_global_rows']==[32,37,42,47]
    assert result['full_file_sha256'] is None
    assert not result['historical_checkpoint_training_dataset_identity_proven']


def test_completed_saved_p4_temporal_only():
    # Real saved traces, no previous outcome/Q/camera/normalization recomputation.
    p=json.loads((m.RUNTIME.parent/'protocol.json').read_text())
    events=[json.loads(s) for s in (m.RUNTIME.parent/'runtime.events.jsonl').read_text().splitlines()]
    roots=np.load(p['root_rgb'],allow_pickle=False)
    hashes=[hashlib.sha256(f.tobytes()).hexdigest() for f in roots]
    for arm in ('actual','imagined'):
        assert len(m.audit_arm(p,events,arm,hashes))==3


def test_no_model_or_physics_imports():
    import ast
    tree=ast.parse(m.Path(m.__file__).read_text())
    names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(n and n.split('.')[0] in ('torch','gym','gymnasium','pymunk','hydra','lightning','stable_worldmodel') for n in names)


def test_actual_audit_retains_historical_provenance_gap():
    result=m.audit()
    assert len(result['rows'])==6
    assert result['visible_training_contract']['target_relative_rows']==[5,10,15]
    assert not result['original_checkpoint_training_contract_proven']
    assert not result['specific_temporal_misalignment_found']
    assert result['new_neural_calls']==result['new_physics_steps']==0
    assert not result['P1_admission'] and not result['P4_admission']


def test_output_is_exclusive_before_audit(tmp_path):
    with pytest.raises(ValueError,match='exclusive output'):
        m.run(tmp_path/m.TAG)
