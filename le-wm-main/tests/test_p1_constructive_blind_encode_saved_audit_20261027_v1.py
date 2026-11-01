"""Pure NumPy/stdlib tests; actual saved tests do not execute any model."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_constructive_blind_encode_saved_audit_20261027_v1 as a

REPORT = a.ROOT/'results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/report.json'
REPORT_SHA = '684456d6d3c45b99f20db354d2a0fb7b810bb61a8ebc8185816aa3d87ff40e91'


def test_serialization_is_independent_core_format():
    v = dict(b=[1.,2.],a=False)
    assert a.task_hash(v) == a.hash_bytes(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    assert a.task_hash(v) != a.hash_bytes(a.canonical(v))


@pytest.mark.parametrize('raw', [b'{"a":0,"a":1}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_json_rejects_duplicate_and_nonfinite(raw):
    with pytest.raises(ValueError):
        a.parse(raw)


def fixture_task():
    norm = dict(mean=np.asarray([-.0087,.0068],np.float32).tolist(),
                std=np.asarray([.2019,.2002],np.float32).tolist())
    packet = dict(rgb_history=np.zeros((3,224,224,3),np.uint8),goal_rgb=np.ones((224,224,3),np.uint8),
                  raw_action_history=np.zeros((10,2),np.float32))
    zs = [np.full(192,i,np.float32) for i in range(4)]
    return dict(arguments=dict(whitelist_sha='a'*64)), dict(source_slot=0,blind_sha256='b'*64), packet, zs, norm


def test_reconstruction_distinct_chronology_goal_identity_Q():
    p,slot,packet,zs,norm = fixture_task()
    tc,t,ident,pixels = a.reconstruct(p,'c'*64,slot,packet,zs,norm)
    assert t['root_context']['latent_history'] == [z.tolist() for z in zs[:3]]
    assert t['goal'] == zs[3].tolist()
    assert np.array_equal(t['projection'],np.eye(192))
    assert len(pixels) == 4 and pixels[0] != pixels[3]
    assert tc['with_target'] is False and ident == a.task_hash(t)
    assert np.asarray(t['root_context']['action_history']).shape == (2,10)
    assert all(v != 0 for v in t['root_context']['action_history'][0])


def test_time_major_action_alignment_not_padding():
    p,slot,packet,zs,norm = fixture_task()
    # Deliberately distinct rows for a generic arithmetic chronology regression.
    packet['raw_action_history'] = np.arange(20,dtype=np.float32).reshape(10,2)/100
    _,t,_,_ = a.reconstruct(p,'c'*64,slot,packet,zs,norm)
    want = ((packet['raw_action_history']-np.asarray(norm['mean'],np.float32))/np.asarray(norm['std'],np.float32)).reshape(2,10)
    assert np.array_equal(t['root_context']['action_history'],want)
    assert not np.array_equal(want[0],want[1])


@pytest.mark.parametrize('bad', ['float64','shape','nan','count'])
def test_reconstruction_rejects_invalid_latents(bad):
    p,slot,packet,zs,norm = fixture_task()
    if bad == 'float64': zs[0] = zs[0].astype(float)
    if bad == 'shape': zs[0] = zs[0][None,:]
    if bad == 'nan': zs[0][0] = np.nan
    if bad == 'count': zs.pop()
    with pytest.raises(ValueError):
        a.reconstruct(p,'c'*64,slot,packet,zs,norm)


@pytest.fixture
def saved():
    if not REPORT.exists():
        pytest.skip('MAIN saved receipt not installed; arithmetic tests still available')
    return a.read_json(REPORT,REPORT_SHA)


def test_actual_saved_complete(saved):
    result = a.audit(REPORT,REPORT_SHA)
    assert result['status'] == 'saved_consistency_verified'
    assert result['event_count'] == 50
    assert result['saved_output_bytes'] < a.LIMIT
    assert result['new_model_calls'] == result['new_physics'] == 0
    assert result['P1_admission'] is False


@pytest.mark.parametrize('mutation', ['partial','errors','count','extra_artifact','slot','task_id','model','predict','target'])
def test_saved_report_corruption_rejected(saved,monkeypatch,mutation):
    r = deepcopy(saved)
    if mutation == 'partial': r['status'] = 'partial_or_failed'
    if mutation == 'errors': r['errors'] = [dict(message='failed')]
    if mutation == 'count': r['resources']['encoder']['attempted'] = 7
    if mutation == 'extra_artifact': r['artifact_sha256']['certificate.json'] = '0'*64
    if mutation == 'slot': r['rows'][0]['order'] = 1
    if mutation == 'task_id': r['rows'][0]['task_id'] = '0'*64
    if mutation == 'model': r['model_hash_after'] = '0'*64
    if mutation == 'predict': r['new_predict'] = 1
    if mutation == 'target': r['with_target'] = True
    original = a.read_json
    monkeypatch.setattr(a,'read_json',lambda path,expected=None: r if Path(path) == REPORT else original(path,expected))
    with pytest.raises(ValueError): a.audit(REPORT,REPORT_SHA)


@pytest.mark.parametrize('mutation', ['history','zero_padding','goal','Q','contract','native_root','task_id'])
def test_task_content_not_just_shapes(saved,monkeypatch,mutation):
    original = a.read_json
    def altered(path,expected=None):
        v = original(path,expected)
        if Path(path).name != 'task_00.json.gz': return v
        if mutation == 'history': v['task']['root_context']['latent_history'][1][0] += 1
        if mutation == 'zero_padding': v['task']['root_context']['action_history'] = np.zeros((2,10)).tolist()
        if mutation == 'goal': v['task']['goal'][0] += 1
        if mutation == 'Q': v['task']['projection'][0][0] = 0
        if mutation == 'contract': v['task_contract']['with_target'] = True
        if mutation == 'native_root': v['native_root_sha256'] = '0'*64
        if mutation == 'task_id': v['task_id'] = '0'*64
        return v
    monkeypatch.setattr(a,'read_json',altered)
    with pytest.raises(ValueError): a.audit(REPORT,REPORT_SHA)


@pytest.mark.parametrize('mutation',['remove','duplicate','reorder','forbidden'])
def test_event_census_all_calls_required(saved,monkeypatch,mutation):
    original = a.read_bytes
    def altered(path,expected=None):
        raw = original(path,expected)
        if Path(path).name != 'image.events.jsonl': return raw
        events = [a.parse(line) for line in raw.splitlines()]
        if mutation == 'remove': events.pop(2)
        if mutation == 'duplicate': events.insert(2,deepcopy(events[2]))
        if mutation == 'reorder': events[1],events[2] = events[2],events[1]
        if mutation == 'forbidden': events.append(dict(kind='forbidden',name='predict'))
        return b''.join(a.canonical(e) for e in events)
    monkeypatch.setattr(a,'read_bytes',altered)
    with pytest.raises(ValueError): a.audit(REPORT,REPORT_SHA)


def test_wrong_external_report_sha(saved):
    with pytest.raises(ValueError,match='SHA'): a.audit(REPORT,'0'*64)


def test_no_producer_or_model_import():
    import ast
    tree = ast.parse(Path(a.__file__).read_text())
    imports = [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
    assert not any(any(s in name for s in ('torch','producer','stable_worldmodel','task_admission','encode_cuda')) for name in imports)
