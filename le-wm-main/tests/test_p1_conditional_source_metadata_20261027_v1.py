"""Whitelist JSON projection and provenance graph; never decode future values."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_conditional_source_metadata_20261027_v1 as m


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return path


def test_projection_skips_future_values_including_nested_strings(tmp_path,monkeypatch):
    path=write(tmp_path/'report.json',dict(protocol=dict(root_seed=500,root_count=2),
        rows=[dict(root_sha256='a'*64,physical_future={'success':'DO_NOT_DECODE','trajectory':[1,2,3]})]))
    original=m.json.loads
    def guarded(raw,*args,**kwargs):
        assert 'DO_NOT_DECODE' not in raw
        return original(raw,*args,**kwargs)
    monkeypatch.setattr(m.json,'loads',guarded)
    result=m.project_metadata(path)
    assert result['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['leaves'][('protocol','root_seed')]==500
    assert result['leaves'][('rows',0,'root_sha256')]=='a'*64
    assert all('physical_future' not in key for key in result['leaves'])


def test_projection_does_not_promote_optimizer_seed(tmp_path):
    p=write(tmp_path/'args.json',dict(seed=123,root_seed=456,optimizer=dict(seed=789)))
    assert set(m.project_metadata(p)['leaves'])=={('root_seed',)}


def test_graph_resolves_root_identity_inheritance_not_arbitrary_directory_names(tmp_path):
    root=tmp_path/'results'
    origin=write(root/'pusht_origin'/'report.json',dict(protocol=dict(root_seed=300,root_count=1),rows=[dict(root_sha256='a'*64)]))
    sha=hashlib.sha256(origin.read_bytes()).hexdigest()
    write(root/'pusht_derived'/'source_seal'/'args.json',dict(bindings={str(origin):sha}))
    write(root/'pusht_derived'/'report.json',dict(rows=[dict(root_sha256='a'*64,terminal_success='FORBIDDEN')]))
    (root/'pusht_unrelated_empty_name_only').mkdir()
    result=m.resolve_history(tmp_path)
    assert len(result['root_hashes'])==1 and result['seed_families'][0]['seed']==300
    assert any(x['kind']=='root_set_inherited' for x in result['evidence'])
    assert not any('unrelated_empty' in x['source'] for x in result['unknown_sources'])
    assert result['confirmation_authority']['status']=='unknown'
    assert any(x['reason']=='historical_reset_attempt_bound_not_attested' for x in result['unknown_sources'])


def test_unknown_means_actual_unresolved_root_not_every_empty_directory(tmp_path):
    write(tmp_path/'results'/'pusht_no_seed'/'report.json',dict(rows=[dict(root_sha256='b'*64)]))
    result=m.resolve_history(tmp_path)
    assert any(x['reason']=='root_has_no_seed_provenance' for x in result['unknown_sources'])
    assert result['root_hashes']==['b'*64]


def test_bad_sha_reference_is_not_authority(tmp_path):
    origin=write(tmp_path/'results'/'pusht_origin'/'report.json',dict(protocol=dict(root_seed=10,root_count=1),rows=[dict(root_sha256='a'*64)]))
    write(tmp_path/'results'/'pusht_child'/'source_seal'/'args.json',dict(bindings={str(origin):'0'*64}))
    result=m.resolve_history(tmp_path)
    assert any(x['reason']=='input_binding_hash_mismatch' for x in result['unknown_sources'])


def test_actual_metadata_census_protects_known_seeds_and_does_not_self_certify():
    result=m.resolve_history(m.PROJECT)
    assert {202610241,202710241,202753000,202753010}<={x['seed'] for x in result['seed_families']}
    assert len(result['root_hashes'])>=20
    assert result['confirmation_authority']['status']=='unknown'
    assert not any('opaque_result_directory' in x['reason'] for x in result['unknown_sources'])
    assert result['physical_future_values_decoded']==0


def test_new_generator_sourcepack_and_request_reserve_all_slots_without_outcomes(tmp_path,monkeypatch):
    plan=dict(schema='p1_conditional_procedural_source_20261027_v1.plan',slots=[
        dict(root_seed=202761010,reset_seed_span=[202781010,202781014]),
        dict(root_seed=202761011,reset_seed_span=[202781014,202781018])])
    write(tmp_path/'results'/'p1_conditional_calibration2'/'sourcepack.json',dict(
        plan=plan,slots=[dict(metadata=dict(root_sha256='c'*64))],outcomes='NEVER_DECODE'))
    write(tmp_path/'results'/'arbitrary_request.json',dict(plan=plan,outcomes='NEVER_DECODE'))
    original=m.json.loads
    def guarded(raw,*args,**kwargs):
        assert 'NEVER_DECODE' not in raw
        return original(raw,*args,**kwargs)
    monkeypatch.setattr(m.json,'loads',guarded)
    result=m.resolve_history(tmp_path)
    assert {202761010,202761011}<={x['seed'] for x in result['seed_families']}
    assert [202781010,202781014] in [x['span'] for x in result['reset_spans']]
    assert [202781014,202781018] in [x['span'] for x in result['reset_spans']]
    assert 'c'*64 in result['root_hashes']
    assert result['confirmation_authority']['status']=='unknown'
