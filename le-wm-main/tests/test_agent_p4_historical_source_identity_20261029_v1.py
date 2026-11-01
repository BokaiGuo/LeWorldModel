"""RAM projection/mutations plus fixed saved metadata regression; no experiments."""
import ast
import base64
import copy
import gzip
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_p4_historical_source_identity_20261029_v1 as m


def fixture_report(sid='A'):
    container = 'calibration_roots' if sid in 'ABC' else 'rows'
    count = 24 if sid == 'D' else 4
    report = dict(schema='test.metadata', protocol=dict(root_seed=101),
        **{container: [dict(root_id=i, root_sha256=f'{i:064x}',
            history_rgb_sha256='b'*64, goal_rgb_sha256='c'*64,
            compilation={'score': 'DO_NOT_DECODE'}, jacobian={'J': 'DO_NOT_DECODE'})
            for i in range(count)]})
    source = dict(id=sid, path='/saved/report.json', sha256='a'*64,
        protocol_projection=[dict(pointer='/protocol/root_seed', value=101)])
    return report, source


@pytest.mark.parametrize('sid', list('ABCDEFGH'))
def test_identity_all_denominators_and_nulls(sid):
    v, source = fixture_report(sid)
    rows = m.identity_projection(m.encode(v), source)['selected_rows']
    assert len(rows) == (24 if sid == 'D' else 4)
    for row in rows:
        assert row['source_episode_identity'] is None
        assert row['physical_state_identity'] is None
        assert row['ordinal_is_not_episode_id']
        assert row['missing'] == list(m.ABSENT)
        assert all(f['sha256'] == source['sha256'] and f['path'] == source['path'] for f in row['fields'])
        assert not any('compilation' in f['pointer'] or 'jacobian' in f['pointer'] for f in row['fields'])


def test_skipped_outcomes_never_json_decode():
    v, source = fixture_report()
    # Deliberately invalid escape would fail if the opaque metric string were
    # ever decoded. This tests lexical skipping, not JSON validity certification.
    raw = m.encode(v).replace(b'DO_NOT_DECODE', b'\\uQQQQ')
    assert len(m.identity_projection(raw, source)['selected_rows']) == 4


@pytest.mark.parametrize('field,value', [('root_sha256', 'short'), ('root_id', True),
    ('goal_rgb_sha256', None), ('history_rgb_sha256', 'G'*64)])
def test_bad_identity_rejected(field, value):
    v, source = fixture_report(); v['calibration_roots'][0][field] = value
    with pytest.raises(ValueError):
        m.identity_projection(m.encode(v), source)


@pytest.mark.parametrize('field', ['root_id', 'root_sha256', 'goal_rgb_sha256', 'history_rgb_sha256'])
def test_required_identity_missing(field):
    v, source = fixture_report(); del v['calibration_roots'][0][field]
    with pytest.raises(ValueError):
        m.identity_projection(m.encode(v), source)


def test_config_mismatch_and_denominator():
    v, source = fixture_report(); v['protocol']['root_seed'] += 1
    with pytest.raises(ValueError, match='config join'):
        m.identity_projection(m.encode(v), source)
    v, source = fixture_report(); v['calibration_roots'].pop()
    with pytest.raises(ValueError, match='denominator'):
        m.identity_projection(m.encode(v), source)


def test_static_python_never_executes():
    raw = b'import forbidden_runtime\nraise RuntimeError("NEVER EXECUTE")\ndef root_fingerprint(x):\n return x\nenv.reset(seed=0)\n'
    result = m.static_python(raw)
    assert not result['executed'] and not result['original_load_attested']
    assert result['import_declarations'][0]['line'] == 1
    assert result['reset_collector_identity_calls'][0]['line'] == 5
    assert result['identity_related_definitions'][0]['name'] == 'root_fingerprint'


@pytest.fixture(scope='module')
def actual_inputs():
    authority = m.a.parse((m.ROOT/m.AUTHORITY[0]).read_bytes())
    catalog = m.a.parse((m.ROOT/m.CATALOG[0]).read_bytes())
    assert m.sha((m.ROOT/m.AUTHORITY[0]).read_bytes()) == m.AUTHORITY[1]
    assert m.sha((m.ROOT/m.CATALOG[0]).read_bytes()) == m.CATALOG[1]
    return authority, catalog


def test_original_exact_eight_sixteen(actual_inputs):
    authority, catalog = actual_inputs
    result = m.selected_refs(authority, catalog)
    assert len(result) == 8
    assert sum(len(x['inherited_refs']) for x in result.values()) == 16
    rows = []
    for source in catalog['sources']:
        raw = Path(source['path']).read_bytes()
        assert m.sha(raw) == source['sha256']
        rows += m.identity_projection(raw, source)['selected_rows']
    assert len(rows) == 52
    assert not any('episode_id' in f['pointer'] for row in rows for f in row['fields'])


@pytest.mark.parametrize('mutation', ['drop_ref', 'drop_source', 'sha', 'parent', 'duplicate_path'])
def test_authority_join_mutations(actual_inputs, mutation):
    authority, catalog = copy.deepcopy(actual_inputs)
    if mutation == 'drop_ref':
        authority['history_file_requests'][0]['inherited_refs'].pop()
    elif mutation == 'drop_source':
        catalog['sources'].pop()
    elif mutation == 'sha':
        catalog['sources'][0]['sha256'] = 'f'*64
    elif mutation == 'parent':
        catalog['sources'][0]['references'][0]['parent'] = 'not_a_parent'
    else:
        catalog['sources'][1]['path'] = catalog['sources'][0]['path']
    with pytest.raises(ValueError):
        m.selected_refs(authority, catalog)


def delta_raw(disposition):
    raw = m.encode(disposition)
    return m.encode(dict(schema='agent_freshroots_resume_20261027_next.lossless-envelope.v1',
        semantic_bytes=len(raw), semantic_sha256=m.sha(raw),
        gzip_base64=base64.b64encode(gzip.compress(raw)).decode()))


def test_saved_receipts_do_not_imply_reset_or_load():
    raw = delta_raw(dict(disposition=dict(sources=[dict(dispatch_status=[
        dict(dispatch_call_id='call_x', dispatch_line=12, state='terminal_exit_recorded', exit_code=0)])]),
        metadata_delta={'outcomes': 'skipped'}))
    receipt = m.process_receipts(raw)['call_x']
    assert receipt['fields']['exit_code'] == 0
    assert receipt['envelope_pointer'] == '/disposition/sources/0/dispatch_status/0'
    e = m.a.parse(raw); e['semantic_sha256'] = '0'*64
    with pytest.raises(ValueError, match='payload SHA'):
        m.process_receipts(m.encode(e))
    e = m.a.parse(raw)
    e['gzip_base64'] = base64.b64encode(base64.b64decode(e['gzip_base64'])+b'extra').decode()
    with pytest.raises(ValueError, match='single delta'):
        m.process_receipts(m.encode(e))


def test_original_process_receipts(actual_inputs):
    raw = (m.ROOT/m.DELTA[0]).read_bytes(); assert m.sha(raw) == m.DELTA[1]
    receipts = m.process_receipts(raw)
    catalog = actual_inputs[1]
    assert set(receipts) == {d['call_id'] for d in catalog['recovery']['dispatches']}
    rows = m.closure_rows(catalog, [], receipts)
    assert len(rows) == 10
    assert all(not x['original_load_closure_complete'] for x in rows)
    assert all(r['original_loaded_sha256'] is None for x in rows for r in x['roles'])
    assert all(not x['candidate_bytes_available'] for row in rows for x in row['roles'])


def test_postrun_sha_match_not_original_load(actual_inputs):
    catalog = actual_inputs[1]
    candidate = dict(path='/postrun/snapshot.py', sha256=m.POSTRUN_FILES[0][1],
        original_path='experiments/spectral_reachability/preflight_pusht_d2p_p0.py',
        origin='postrun')
    rows = m.closure_rows(catalog, [candidate], {})
    row = next(x for x in rows if x['source_id'] == 'A')
    producer = next(x for x in row['roles'] if x['role'] == 'producer')
    assert producer['candidate_bytes_available']
    assert not producer['original_load_attested']
    assert producer['original_loaded_sha256'] is None


def test_fixed_git_blob_path_rejected():
    with pytest.raises(ValueError, match='fixed historical Git'):
        m.git_blob('git:HEAD:somewhere.py')


def test_snapshot_hash_failure_not_missing(actual_inputs):
    reader = m.r.Reader(provider=lambda _:b'wrong bytes')
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.source_candidates(reader, actual_inputs[1])


def test_module_has_no_model_env_execution():
    tree = ast.parse(Path(m.__file__).read_bytes())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(x.name not in ('torch', 'gym', 'gymnasium', 'stable_worldmodel') for x in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ('eval', 'exec', '__import__')


def test_writer_exclusive_and_verify_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda _:type('D', (), {'free':10**10})())
    result = {'status':'test_not_evidence'}
    p = tmp_path/'results'/(m.NAME+'_RAM.json')
    digest = m.write_new(p, result)
    with pytest.raises(FileExistsError):
        m.write_new(p, result)
    monkeypatch.setattr(m, 'build', lambda:result)
    assert m.verify(p, digest)['freshness_certified'] is False
    monkeypatch.setattr(m, 'build', lambda:{'status':'changed'})
    with pytest.raises(ValueError, match='rebuild'):
        m.verify(p, digest)
    with pytest.raises(ValueError, match='report SHA'):
        m.verify(p, 'f'*64)


@pytest.mark.parametrize('name', ['foreign.json', '../outside.json'])
def test_writer_rejects_nonexclusive_path(tmp_path, monkeypatch, name):
    monkeypatch.setattr(m, 'ROOT', tmp_path); (tmp_path/'results').mkdir()
    with pytest.raises(ValueError, match='exclusive new result'):
        m.write_new(tmp_path/'results'/name, {})


@pytest.fixture(scope='module')
def actual_recovery():
    return m.build()


def test_actual_saved_metadata_closure_recovery(actual_recovery):
    v = actual_recovery
    assert v['summary']['selected_identity_rows'] == 52
    assert v['summary']['inherited_references'] == 16
    assert len(v['byte_verified_source_candidates']) == 21
    assert v['missing_snapshot_files'] == []
    assert v['summary']['original_load_closures_attested'] == 0
    assert v['global_history_complete'] is None
    assert not v['P1_admitted'] and not v['freshness_certified']
    assert not v['source_run_authorized'] and not v['outcome_values_decoded']
    assert v['neural_calls'] == v['physics_calls'] == 0
    assert all(len(s['inherited_refs']) == 2 for s in v['sources'])
    assert all(x['parent_evidence']['sha256'] for s in v['sources'] for x in s['inherited_refs'])


def test_actual_bindings_cover_projector_and_no_model_bytes(actual_recovery):
    paths = {x['path'] for x in actual_recovery['evidence_files']}
    assert str(m.ROOT/'experiments/spectral_reachability'/m.DEPENDENCIES[0][0]) in paths
    assert not any(p.endswith(('weights.pt', '.jsonl')) for p in paths)
    assert all(not c['original_load_attested'] for c in actual_recovery['byte_verified_source_candidates'])
