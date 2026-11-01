"""RAM-only synthetic provenance and interval fixtures; no producer imports."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import moving_historical_reset_quarantine_20261027_v1 as q


@pytest.fixture
def memory(monkeypatch):
    files, roster, requests = {}, [], []
    for ident, directory, _, base, n, schema, reserved in q.SOURCES:
        p0 = ident in ('A','B','C')
        proto = (dict(calibration_root_seed=base, calibration_roots=n) if p0 else
                 dict(root_seed=base, root_count=n, root_oversample_factor=4))
        if reserved is not None:
            proto['outcome'] = dict(root_seed=reserved,root_count=24,oversample_factor=4,status='reserved_not_opened')
        report = dict(schema=schema, protocol=proto,
                      **{('calibration_roots' if p0 else 'rows'): [dict(root_sha256=q.sha(str(i).encode())) for i in range(n)]})
        path = f'results/{directory}/report.json'
        files[path] = q.encode(report); h = q.sha(files[path])
        roster.append((ident,directory,h,base,n,schema,reserved))
        requests.append(dict(path=str(q.ROOT/path),sha256=h,inherited_refs=[
            dict(parent=parent,pointer=f'/exclusions/unknown_sources/{len(requests)}/source',
                 historical_sha256=h,original_reason=q.REASON)
            for parent in ('reachable_prepare','moving_prepare')]))
    monkeypatch.setattr(q,'SOURCES',tuple(roster))
    files[q.AUTHORITY] = q.encode(dict(history_file_requests=requests))
    monkeypatch.setattr(q,'AUTHORITY_SHA',q.sha(files[q.AUTHORITY]))
    files[q.DISPOSITION] = b'MOCK_DISPOSITION_NOT_ACTUAL'
    monkeypatch.setattr(q,'DISPOSITION_SHA',q.sha(files[q.DISPOSITION]))
    family = []
    for path, _, fn in q.FAMILY:
        name = f'git:{q.COMMIT}:le-wm-main/{path}'; files[name] = ('MOCK_CODE_'+fn).encode()
        family.append((path,q.sha(files[name]),fn))
    monkeypatch.setattr(q,'FAMILY',tuple(family))
    history = []
    for name, _ in q.HISTORICAL:
        if name.endswith('prerun_seal.json'):
            refs = []
            for i in range(7):
                snapshot = str(q.ROOT/f'results/mock_snapshots/{i}.txt')
                files[snapshot] = f'MOCK_SNAPSHOT_{i}'.encode()
                refs.append(dict(snapshot=snapshot,sha256=q.sha(files[snapshot])))
            files[name] = q.encode(dict(inputs=refs))
        elif name.endswith('/manifest.json'):
            refs = []
            for i in range(7):
                archive = f'sources/{i:03}.source'
                dest = str(q.Path(name).parent/archive); files[dest] = f'MOCK_ARCHIVE_{i}'.encode()
                refs.append(dict(archive=archive,sha256=q.sha(files[dest])))
            files[name] = q.encode(dict(files=refs))
        else:
            files[name] = ('MOCK_HISTORICAL_'+name).encode()
        history.append((name,q.sha(files[name])))
    monkeypatch.setattr(q,'HISTORICAL',tuple(history))
    return files


def build(memory):
    return q.build(memory.__getitem__)


def request(start=999999999, end=None, explicit=(), complete=True):
    return dict(schema=q.STEM+'.new_seed_spans',roots=[dict(new_root_id='new0',
        random_reset_spans=[dict(start=start,end=start if end is None else end)],
        explicit_state_reset_seeds=list(explicit),reset_coverage_declared_complete=complete)])


def test_full_fixed_denominator_noncertification(memory):
    out = build(memory)
    assert out['metadata_status'] == 'assembled'
    assert out['source_count'] == len(out['rows']) == 8
    assert sum(len(r['inherited_refs']) for r in out['rows']) == out['inherited_reference_count'] == 16
    assert out['certified_source_count'] == 0
    assert out['status'] == 'freshness_not_certified'
    for key in ('freshness_certified','historical_coverage_complete','old_freshness_blockers_cleared','P1_admission'):
        assert out[key] is False
    assert out['actual_neural_calls'] == out['actual_physics_calls'] == 0
    assert all(not r['historical_execution_source_bound'] and not r['actual_attempt_bound_certified'] for r in out['rows'])


def test_exact_eight_n_envelopes_and_seed_zero(memory):
    out = build(memory)
    for row, original in zip(out['rows'],q.SOURCES):
        _,_,_,seed,n,_,_ = original
        env = row['conditional_envelopes'][0]
        assert (env['start'],env['end']) == (seed,seed+8*n-1)
        assert env['assumptions'] == q.ASSUMPTIONS and env['unknown'] == q.UNKNOWN
        assert len(env['code_evidence']) == 4 and not env['certified']
        assert row['explicit_state_reset']['seed'] == 0
        assert row['explicit_state_reset']['not_a_random_root_identity']
    d = out['rows'][3]['conditional_envelopes'][0]
    assert (d['start'],d['end']) == (202710281,202710472)


def test_reservation_not_observed_history(memory):
    out = build(memory)
    for row in out['rows']:
        reserved = [e for e in row['conditional_envelopes'] if e['kind']=='reserved_not_observed_reset_envelope']
        assert len(reserved) == (1 if row['source_id'] in 'ABC' else 0)
        for e in reserved: assert not e['certified'] and e['root_count'] == 24
    assert out['rows'][0]['conditional_envelopes'][1]['end'] == 202753291


@pytest.mark.parametrize('seed', [202710281,202710472,202710350])
def test_closed_boundary_intersections(memory,seed):
    out = q.check_new(build(memory),request(seed))
    hits = out['checks'][0]['overlaps']
    assert any(h['source_id']=='D' for h in hits)
    assert any(h['source_id']=='C' and h['kind']=='reserved_not_observed_reset_envelope' for h in hits)
    assert out['status']=='freshness_not_certified'


def test_no_intersection_is_not_freshness(memory):
    out = q.check_new(build(memory),request())
    check = out['checks'][0]
    assert check['no_known_span_intersection'] and not check['overlaps']
    assert check['fresh_coverage_unknown'] and not out['freshness_certified']
    assert not out['old_freshness_blockers_cleared']


@pytest.mark.parametrize('via_random', [True,False])
def test_seed_zero_label_not_physical_identity(memory,via_random):
    req = request(0) if via_random else request(explicit=[0])
    c = q.check_new(build(memory),req)['checks'][0]
    assert len(c['explicit_state_seed_labels']) == 8 and c['no_known_span_intersection']
    assert all('not_physical_identity_proof' in x['classification'] for x in c['explicit_state_seed_labels'])


def test_incomplete_new_reset_coverage_remains_missing(memory):
    c = q.check_new(build(memory),request(complete=False))['checks'][0]
    assert 'new_reset_coverage_not_declared_complete' in c['missing']


def test_span_arithmetic_without_enumeration(memory):
    c = q.check_new(build(memory),request(0,2**63-1))['checks'][0]
    assert len(c['overlaps']) == 11 and len(c['explicit_state_seed_labels']) == 8


def test_missing_report_preserves_eight_and_refs(memory):
    del memory['results/'+q.SOURCES[3][1]+'/report.json']
    out = build(memory)
    assert out['metadata_status']=='blocked_evidence' and len(out['rows'])==8
    assert out['rows'][3]['status']=='blocked_report_evidence'
    assert sum(len(r['inherited_refs']) for r in out['rows'])==16
    assert not out['freshness_certified']


def test_tampered_report_and_missing_archive_not_certified(memory):
    p = 'results/'+q.SOURCES[0][1]+'/report.json'; memory[p] += b' '
    out = build(memory)
    assert out['rows'][0]['status']=='blocked_report_evidence'
    snap = next(k for k in memory if '/mock_snapshots/' in k); del memory[snap]
    out = build(memory)
    assert any(r['status']=='evidence_unavailable' for r in out['evidence'])
    assert out['metadata_status']=='blocked_evidence'


def test_bound_facts_only_report_not_actual_loop(memory):
    out = build(memory)
    for row in out['rows']:
        assert all(f['classification']=='report_bytes_bound_fact_not_execution_source' for f in row['bound_facts'])
        assert len(row['bound_facts'])>=2
    assert any(e.get('verified_snapshot_count')==7 for e in out['evidence'])
    assert any(e.get('verified_archive_count')==7 for e in out['evidence'])


def test_authority_sha_and_ref_integrity(memory,monkeypatch):
    a = q.parse(memory[q.AUTHORITY]); a['history_file_requests'][0]['inherited_refs'].pop()
    memory[q.AUTHORITY]=q.encode(a); monkeypatch.setattr(q,'AUTHORITY_SHA',q.sha(memory[q.AUTHORITY]))
    with pytest.raises(ValueError,match='references'): build(memory)


def test_roster_cannot_shrink_or_result_filter(memory,monkeypatch):
    a = q.parse(memory[q.AUTHORITY]); a['history_file_requests'].pop()
    memory[q.AUTHORITY]=q.encode(a); monkeypatch.setattr(q,'AUTHORITY_SHA',q.sha(memory[q.AUTHORITY]))
    with pytest.raises(ValueError,match='eight'): build(memory)


@pytest.mark.parametrize('value', [-1,True,1.0,2**63,'1'])
def test_invalid_seed_types(memory,value):
    with pytest.raises(ValueError): q.check_new(build(memory),request(value))


def test_invalid_new_request_unknown_field_reversed_duplicate(memory):
    req = request(2,1)
    with pytest.raises(ValueError): q.check_new(build(memory),req)
    req = request(); req['outcome']=1
    with pytest.raises(ValueError): q.check_new(build(memory),req)
    req = request(); req['roots'].append(deepcopy(req['roots'][0]))
    with pytest.raises(ValueError): q.check_new(build(memory),req)


def test_publish_floor_exclusive_and_cap_no_disk(monkeypatch):
    monkeypatch.setattr(q.shutil,'disk_usage',lambda _:SimpleNamespace(free=q.FLOOR))
    with pytest.raises(ValueError,match='floor'): q.write_new('never.json',{})
    with pytest.raises(ValueError,match='cap'): q.write_new('never.json',{'large':'a'*q.CAP})
    monkeypatch.setattr(q.shutil,'disk_usage',lambda _:SimpleNamespace(free=q.FLOOR+q.CAP))
    def exists(path,mode):
        assert mode=='xb'; raise FileExistsError('preserve existing')
    monkeypatch.setattr(q.Path,'open',exists)
    with pytest.raises(FileExistsError): q.write_new('old.json',{})


def test_no_pickle_no_neural_imports_and_duplicate_json():
    with pytest.raises(ValueError): q.parse(b'{"x":1,"x":2}')
    with pytest.raises(ValueError): q.parse(b'{"x":NaN}')
    with pytest.raises(ValueError): q.local_read('git:anything:arbitrary')
    with pytest.raises(ValueError): q.local_read('../outside')


def test_cli_build_request_and_help_in_ram(memory,monkeypatch,capsys):
    outputs = {}
    # Explicit original implementation avoids recursion after patch.
    monkeypatch.setattr(q,'build',lambda:original_build(memory.__getitem__))
    monkeypatch.setattr(q,'write_new',lambda name,value:outputs.setdefault(name,value) and q.sha(q.encode(value)))
    req = q.encode(request()); memory['new.json']=req
    monkeypatch.setattr(q,'local_read',memory.__getitem__)
    assert q.main(['--output','meta.json'])==2
    assert outputs['meta.json']['new_seed_check']['status']=='not_requested'
    assert q.main(['--output','check.json','--new-seeds','new.json','--new-seeds-sha',q.sha(req)])==2
    assert outputs['check.json']['new_seed_check']['status']=='freshness_not_certified'
    with pytest.raises(ValueError): q.main(['--output','missing.json','--new-seeds','new.json'])
    with pytest.raises(SystemExit) as e:q.main(['--help'])
    assert e.value.code==0


original_build=q.build


def test_each_row_version_scope_is_not_backdated(memory):
    out=build(memory)
    assert out['rows'][5]['historical_version_disposition'].startswith('exact_P1_v2_producer_not_recovered')
    assert all(not x['proves_original_eight_run_source_binding'] for x in out['evidence'])
    assert any(x['scope']=='postrun_snapshot_only' for x in out['evidence'])
    assert any(x['scope']=='later_run_evidence_only' for x in out['evidence'])


def test_unavailable_evidence_exposes_partial_comparison(memory):
    del memory['results/'+q.SOURCES[3][1]+'/report.json']
    out=q.check_new(build(memory),request())
    assert not out['checks'][0]['comparison_complete_for_listed_envelopes']
    assert not out['freshness_certified']
