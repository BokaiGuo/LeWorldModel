"""RAM saved-byte contracts only, never neural or physical evidence."""
from copy import deepcopy
import runpy
from pathlib import Path

import pytest

from experiments.spectral_reachability import chat_p2_20261027_lifecycle as q

fixture = runpy.run_path(str(Path(__file__).with_name('test_moving_p2_saved_program_graph_20261027_v1.py')))['fixture']


def prepared():
    g, memory = fixture()
    p = q.prepare(g, memory.__getitem__, algorithm='shortest', max_expansions=100)
    return p, q.graph.sha(q.graph.encode(p))


def test_full_roundtrip_keeps_all_edges_and_exact_boundaries():
    p, sha = prepared()
    r = q.run(p, sha)
    exported = q.consume(p, sha, r)
    assert r['candidate_edge_ids'] == ['sa', 'at', 'st']
    assert r['candidate_edge_denominator'] == 3
    assert exported['commands'] == [[.25], [-.5]]
    edges = exported['path']['saved_edges']
    assert edges[0]['child'] == edges[1]['parent']
    assert r['measured_world_model_predict_calls'] == r['measured_world_model_batch_samples'] == 0
    assert not exported['execution_authorized'] and not exported['P2_admission']


@pytest.mark.parametrize('field', ['state_sha256', 'context_sha256', 'latent_sha256', 'contract_sha256'])
def test_bad_boundary_rejected_before_prepare(field):
    g, memory = fixture()
    g['edges'][1]['parent'][field] = '0'*64
    with pytest.raises(ValueError):
        q.prepare(g, memory.__getitem__, algorithm='shortest', max_expansions=100)


def test_external_sha_and_full_nonwinner_census():
    p, sha = prepared()
    with pytest.raises(ValueError, match='external'):
        q.verify(p, '0'*64)
    p['candidate_edge_ids'].pop()
    with pytest.raises(ValueError, match='census'):
        q.verify(p, q.graph.sha(q.graph.encode(p)))


def test_embedded_sources_not_just_source_hash():
    p, sha = prepared()
    entry = next(iter(p['sources'].values()))
    entry['bytes_hex'] = b'changed'.hex()
    with pytest.raises(ValueError, match='source'):
        q.verify(p, q.graph.sha(q.graph.encode(p)))


@pytest.mark.parametrize('axis', ['predict_calls', 'predict_samples', 'physics_steps', 'flops'])
def test_unknown_saved_cost_never_zero(axis):
    g, memory = fixture()
    g['base_resources'][axis] = None
    with pytest.raises(ValueError):
        q.prepare(g, memory.__getitem__, algorithm='shortest', max_expansions=100)


def test_confirmatory_no_dispatch_even_with_engineering_valid_graph():
    p, sha = prepared()
    manifest = q.queue.build_draft(['slot0', 'slot1'])
    r = q.run(p, sha, confirmatory=True, manifest=manifest, load=lambda _: b'')
    assert r['status'] == 'blocked_confirmatory' and r['search'] is None
    assert len(r['queue_audit']['rows']) == 2
    assert r['queue_audit']['status'] == 'blocked'
    assert not r['execution_authorized'] and not r['P2_admission']


@pytest.mark.parametrize('mutation', ['winner', 'denominator', 'admission', 'samples'])
def test_consume_recomputes_not_just_report_sha(mutation):
    p, sha = prepared(); r = q.run(p, sha)
    if mutation == 'winner':
        r['search']['path']['commands'][0][0] = .9
    elif mutation == 'denominator':
        r['candidate_edge_denominator'] = 1
    elif mutation == 'admission':
        r['P2_admission'] = True
    else:
        r['measured_world_model_batch_samples'] = 1
    with pytest.raises(ValueError, match='drift'):
        q.consume(p, sha, r)


def test_search_limit_does_not_export_partial_winner():
    g, mem = fixture()
    p = q.prepare(g, mem.__getitem__, algorithm='shortest', max_expansions=1)
    sha = q.graph.sha(q.graph.encode(p)); r = q.run(p, sha)
    assert r['status'] == 'search_limit'
    with pytest.raises(ValueError, match='complete'):
        q.consume(p, sha, r)


def test_exclusive_write_failure_and_no_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(q, 'PROJECT', tmp_path)
    (tmp_path/'results').mkdir()
    p, sha = prepared()
    source = tmp_path/'results'/f'{q.PREFIX}prepared.json'
    output = tmp_path/'results'/f'{q.PREFIX}report.json'
    q.write(source, p)
    with pytest.raises(ValueError):
        q.run_file(source, '0'*64, output)
    assert q.read(output)['status'] == 'failed'
    assert q.read(output)['prepared_sha256'] == '0'*64
    with pytest.raises(ValueError, match='already exists'):
        q.run_file(source, sha, output)


def test_output_scope_and_prefix_rejection(tmp_path):
    with pytest.raises(ValueError, match='prefix'):
        q.write(tmp_path/'unowned.json', {})
    with pytest.raises(ValueError, match='subtree'):
        q.write(tmp_path/f'{q.PREFIX}outside.json', {})


def test_exact_producer_dispatch_without_weaker_fallback(monkeypatch):
    calls = []
    class Producer:
        @staticmethod
        def load_prepared(path):
            calls.append(path)
            raise ValueError('old source seal rejected')
        @staticmethod
        def selected_actions(path):
            calls.append(path)
            raise ValueError('old source seal rejected')
    monkeypatch.setattr(q.importlib, 'import_module', lambda _: Producer)
    for kind in q.PRODUCERS:
        with pytest.raises(ValueError, match='old source'):
            q.consume_producer(kind, '/not/read/by/mock')
    assert len(calls) == 3
