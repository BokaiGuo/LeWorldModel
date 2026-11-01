"""Real saved-input binding audit and isolated corruption tests; no inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_accounting_binding_audit_20261027 as binding


ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return {key: binding.strict_json(binding.read(ROOT / relative)) for key, relative in (
        ('report', binding.REPORT), ('protocol', binding.PROTOCOL),
        ('compilation', binding.COMPILATION), ('main', binding.MAIN))}


def rebind(data):
    """Rehash mutated protocols so structural checks cannot hide behind SHA rejection."""
    identity = binding.protocol_id(data['protocol']['protocol'])
    data['protocol']['protocol_id'] = identity
    data['protocol']['paired_plan']['source_seal'] = identity
    data['report']['protocol_id'] = identity
    for root in data['protocol']['paired_plan']['roots']:
        root['contract']['binding']['protocol_id'] = identity


def test_actual_saved_inputs_pass_without_mutation_or_neural_execution():
    data = inputs()
    original = deepcopy(data)
    result = binding.audit(**data)
    assert data == original
    assert result['sources_count'] == 52 and result['root_count'] == 4
    assert result['root_method_rows'] == 8 and result['kernels'] == 72
    assert result['planning_events'] == 1152 and result['actual_predict_samples'] == 1171
    assert result['corrected_rows'] == 2
    assert [r['actual_horizon'] for r in result['rows']] == [3, 2, 3, 3, 3, 2, 3, 3]
    assert sum(pair['equal_resources'] for pair in result['paired']['pairs']) == 2
    assert result['paired']['all_pairs_within_budget']
    assert not any(result[k] for k in ('independent_neural_replay', 'P1_admission',
                                      'P3_admission', 'efficacy_claim', 'compute_advantage'))
    assert result['new_predict_samples'] == result['new_physical_steps'] == 0


def test_protocol_hash_is_original_runner_serialization_not_compact_or_double_sha():
    data = inputs()
    protocol = data['protocol']['protocol']
    expected = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert binding.protocol_id(protocol) == expected == data['protocol']['protocol_id']
    assert expected != hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert expected != hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize('change', ['content', 'envelope_id', 'report_id', 'plan_id'])
def test_protocol_content_sha_rejects_mutation(change):
    data = inputs()
    if change == 'content':
        data['protocol']['protocol']['predict_cap'] += 1
    elif change == 'envelope_id':
        data['protocol']['protocol_id'] = '0' * 64
    elif change == 'report_id':
        data['report']['protocol_id'] = '0' * 64
    else:
        data['protocol']['paired_plan']['source_seal'] = '0' * 64
    with pytest.raises(ValueError, match='protocol content SHA256'):
        binding.audit(**data)


@pytest.mark.parametrize('change', ['empty_sources', '51_sources', '53_sources', 'relative_source',
                                  'bad_source_sha', 'three_roots', 'duplicate_root', 'root_order',
                                  'plan_order', 'compilation_order', 'compiled_row_root',
                                  'model', 'task', 'plan_binding', 'duplicate_row', 'missing_row'])
def test_protocol_sources_cohort_and_plan_binding_reject_corruption(change):
    data = inputs()
    p = data['protocol']['protocol']
    plan = data['protocol']['paired_plan']
    sources = p['sources']
    first = next(iter(sources))
    if change == 'empty_sources':
        p['sources'] = {}
    elif change == '51_sources':
        sources.pop(first)
    elif change == '53_sources':
        sources['/extra_source.py'] = '0' * 64
    elif change == 'relative_source':
        sources['relative.py'] = sources.pop(first)
    elif change == 'bad_source_sha':
        sources[first] = 'not-a-digest'
    elif change == 'three_roots':
        p['roots'].pop()
    elif change == 'duplicate_root':
        p['roots'][1] = p['roots'][0]
    elif change == 'root_order':
        p['roots'].reverse()
    elif change == 'plan_order':
        plan['roots'].reverse()
    elif change == 'compilation_order':
        data['compilation']['protocol']['roots'].reverse()
    elif change == 'compiled_row_root':
        data['compilation']['rows'][0]['root_sha256'] = '0' * 64
    elif change in ('model', 'task'):
        plan['roots'][0]['contract']['binding'][{'model': 'frozen_model_id', 'task': 'task_id'}[change]] = 'wrong'
    elif change == 'duplicate_row':
        data['report']['rows'][1] = deepcopy(data['report']['rows'][0])
    elif change == 'missing_row':
        data['report']['rows'].pop()
    rebind(data)
    if change == 'plan_binding':
        plan['roots'][0]['contract']['binding']['protocol_id'] = '0' * 64
    with pytest.raises(ValueError):
        binding.audit(**data)


@pytest.mark.parametrize('change', ['parent999', 'parent_negative', 'parent_bool', 'parent_wrong_legal',
                                  'parent_order', 'candidate999', 'candidate_bool', 'candidate_order',
                                  'extra_kernel', 'failed_event', 'incomplete_replay', 'event_count',
                                  'event_cost', 'decision', 'beam_width', 'retained', 'best999',
                                  'best_other_legal', 'best_cost', 'actions999', 'actions_outofbox',
                                  'actions_inbox_wrong', 'actions_nan', 'missing_action', 'main_drift'])
def test_all_saved_frontier_candidate_witness_bindings_fail_closed(change):
    data = inputs()
    row = data['report']['rows'][0]
    search = row['search']
    layer = search['layers'][1]
    kernel = layer['kernels'][0]
    event = kernel['receipt']['attempts'][0]
    best = search['best_plan']
    if change.startswith('parent_') or change == 'parent999':
        if change == 'parent_order':
            layer['kernels'][0], layer['kernels'][1] = layer['kernels'][1], layer['kernels'][0]
        else:
            kernel['parent_indices'][0] = {'parent999': 999, 'parent_negative': -1,
                'parent_bool': True, 'parent_wrong_legal': (kernel['parent_indices'][0]+1) % 16}[change]
    elif change == 'candidate999':
        event['primitive_indices'][0] = 999
    elif change == 'candidate_bool':
        event['primitive_indices'][0] = False
    elif change == 'candidate_order':
        kernel['receipt']['attempts'].reverse()
    elif change == 'extra_kernel':
        layer['kernels'].append(deepcopy(kernel))
    elif change == 'failed_event':
        event['succeeded'] = False
    elif change == 'incomplete_replay':
        event['replay_completed'] = False
    elif change == 'event_count':
        kernel['receipt']['attempts'].pop()
    elif change == 'event_cost':
        event['cost'] = -1
    elif change == 'decision':
        layer['decision']['horizon_tokens'] = 999
    elif change == 'beam_width':
        layer['beam_width'] += 1
    elif change == 'retained':
        layer['retained'] += 1
    elif change == 'best999':
        best['primitive_indices'][0] = 999
    elif change == 'best_other_legal':
        best['primitive_indices'][0] = (best['primitive_indices'][0]+1) % 16
    elif change == 'best_cost':
        best['cost'] += 1
    elif change in ('actions999', 'actions_outofbox', 'actions_nan'):
        best['actions'][0][0] = {'actions999': 999, 'actions_outofbox': 1.0001, 'actions_nan': float('nan')}[change]
    elif change == 'actions_inbox_wrong':
        best['actions'][0][0] = .123456789
    elif change == 'missing_action':
        best['actions'].pop()
    elif change == 'main_drift':
        data['main']['actual_predict_samples'] += 1
    with pytest.raises(ValueError):
        binding.audit(**data)


@pytest.mark.parametrize('change', ['order', 'missing', 'outofbox', 'nonfinite', 'shape', 'wrong_inbox'])
def test_compilation_raw_bank_identity_and_actions(change):
    data = inputs()
    receipts = data['compilation']['rows'][0]['compiled']['receipts']
    if change == 'order':
        receipts.reverse()
    elif change == 'missing':
        receipts.pop()
    else:
        # Change the actual selected prefix, not an unvisited bank command.
        index = data['report']['rows'][0]['search']['best_plan']['primitive_indices'][0]
        command = receipts[index]['program_live_commands'][0]
        if change == 'shape':
            command.pop()
        else:
            command[0] = {'outofbox': 999, 'nonfinite': float('inf'), 'wrong_inbox': .123456789}[change]
    with pytest.raises(ValueError):
        binding.audit(**data)


def test_main_v2_and_all_current_sources_match_exact_frozen_hashes():
    data = inputs()
    assert binding.sha(binding.read(ROOT / binding.MAIN)) == binding.MAIN_SHA
    assert data['main']['inputs'] == {str(ROOT / p): binding.sha(binding.read(ROOT / p))
                                    for p in (binding.REPORT, binding.PROTOCOL)}
    assert data['protocol']['protocol']['bindings'][str(ROOT / binding.COMPILATION)] == binding.sha(binding.read(ROOT / binding.COMPILATION))
    for path, expected in data['protocol']['protocol']['sources'].items():
        assert binding.sha(binding.read(path)) == expected
    seal = data['main']['source_seal']
    assert binding.seals.verify_seal(output_dir=seal['output_dir'],
        expected_manifest_sha256=seal['manifest_sha256'])['status'] == 'verified'


@pytest.mark.parametrize('change', ['root_id', 'protocol_id', 'task_id', 'frozen_model_id',
                                  'method', 'phase_method', 'phase_swap', 'calls'])
def test_reused_paired_accounting_still_rejects_receipt_misbinding(change):
    data = inputs()
    receipt = data['report']['rows'][0]['prediction_receipt']
    if change in ('root_id', 'protocol_id', 'task_id', 'frozen_model_id'):
        receipt['binding'][change] = 'wrong'
    elif change == 'method':
        receipt['method'] = 'wrong'
    elif change == 'phase_method':
        receipt['phases'][0]['method'] = 'wrong'
    elif change == 'phase_swap':
        receipt['phases'].reverse()
    else:
        receipt['phases'][0]['predict_calls'] += 1
    with pytest.raises(ValueError):
        binding.audit(**data)


@pytest.mark.parametrize('change', ['authority_sha', 'original_input_sha'])
def test_run_rejects_wrong_authority_before_creating_output(tmp_path, monkeypatch, change):
    original_read = binding.read

    def corrupted_read(path):
        raw = original_read(path)
        if Path(path) == ROOT / binding.MAIN and change == 'authority_sha':
            return raw + b' '
        if Path(path) == ROOT / binding.REPORT and change == 'original_input_sha':
            return raw + b' '
        return raw

    monkeypatch.setattr(binding, 'read', corrupted_read)
    output = tmp_path / binding.OUTPUT
    with pytest.raises(ValueError, match='SHA mismatch|input hash bindings'):
        binding.run(ROOT, output=output)
    assert not output.exists()


def test_existing_output_rejected_before_any_input_reads(tmp_path, monkeypatch):
    def forbidden_read(path):
        raise AssertionError('must refuse overwrite before loading inputs')
    monkeypatch.setattr(binding, 'read', forbidden_read)
    with pytest.raises(ValueError, match='exclusive output'):
        binding.run(ROOT, output=tmp_path)
