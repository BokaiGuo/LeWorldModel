from pathlib import Path
import pytest
from experiments.spectral_reachability import main_p3_history_preconditions_compat_20261029_v1 as m
import test_agent_p3_same_action_history_preconditions_20261029_v1 as fixtures


def test_only_large_original_report_uses_its_existing_producer_cap():
    assert m.input_cap('planner_report') == 256*1024**2
    assert all(m.input_cap(k) == 128*1024**2 for k in m.resolver.KEYS if k != 'planner_report')
    with pytest.raises(ValueError): m.input_cap('unregistered_extra')
    assert m.resolver.sha(Path(m.resolver.__file__).read_bytes()) == m.PIN


def test_complete_saved_eight_slot_fixture(tmp_path):
    args = fixtures.save_inputs(tmp_path)
    result = m.build(**args)
    assert result['ready'] == 8 and result['blocked'] == 0
    assert result['compatibility']['R_IO_math_resource_checks_unchanged']
    assert result['resources_here']['predict'] == result['resources_here']['physics'] == 0


def test_missing_journal_not_bypassed(tmp_path):
    args = fixtures.save_inputs(tmp_path)
    (tmp_path/'planner/s0_adaptive_endpoints_b05/events.jsonl').unlink()
    result = m.build(**args)
    assert result['ready'] == 6 and result['blocked'] == 2 and len(result['jobs']) == 8


def test_authority_sha_still_required(tmp_path):
    args = fixtures.save_inputs(tmp_path); args['planner_report_sha'] = '0'*64
    with pytest.raises(ValueError): m.build(**args)
