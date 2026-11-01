"""Saved sealed evidence only; no model, native simulator or output mutation."""
import pytest

from experiments.spectral_reachability import audit_kinodynamic_native_MAIN_20261028_v1 as beam
from experiments.spectral_reachability import audit_p4_native_MAIN_20261028_v1 as p4
from experiments.spectral_reachability import audit_p2_native_MAIN_20261028_v1 as p2


@pytest.mark.parametrize('module,folder', [
    (beam, 'agent_kinodynamic_matched_physical_20261028_v1_MAIN01'),
    (p4, 'agent_p4_moving_multiboundary_20261028_v1_MAIN02'),
    (p2, 'agent_p2_composition_execution_20261028_v1_physical_MAIN01'),
])
def test_external_hash_failclosed(module, folder):
    with pytest.raises(ValueError, match='report'):
        module.audit(module.ROOT/'results'/folder, '0'*64)


def test_full_sealed_beam_independent_native_audit():
    result = beam.audit(beam.ROOT/'results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01', beam.EXPECTED)
    assert result['verified_RGB'] == 44 and result['journal_events'] == 1186
    assert len(result['endpoints']) == 2 and not result['P1_admission']


def test_full_sealed_p4_independent_native_audit():
    result = p4.audit(p4.ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_MAIN02', p4.EXPECTED)
    assert result['verified_RGB'] == 247 and result['verified_native_control_returns'] == 151
    assert len(result['rows']) == 8 and sum(r['native_success'] for r in result['rows']) == 1
    assert not result['P4_admission']


def test_full_sealed_p2_independent_native_audit():
    result = p2.audit(p2.ROOT/'results/agent_p2_composition_execution_20261028_v1_physical_MAIN01', p2.EXPECTED)
    assert result['native_returns_verified'] == 80 and len(result['rows']) == 8
    assert not any(row['native_success'] for row in result['rows']) and not result['P2_admission']
