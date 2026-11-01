import base64
import hashlib
import struct

import pytest

from experiments.spectral_reachability import agent_e1_shared_executable_target_20261029_v1 as m


def records():
    result = []
    for root in (0, 1):
        raw = struct.pack('<192f', *([float(root)] * 192))
        result.append(dict(root=root, task_sha256='1'*64, history_sha256='2'*64,
                           goal_sha256='3'*64, target_b64=base64.b64encode(raw).decode(),
                           target_sha256=hashlib.sha256(raw).hexdigest(),
                           witness_sha256='4'*64, witness_receipt_sha256='5'*64))
    return result


def test_common_target_and_no_witness_in_payload():
    result = m.assemble(records(), ['control', 'intervention'])
    assert len(result['rows']) == 4
    for i in (0, 2):
        assert result['rows'][i]['controller_input'] == result['rows'][i+1]['controller_input']
        assert not any('witness' in k for k in result['rows'][i]['controller_input'])
    assert not result['physical_executability_authenticated']
    assert not result['production_authorized']


@pytest.mark.parametrize('mutation', ['order', 'digest', 'extra', 'nan', 'labels'])
def test_invalid_inputs_rejected(mutation):
    data = records()
    labels = ['control', 'intervention']
    if mutation == 'order': data.reverse()
    if mutation == 'digest': data[0]['target_sha256'] = '0'*64
    if mutation == 'extra': data[0]['outcome'] = 1
    if mutation == 'labels': labels = ['control', 'control']
    if mutation == 'nan':
        raw = struct.pack('<192f', float('nan'), *([0.0]*191))
        data[0]['target_b64'] = base64.b64encode(raw).decode()
        data[0]['target_sha256'] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError): m.assemble(data, labels)


def test_separate_witness_does_not_change_controller_input():
    data = records()
    before = m.assemble(data, ['a', 'b'])
    data[0]['witness_sha256'] = 'a'*64
    after = m.assemble(data, ['a', 'b'])
    assert before['rows'][0]['controller_input'] == after['rows'][0]['controller_input']
    assert m.canonical_sha(before) != m.canonical_sha(after)
