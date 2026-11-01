import hashlib
import json

import pytest

from experiments.spectral_reachability import codex_d2p_historical_exclusion_index_20261029 as index


def test_real_evidence_index():
    value = index.build()
    assert value['counts'] == {'original_files_verified': 88, 'observed_prefix_records': 80, 'selected_records': 52, 'conditional_envelopes': 11}
    assert value['freshness_certified'] is False
    assert value['original_reset_bounds_cleared'] == 0
    assert all(not x['certified'] for x in value['conditional_seed_envelopes'])
    assert all(x['physical_state_identity'] is None for x in value['selected_declared_records'])


def test_wrong_sha_rejected():
    name, _ = index.INPUTS['prefix']
    with pytest.raises(ValueError, match='SHA mismatch'):
        index.read_checked(index.RESULTS / name, '0' * 64)


def test_pointer_escapes_and_invalid():
    assert index.pointer({'a/b': {'x~y': [4]}}, '/a~1b/x~0y/0') == 4
    with pytest.raises(ValueError):
        index.pointer({}, 'not-a-pointer')
