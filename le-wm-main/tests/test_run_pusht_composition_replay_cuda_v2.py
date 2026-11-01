import numpy as np
import pytest
from experiments.spectral_reachability.run_pusht_composition_replay_cuda_v2 import root_image_binding
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest


def test_whole_context_hash_not_last_character():
    value = 'b'*64
    binding = root_image_binding({'history_rgb_sha256': value})
    assert binding['root_rgb_sha256'] == value
    assert binding['root_rgb_scope'] == 'stacked_three_frame_context_not_single_frame'


@pytest.mark.parametrize('value', ['b', ['a'*64]*3, None, 'z'*64])
def test_wrong_digest_schema_rejected(value):
    with pytest.raises(ValueError): root_image_binding({'history_rgb_sha256': value})


def test_numpy_normalization_statistics_use_native_json_scalars():
    values = np.array([.1, .2], np.float32)
    assert len(digest({'mean': values.tolist(), 'std': values.tolist()})) == 64
    with pytest.raises(ValueError): digest({'mean': list(values)})
