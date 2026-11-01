import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability.audit_pusht_d2p_action_statistics import action_statistics


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_chunked_moments_match_independent_complete_array(chunk):
    rows = np.array([[1, 5], [3, 2], [np.nan, 200], [10, 7], [4, np.nan], [6, 3]], dtype=np.float32)
    result = action_statistics(rows, chunk)
    valid = rows[~np.isnan(rows).any(axis=1)].astype(np.float64)
    np.testing.assert_allclose(result["mean"], valid.mean(axis=0))
    np.testing.assert_allclose(result["sample_std"], valid.std(axis=0, ddof=1))
    np.testing.assert_allclose(result["population_std"], valid.std(axis=0))
    assert result["valid_rows"] == 4
    assert result["nan_rows_excluded"] == 2
    assert result["action_bytes_sha256"] == hashlib.sha256(rows.tobytes()).hexdigest()


@pytest.mark.parametrize("rows", [np.zeros((2, 2)), np.zeros((1, 2)), np.zeros((2, 3)),
                                  np.array([[1., np.inf], [2., 3.]]),
                                  np.ones((2, 2), dtype=np.int64)])
def test_invalid_statistics_fail_closed(rows):
    with pytest.raises(ValueError):
        action_statistics(rows)


def test_never_accesses_other_fields_or_unbounded_slices():
    class ActionOnly:
        shape, dtype = (7, 2), np.dtype("float32")

        def __getitem__(self, selection):
            assert selection.stop - selection.start <= 2
            return np.arange(14, dtype=self.dtype).reshape(7, 2)[selection]

    assert action_statistics(ActionOnly(), 2)["valid_rows"] == 7


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        action_statistics(np.ones((2, 2)), 0)
