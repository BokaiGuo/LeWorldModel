import numpy as np

from experiments.spectral_reachability.reacher_d2p_state_conditioned_support import _nearest


def test_state_conditioned_retrieval_whitens_source_anchor_dimensions():
    anchors = np.asarray([[0.0, 1.0], [10.0, 0.0]])
    # Raw L2 would choose row 0 for query [1, 0]; source z-scoring makes the
    # low-variance second coordinate decisive and chooses row 1.
    chosen = _nearest(np.asarray([1.0, 0.0]), anchors, count=1,
                      mean=np.asarray([5.0, 0.5]), scale=np.asarray([10.0, 0.1]))
    assert chosen.tolist() == [1]
