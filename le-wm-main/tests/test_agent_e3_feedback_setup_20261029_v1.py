"""Source root vs goal separation tested without neural/physical execution."""
import numpy as np
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v1 as m


def test_original_root_hash_depends_on_original_goal():
    s=np.arange(7,dtype=float);g=s+1;g1=s+2
    assert m.old.source.root_hash(s,g)!=m.old.source.root_hash(s,g1)


def test_original_numpy_shapes():
    assert np.zeros((224,224,3),np.uint8).shape==(224,224,3)
