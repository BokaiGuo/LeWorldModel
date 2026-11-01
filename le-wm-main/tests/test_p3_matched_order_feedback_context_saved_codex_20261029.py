"""Synthetic observed history rebuild; no neural model or physics."""
import numpy as np
import pytest
import agent_p3_fullcommit_live_20261028_v1_tests as fixture
import test_p3_matched_order_feedback_seam_codex_20261029 as seam_fixture
from experiments.spectral_reachability import p3_matched_order_feedback_context_saved_codex_20261029 as consumer


def test_observed_context_reconstructed_and_tamper_rejected():
    root,_ = fixture.request(); warm = fixture.m.old.kernel.moving.warmup_commands()
    root['root_context']['action_history'] = fixture.m.moving.normalized(root,warm)
    root['binding']['context_sha256'] = consumer.seam.core.digest(root['root_context'])
    result,events = seam_fixture.run('fixed_order')
    contexts = [e for e in events if e['kind'] == 'actual_context']
    observations = [dict(boundary=e['end'],latent=np.full(192,e['end']/100,np.float32),
                         rgb_sha256=e['rgb_sha256']) for e in contexts]
    assert not consumer.audit(root,warm,result,events,observations)['physical_authenticated']
    observations[0]['latent'][0] += 1
    with pytest.raises(ValueError,match='independent observed context'):
        consumer.audit(root,warm,result,events,observations)
