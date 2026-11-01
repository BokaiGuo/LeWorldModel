"""Synthetic callbacks only; no neural model or native physics."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
import agent_p3_fullcommit_live_20261028_v1_tests as fixture
from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as seam
from experiments.spectral_reachability import p3_matched_order_feedback_check_codex_20261029 as check


class Session:
    episode_id = 'synthetic'
    def packet(self, start, raw):
        return dict(raw=raw.copy(), rgb_bytes=('frame%d' % (start+5)).encode(),
                    end=start+5, stopped=False, episode_id=self.episode_id)
    def encode(self, end, rgb):
        return dict(latent=np.full(192, end/100, np.float32), input_sha256=hashlib.sha256(rgb).hexdigest(),
                    end=end, episode_id=self.episode_id)


def run(policy, session=None):
    root,_ = fixture.request()
    warm = fixture.m.old.kernel.moving.warmup_commands()
    root['root_context']['action_history'] = fixture.m.moving.normalized(root, warm)
    root['binding']['context_sha256'] = seam.core.digest(root['root_context'])
    prior = warm[-5:].copy()
    events = fixture.Events()
    result = seam.execute_slot(root, policy, prior, session or Session(),
                               lambda task, store: fixture.RamBackend(task, store), events, permit_ram=True)
    return result, events.events


@pytest.mark.parametrize('policy', ['fixed_order', 'signal_order'])
def test_actual_context_replans_and_structural_checker(policy):
    result, events = run(policy)
    assert result['status'] == 'callback_feedback_complete_not_certified', result['errors']
    assert check.check(result, events)['physical_certified'] is False
    bad = deepcopy(result); bad['blocks'][1]['context_sha256'] = '0'*64
    bad_events = deepcopy(events)
    next(e for e in bad_events if e['kind'] == 'block_compiled' and e['stage'] == 1)['context_sha256'] = '0'*64
    with pytest.raises(ValueError, match='actual joined context'):
        check.check(bad, bad_events)


def test_wrong_encoder_binding_stops_before_replanning():
    class Bad(Session):
        def encode(self, end, rgb):
            r = super().encode(end, rgb); r['input_sha256'] = '0'*64; return r
    result, _ = run('fixed_order', Bad())
    assert result['status'] == 'failed'
    assert len(result['blocks']) == len(result['packets']) == 1


def test_partial_native_stop_does_not_pad_or_replan():
    class Stop(Session):
        def packet(self, start, raw):
            r = super().packet(start, raw); r.update(raw=raw[:2].copy(), end=start+2, stopped=True); return r
    result, _ = run('fixed_order', Stop())
    assert result['status'] == 'native_stopped'
    assert len(result['blocks']) == 1 and result['packets'][0]['end'] == 2
    assert result['resource_receipt']['attempted']['predict_API'] < seam.design.expected()['predict_API']


def test_fixed_roster_retains_all_slots_after_setup_failure():
    store = fixture.Events()
    def failed_setup(slot, root, policy):
        raise RuntimeError('synthetic setup failure')
    result = seam.execute_roster([None, None], [None, None], failed_setup, None, store)
    assert result['fixed_denominator'] == 4
    assert [(r['slot'],r['root'],r['policy']) for r in result['rows']] == seam.ROSTER
    assert [r['status'] for r in result['rows']] == ['setup_failed']*4
    assert not result['physical_certified']
