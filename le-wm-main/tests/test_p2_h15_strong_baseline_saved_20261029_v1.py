import copy
import pytest
from experiments.spectral_reachability import p2_h15_strong_baseline_saved_20261029_v1 as m


def row(arm, n=15):
    return dict(arm=arm, root_slot=0, condition='off', initial_exact=True, task_id='t',
                root_sha256='r', root_state=[1], goal=[2], goal_pose=[3], goal_pixel_sha256='g',
                own_returned=n, status='native_stopped' if n < 15 else 'horizon_completed',
                responses=[dict(step=i, native_metrics=dict(native_distance=float(i),
                    object_position_error=float(i), wrapped_object_angle_error=float(i), native_success=False),
                    terminated=i==n and n<15, truncated=False) for i in range(1,n+1)])


def test_pair_no_padding_and_difference():
    p=m.pair(row('h15_sra'), row('bounded_native_CEM_H3_actualmean',10))
    assert len(p['common_steps'])==10
    assert p['timeline'][9]['delta_native_distance']==0
    assert p['timeline'][10]['control'] is None and p['timeline'][10]['delta_native_distance'] is None


@pytest.mark.parametrize('field', ['goal','root_state','goal_pixel_sha256','task_id'])
def test_mismatch_fails(field):
    b=row('grasp'); b[field]='tampered'
    with pytest.raises(ValueError): m.pair(row('h15_sra'),b)


def test_bad_timeline_rejected():
    b=row('grasp'); b['responses'][2]['step']=4
    with pytest.raises(ValueError): m.pair(row('h15_sra'),b)


def test_pinned_saved_sources_real():
    report=m.build()
    assert len(report['pairs'])==4
    p=next(x for x in report['pairs'] if x['root_slot']==0 and x['control']=='bounded_native_CEM_H3_actualmean')
    assert p['control_last_step']==10 and p['timeline'][14]['control'] is None
    assert report['equal_compute'] is False and report['fresh_roots']==0


def test_digest_fails(tmp_path):
    p=tmp_path/'receipt.json'; p.write_text('{}')
    with pytest.raises(ValueError): m.read(p,'0'*64)
