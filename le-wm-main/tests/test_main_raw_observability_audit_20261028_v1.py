import pytest
from experiments.spectral_reachability import main_raw_observability_audit_20261028_v1 as m


def case(gm=1., ga=1., pixels=False, latent=False):
    return dict(model_scalar_fd=gm, actual_scalar_fd=ga,
                RGB_pixels_identical=pixels, actual_latents_identical=latent)


@pytest.mark.parametrize('c,label', [
    (case(ga=0, pixels=True, latent=True), 'unobservable_RGB'),
    (case(ga=0, latent=True), 'encoder_invariant'),
    (case(ga=0), 'actual_goal_tie'),
    (case(gm=0), 'model_goal_tie'),
    (case(gm=-1, ga=-2), 'same_direction'),
    (case(gm=-1), 'opposite_direction'),
])
def test_exact_partition(c, label):
    assert m.classify(c) == label


@pytest.mark.parametrize('c', [case(pixels=True), case(ga=0, pixels=True),
                              case(latent=True), case(gm=float('nan'))])
def test_inconsistent_or_nonfinite_evidence_rejected(c):
    with pytest.raises(ValueError):
        m.classify(c)


def test_small_pose_change_is_not_velocity_only_or_zero():
    a = [0.]*7
    b = a.copy()
    b[0] = 1e-12
    b[5] = 1
    r = m.state_response(a, b)
    assert r['pose_changed'] and r['velocity_changed'] and not r['velocity_only']


def test_zero_real_responses_cannot_be_counted_as_direction_agreement():
    rows = [dict(label='unobservable_RGB', state=dict(pose_changed=True, velocity_only=False))]*10
    r = m.summarize(rows)
    assert r['informative_direction_pairs'] == 0
    assert r['opposite_fraction_informative'] is None
    assert r['labels']['same_direction'] == 0 and r['invisible_pose_changes'] == 10


def test_velocity_only_is_distinct_from_pose_change():
    a = [0.]*7
    b = a.copy()
    b[6] = 2
    assert m.state_response(a, b)['velocity_only']
