"""Pure saved-latent arithmetic, not CPU neural inference."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import p1_witness_saved24_audit_20261027_v1 as a


@pytest.fixture(scope='module')
def saved():
    return (a.read(a.PREPARED/'protocol.json')['frames'],
        [a.json.loads(line) for line in (a.RUN/'image.events.jsonl').read_text().splitlines()])


def test_actual_raw_arrays_and_initials(saved):
    rows,initials=a.recompute(*saved)
    assert len(rows)==24 and len(initials)==2
    original=a.keyed(a.read(a.RUN/'analysis.json')['rows'])
    for row in rows:
        peer=original[row['root_sha256'],row['arm'],row['step']]
        for field in ('predicted_Q_mse','actual_Q_mse','native_prediction_mse'):
            assert row[field]==pytest.approx(peer[field],abs=2e-12,rel=0)
    assert all(x['initial_pixel_identical'] for x in initials)


@pytest.mark.parametrize('damage',['missing_frame','missing_event','pixel','scope','attempt','counts','nonfinite','shape','duplicate','initial_pixel'])
def test_reject_corruption(saved,damage):
    frames,events=deepcopy(saved)
    if damage=='missing_frame':frames.pop()
    if damage=='missing_event':events.pop()
    if damage=='pixel':events[3]['rgb_sha256']='tampered'
    if damage=='scope':events[0]['root_sha256']='wrong'
    if damage=='attempt':events[1]['samples']=2
    if damage=='counts':events[3]['counts']['encoder']=0
    if damage=='nonfinite':events[3]['latent'][0]=float('nan')
    if damage=='shape':events[3]['latent'].pop()
    if damage=='duplicate':frames[3]=deepcopy(frames[0])
    if damage=='initial_pixel':
        frames[3]['pixel_sha256']='different';events[18]['rgb_sha256']='different'
    with pytest.raises(ValueError): a.recompute(frames,events)


def test_metric_uses_original_q_not_unprojected(saved):
    frames,events=deepcopy(saved)
    frames[1]['q']=(2*a.np.asarray(frames[1]['q'])).tolist()
    before,_=a.recompute(*saved);after,_=a.recompute(frames,events)
    assert after[1]['actual_Q_mse']==pytest.approx(4*before[1]['actual_Q_mse'])
    assert after[1]['predicted_Q_mse']==pytest.approx(4*before[1]['predicted_Q_mse'])
    assert after[1]['native_prediction_mse']==before[1]['native_prediction_mse']


def test_stable_analysis_join_rejects_duplicates(saved):
    rows,_=a.recompute(*saved)
    assert a.keyed(rows)==a.keyed(list(reversed(rows)))
    with pytest.raises(ValueError):a.keyed(rows+[rows[0]])


def test_actual_full_saved_binding():
    r=a.audit()
    assert r['physical_mapping_checks']==24 and r['P1_admission'] is False
    assert r['new_neural_calls']==r['new_physical_steps']==0
