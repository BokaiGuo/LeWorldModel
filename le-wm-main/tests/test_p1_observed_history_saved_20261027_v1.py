"""Independent saved acquisition check: NumPy only, no simulator/model import."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/p1_observed_history_source_20261027_v1_main01'
OLD=ROOT/'results/p1_conditional_calibration2_main_20261027_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_observation_journal_and_arrays():
    assert sha(RUN/'sourcepack.json')=='0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546'
    r=json.loads((RUN/'sourcepack.json').read_text())
    assert r['status']=='observed_calibration_source_completed'
    assert r['first_error'] is None and r['secondary_errors']==[]
    assert sha(RUN/'journal.jsonl')==r['journal_sha256']
    events=[json.loads(x) for x in (RUN/'journal.jsonl').read_bytes().splitlines()]
    assert events[-1]['kind']=='run_finished'
    expected=dict(reset=4,public_step=20,explicit_render=6,set_state=8,goal_render=4,physics_step=208)
    for k,n in expected.items():
        assert r['counts'][k]==dict(attempted=n,completed=n)
        assert sum(e['kind']=='operation_started' and e.get('resource')==k for e in events)==n
        assert sum(e['kind']=='operation_completed' and e.get('resource')==k for e in events)==n
    assert len(r['rows'])==2 and all(row['status']=='completed' for row in r['rows'])
    for row in r['rows']:
        cid=row['candidate_id'];path=RUN/row['file']
        assert sha(path)==row['file_sha256']
        with np.load(path,allow_pickle=False) as a,np.load(OLD/f'source_{cid}.npz',allow_pickle=False) as b:
            assert np.array_equal(a['observed_public_step_indices'],[0,5,10])
            # Observed physical stationarity, NOT an assumption that zero action
            # always means zero dynamics. Same bytes do not create fresh roots.
            assert np.array_equal(a['rgb_history'],b['rgb_history'])
            assert np.array_equal(a['state'],b['state'])
            assert np.array_equal(a['observed_states'],np.repeat(b['state'][None],3,axis=0))
            assert np.array_equal(a['raw_action_history'],b['raw_action_history'])
            assert row['metadata']['root_sha256']==row['metadata']['parent_root_sha256']
            norm=r['prepare']['normalization']
            expected_tokens=((a['raw_action_history']-np.asarray(norm['mean'],np.float32))/np.asarray(norm['std'],np.float32)).reshape(2,10)
            assert np.array_equal(expected_tokens,a['normalized_action_history'])
            for idx,checkpoint in enumerate(row['checkpoints']):
                p=RUN/checkpoint['file'];assert sha(p)==checkpoint['sha256']
                with np.load(p,allow_pickle=False) as c:
                    assert np.array_equal(c['rgb'],a['rgb_history'][idx])
                    assert np.array_equal(c['state'],a['observed_states'][idx])
    assert r['P1_admission'] is False and r['global_freshness']=='unknown'
