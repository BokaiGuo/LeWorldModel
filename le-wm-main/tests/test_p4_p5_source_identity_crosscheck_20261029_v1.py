from copy import deepcopy
import json

import pytest
from experiments.spectral_reachability import p4_p5_source_identity_crosscheck_20261029_v1 as c


@pytest.fixture
def case():
    s=dict(id='A',path='/original/report.json',sha256='a'*64)
    row=dict(root_id=0,root_sha256='b'*64,goal_rgb_sha256='c'*64,history_rgb_sha256='d'*64,
             outcome={'score':999})
    data=json.dumps(dict(calibration_roots=[dict(row,root_id=i) for i in range(4)],outcome={'secret':'ignored'})).encode()
    got=c.extract(data,s)
    old=dict(source_id='A',report=dict(path=s['path'],sha256=s['sha256']),selected_rows=[])
    for entry in got:
        old['selected_rows'].append(dict(row_pointer=entry['row_pointer'], fields=deepcopy(list(entry['fields'].values())),
            ordinal_is_not_episode_id=True,source_episode_identity=None,reset_attempt_identity=None,
            physical_state_identity=None,history_hash_unit='whole_stack_not_individual_frames',missing=list(c.MISSING)))
    return s,data,got,old


def test_extract_join_excludes_outcome(case):
    s,data,got,old=case
    assert len(got)==4 and 'score' not in str(got) and 'secret' not in str(got)
    c.join(s,old,got)


@pytest.mark.parametrize('kind',['value','pointer','report','ordinal','missing','duplicate','rowcount'])
def test_reject_mismatch(case,kind):
    s,data,got,old=deepcopy(case)
    row=old['selected_rows'][0]
    if kind=='value': row['fields'][0]['value']=77
    elif kind=='pointer': row['fields'][0]['pointer']='/calibration_roots/1/root_id'
    elif kind=='report': old['report']['sha256']='f'*64
    elif kind=='ordinal': row['source_episode_identity']='invented'
    elif kind=='missing': row['missing']=[]
    elif kind=='duplicate': row['fields'].append(deepcopy(row['fields'][0]))
    elif kind=='rowcount': old['selected_rows'].pop()
    with pytest.raises(Exception): c.join(s,old,got)


def test_missing_required_identity_is_rejected(case):
    s,data,_,_=case
    v=json.loads(data); del v['calibration_roots'][0]['root_sha256']
    with pytest.raises(Exception): c.extract(json.dumps(v).encode(),s)
