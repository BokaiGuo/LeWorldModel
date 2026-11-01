from copy import deepcopy
import json
from pathlib import Path

import pytest
from experiments.spectral_reachability import p1_future_joint_saved_consumer_20261029_v1 as c

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def actual():
    run = ROOT/'results/p1_future_joint_candidate_host_20261029_v1_cuda_MAIN01'
    report = json.loads((run/'report.json').read_bytes())
    events = [json.loads(x) for x in (run/'events.jsonl').read_bytes().splitlines()]
    seal = json.loads(Path(report['prepared']).read_bytes())
    source = json.loads(Path(seal['source']).read_bytes())
    return report, events, source


def test_real_rows(actual):
    assert c.audit(*actual) == dict(candidates=144,stage_uphill=70,terminal_better=12,intersection=0)


@pytest.mark.parametrize('field', ['commands','order','baseline','initial','flags','start','event','counts','model'])
def test_reject_tampering(actual, field):
    report, events, source = deepcopy(actual)
    row = report['rows'][1]
    if field == 'commands': row['commands'][0][0] += .01
    elif field == 'order': row['iteration'] = 3
    elif field == 'baseline': row['baseline_Q'] += 1
    elif field == 'initial': row['initial_Q'] += 1
    elif field == 'flags': row['stage_uphill'] = not row['stage_uphill']
    elif field == 'start': report['rows'][9]['prefix_logits'][0] += .01; events[9]['row'] = deepcopy(report['rows'][9])
    elif field == 'event': events[1]['row']['terminal_Q'] += 1
    elif field == 'counts': report['calls']['reverse'] -= 1
    elif field == 'model': report['model_after']['original_state_dict'] = '0'*64
    if field in ('commands','order','baseline','initial','flags'): events[1]['row'] = deepcopy(row)
    with pytest.raises(Exception): c.audit(report,events,source)


def test_winner_baseline_and_event_hash_are_required(actual, tmp_path):
    report, events, source = deepcopy(actual)
    for row in report['rows'][:36]: row['baseline_Q'] += 1
    for i in range(36): events[i]['row'] = deepcopy(report['rows'][i])
    with pytest.raises(Exception): c.audit(report,events,source)
    with pytest.raises(Exception): c.decode(b'{"x":1,"x":2}')
