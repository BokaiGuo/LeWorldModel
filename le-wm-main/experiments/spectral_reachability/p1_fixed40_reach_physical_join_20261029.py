"""Join fixed conditional reach with original executed suffix outcomes, without selection."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import p1_fixed40_conditional_reach_consume_20261029 as reach
from experiments.spectral_reachability import p1_value_source_export_20261029 as fixed

ROOT = reach.host.ROOT
REACH = ROOT/'results/p1_fixed40_conditional_reach_host_20261029_MAIN01.json'
REACH_SHA = '9281d3353c8a36bcf7d836614a3142afbc73321064efb68edd52ce2533e9439c'


def join(panel, original):
    if len(original['rows']) != 40 or len(panel['rows']) != 8:
        raise ValueError('fixed denominators required')
    rows = []
    for anchor in panel['rows']:
        index = anchor['anchor_index']
        candidates = original['rows'][5*index:5*index+5]
        if (index != len(rows) or len(candidates) != 5 or
            any(c['index'] != 5*index+j or c['identity']['anchor_id'] != anchor['anchor_id'] or
                c['identity']['task_id'] != anchor['task_id'] for j, c in enumerate(candidates))):
            raise ValueError('anchor identity mismatch')
        rows.append(dict(anchor_index=index, anchor_id=anchor['anchor_id'], task_id=anchor['task_id'],
            conditional_reach={k:anchor['cases'][k]['solve']['own_reach_value'] for k in ('predicted','observed')},
            candidates=[dict(index=c['index'], suffix_hex=c['suffix_hex'],
                predicted_terminal=c['suffix']['value'], actual_terminal=c['observed_terminal']['value'],
                physical=c['physical']['value']) for c in candidates]))
    return dict(schema='p1_fixed40_reach_physical_join_20261029', rows=rows, known_roots=2,
        fixed_candidates=40, neural_calls=0, physical_calls=0, P1_admission=False,
        scope='same action identity, conditional model output vs prior actual execution; not independent or prospective')


def build():
    reach.verify(REACH, REACH_SHA)
    return join(json.loads(REACH.read_bytes()), fixed.export_fixed())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.is_absolute() or a.output.parent != ROOT/'results' or a.output.exists():
        raise ValueError('exclusive absolute results file required')
    print('Join fixed40 conditional scores with original real outcomes; no winner selection.', flush=True)
    payload = json.dumps(build(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with a.output.open('xb') as f:
        f.write(payload)
    print(json.dumps(dict(path=str(a.output), sha256=hashlib.sha256(payload).hexdigest())))


if __name__ == '__main__':
    main()
