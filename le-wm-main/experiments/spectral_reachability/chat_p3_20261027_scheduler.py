"""Predeclared rank-reserve queues; no neural or physical execution."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ARMS = ('U', 'D_rank', 'S_rank', 'N')
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/moving_history_budget_scheduler_cuda_20261027_v1_cuda_main01'
OLD_SHA = '710892008bbd47aa188a9b9391972976d3a94d2a0067e46911ff37f05be8369b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def allocation(arm, weights=None):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if arm in ('U', 'N'):
        if weights is not None:
            raise ValueError('uniform has no signal input')
        return [10, 10]
    if weights is None or len(weights) != 2 or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('two finite nonnegative weights required')
    if weights[0] == weights[1]:
        return [10, 10]
    high = 0 if weights[0] > weights[1] else 1
    if arm == 'S_rank':
        high = 1 - high
    out = [5, 5]
    out[high] += 10
    return out


def queue(blocks):
    if len(blocks) != 2 or any(type(b) is not int or not 5 <= b <= 15 for b in blocks) or sum(blocks) != 20:
        raise ValueError('20 blocks, minimum5, maximum15')
    used = [0, 0]
    out = []
    while used != blocks:
        for slot in (0, 1):
            if used[slot] < blocks[slot]:
                for index in range(used[slot]*3, used[slot]*3+3):
                    out.append({'slot': slot, 'index': index})
                used[slot] += 1
    return out


def execute(jobs, evaluator):
    """Fail-stop queue consumer. Callback owns actual resource metering.

    This is NOT a CUDA executor. No inferred predictor samples or FLOPs.
    """
    records = []
    for job in jobs:
        row = dict(job, status='attempted')
        records.append(row)
        try:
            value = evaluator(job['slot'], job['index'])
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite scalar cost required')
            row.update(status='complete', cost=value)
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            break
    return {'records': records, 'blocked': jobs[len(records):],
            'complete': len(records) == len(jobs) and all(r['status'] == 'complete' for r in records)}


def consume_saved(jobs, candidates):
    lookup = {(r['slot'], r['index']): r['cost'] for r in candidates}
    if len(lookup) != len(candidates):
        raise ValueError('duplicate saved candidate')
    # Missing new tails remain explicit; never substitute old winners.
    missing = [j for j in jobs if (j['slot'], j['index']) not in lookup]
    if missing:
        return {'status': 'missing_scores', 'missing': missing, 'paired_gain': None}
    return execute(jobs, lambda s, i: lookup[s, i])


def build():
    from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as old
    verified = old.consume(OLD, OLD_SHA)
    with gzip.open(OLD / 'panel.json.gz', 'rt') as f:
        panel = json.load(f)
    weights = [r['signal']['w'] for r in panel['signals']]
    with gzip.open(OLD / 'banks.json.gz', 'rt') as f:
        banks = json.load(f)
    plans = {}
    for arm in ARMS:
        blocks = allocation(arm) if arm in ('U', 'N') else allocation(arm, weights)
        jobs = queue(blocks)
        plans[arm] = {'blocks': blocks, 'jobs': jobs,
                      'candidate_sha256': [hashlib.sha256(canonical(banks[j['slot']][j['index']])).hexdigest() for j in jobs]}
    return {'schema': 'chat_p3_20261027.rank_reserve.queue.v1', 'old_report_sha256': OLD_SHA,
            'old_consumer_status': verified['status'], 'weights': weights,
            'old_allocations': panel['allocations'], 'plans': plans,
            'bindings': {str(p): digest(p) for p in [Path(__file__), ROOT.parent/'docs/chat_p3_20261027_prereg.md', OLD/'banks.json.gz', OLD/'panel.json.gz']},
            'P1_admission': False, 'P3_admission': False, 'freshness': False,
            'new_predict_API': 0, 'new_predictor_samples': 0, 'new_encoder': 0,
            'new_AD': 0, 'new_model_SVD': 0, 'equal_FLOPs': False,
            'historical_resources': json.loads((OLD/'report.json').read_text())['resources']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['seal', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    value = build()
    if args.action == 'seal':
        if not args.output.name.startswith('chat_p3_20261027_') or args.output.parent.resolve() != (ROOT/'results').resolve():
            raise ValueError('exclusive flat result prefix required')
        with args.output.open('xb') as f:
            f.write(canonical(value) + b'\n')
    else:
        if not args.sha256 or digest(args.output) != args.sha256 or json.loads(args.output.read_text()) != value:
            raise ValueError('seal/source/input drift')
    print(json.dumps({'status': args.action, 'sha256': digest(args.output), 'blocks': {a: value['plans'][a]['blocks'] for a in ARMS}}))


if __name__ == '__main__':
    main()
