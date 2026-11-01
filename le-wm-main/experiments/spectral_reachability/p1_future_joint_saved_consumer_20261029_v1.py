"""Independent saved-record consistency checks; never executes the model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_future_joint_candidate_host_20261029_v1 as producer

NAME = 'p1_future_joint_saved_consumer_20261029_v1'
need = producer.host.require
ROW_KEYS = {'slot','root_slot','condition','start','iteration','prefix_logits','suffix_logits',
            'commands','initial_Q','stage_Q','terminal_Q','baseline_Q','stage_uphill',
            'terminal_better','original_vreach_gate','P1_admission'}
REPORT_KEYS = {'schema','status','prepared','prepare_sha256','contract','rows','errors','calls',
               'model_before','model_after','device','known_roots','freshness',
               'scientific_admission','P1_admission','physical_release'}


def read(path, limit):
    p = Path(path)
    need(p.is_absolute() and p == p.resolve() and p.is_file() and not p.is_symlink(), 'canonical regular input')
    data = p.read_bytes(); need(len(data) <= limit, 'bounded input')
    return data


def decode(data):
    def pairs(items):
        d = {}
        for key, value in items:
            need(key not in d, 'duplicate JSON key'); d[key] = value
        return d
    def invalid(_): raise ValueError('nonfinite JSON')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def digest(data): return hashlib.sha256(data).hexdigest()


def finite(x):
    return type(x) in (int,float) and math.isfinite(x) and x >= 0


def audit(report, events, source):
    need(type(report) is dict and set(report) == REPORT_KEYS and
         report['schema'] == producer.NAME+'.report' and report['status'] == 'complete_model_only' and
         report['contract'] == producer.contract() and report['errors'] == [] and report['known_roots'] is True and
         all(report[x] is False for x in ('freshness','scientific_admission','P1_admission','physical_release')),
         'complete report contract')
    need(report['calls'] == dict(predict=296,action_encoder=296,reverse=128), 'reported call counts')
    before = report['model_before']; after = report['model_after']
    need(type(before) is dict and before == after and before.get('original_state_dict') == source['expected_model_hash'], 'reported frozen model hash')
    requests = source['requests']; rows = report['rows']
    need(type(requests) is list and len(requests) == 4 and type(rows) is list and len(rows) == 144 and
         type(events) is list and len(events) == 144, 'fixed roster')
    counts = dict(candidates=0,stage_uphill=0,terminal_better=0,intersection=0)
    for slot, q in enumerate(requests):
        task, wx, wy = producer.parent.request(q)
        need(q['slot'] == slot and np.linalg.norm(wy) <= producer.host.F.CAP, 'original request and winner domain')
        initial = producer.host.F.cost(task['goal'], task['root_context'])
        baseline = rows[slot*36]['baseline_Q']
        need(finite(baseline), 'baseline scalar')
        for start, startx in enumerate(producer.starts(wx)):
            for iteration in range(9):
                index = slot*36+start*9+iteration; row = rows[index]; event = events[index]
                need(type(row) is dict and set(row) == ROW_KEYS and type(event) is dict and
                     set(event) == {'kind','row'} and event['kind'] == 'candidate' and
                     producer.host.raw(event['row']) == producer.host.raw(row), 'event/report candidate equality')
                need(type(row['slot']) is int and type(row['root_slot']) is int and type(row['start']) is int and
                     type(row['iteration']) is int and (row['slot'],row['root_slot'],row['condition'],row['start'],row['iteration']) ==
                     (slot,q['root_slot'],q['condition'],start,iteration), 'candidate order and root identity')
                x = np.asarray(row['prefix_logits'],dtype=np.float64); y = np.asarray(row['suffix_logits'],dtype=np.float64)
                need(x.shape == y.shape == (10,) and np.isfinite(x).all() and np.isfinite(y).all() and
                     np.linalg.norm(x) <= producer.host.F.CAP and np.linalg.norm(y) <= producer.host.F.CAP, 'packet chart domains')
                commands = np.asarray(row['commands'],dtype=np.float64)
                chart = np.concatenate((producer.host.F.chart(x),producer.host.F.chart(y))).astype(np.float64)
                need(commands.shape == (10,2) and np.array_equal(commands,chart), 'literal float32 chart commands')
                if iteration == 0:
                    need(np.array_equal(x,startx) and np.array_equal(y,wy), 'fixed start parameters')
                need(all(finite(row[k]) for k in ('initial_Q','stage_Q','terminal_Q','baseline_Q')) and
                     row['initial_Q'] == initial and row['baseline_Q'] == baseline, 'fixed root objective and slot baseline')
                need(type(row['stage_uphill']) is bool and row['stage_uphill'] == (row['stage_Q'] > initial) and
                     type(row['terminal_better']) is bool and row['terminal_better'] == (row['terminal_Q'] < baseline) and
                     row['original_vreach_gate'] == 'not_evaluated' and row['P1_admission'] is False, 'recomputed strict flags')
                if start == 0 and iteration == 0:
                    need(producer.host.F.bits(commands) == q['winner_raw_bits'] and row['terminal_Q'] == baseline,
                         'recorded same-action winner baseline join')
                counts['candidates'] += 1
                counts['stage_uphill'] += int(row['stage_uphill']); counts['terminal_better'] += int(row['terminal_better'])
                counts['intersection'] += int(row['stage_uphill'] and row['terminal_better'])
    return counts


def consume(output, report_sha, events_sha):
    p = Path(output); need(p.is_absolute() and p == p.resolve() and p.is_dir() and not p.is_symlink(), 'canonical run directory')
    need({x.name for x in p.iterdir()} == {'report.json','events.jsonl'}, 'exact run inventory')
    rb = read(p/'report.json', 16*1024**2); eb = read(p/'events.jsonl', 16*1024**2)
    need(digest(rb) == report_sha and digest(eb) == events_sha, 'externally supplied report and event hashes')
    report = decode(rb); need(eb.endswith(b'\n') and b'\n\n' not in eb, 'complete event lines')
    events = [decode(line) for line in eb.splitlines()]
    source, _seal = producer.verify(report['prepared'],report['prepare_sha256'])
    counts = audit(report, events, source)
    need(digest(read(p/'report.json',16*1024**2)) == report_sha and
         digest(read(p/'events.jsonl',16*1024**2)) == events_sha, 'stable saved bytes')
    return dict(schema=NAME+'.consume', verified_saved_consistency=True, report_sha256=report_sha,
        events_sha256=events_sha, prepare_sha256=report['prepare_sha256'], counts=counts,
        source_authority_reverified=True, gradient_recomputed=False, neural_predictions_recomputed=False,
        baseline_neural_value_recomputed=False, actual_native_counts_independently_proved=False,
        event_hash_recorded_in_report=False, physical_release=False, P1_admission=False, freshness=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--report-sha256',required=True); p.add_argument('--events-sha256',required=True)
    a = p.parse_args(); print(json.dumps(consume(a.output,a.report_sha256,a.events_sha256),sort_keys=True))


if __name__ == '__main__': main()
