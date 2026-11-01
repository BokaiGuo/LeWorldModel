"""Saved-only independent module-bridge and fixed40 ranking audit; no inference."""
import argparse
from collections import Counter
import gzip
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'moving_capability_direction_cuda_MAIN_20261027_v1'
CAPS = dict(encoder=40, projector=40, action_encoder=80, predictor=80, pred_proj=80)
NATIVE_PATH = ROOT/'results/moving_overlay_capability_probe_analysis_20261027_v1_MAIN01.json'
NATIVE_SHA = '19661bb010662dcfc7c9397723ecb085080fe18a7de069bdaac1f7c59e0e7557'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    def constant(value): raise ValueError('nonfinite constant '+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def aligned(prediction, actual, goal):
    p, a, g = [np.asarray(v, np.float64) for v in (prediction, actual, goal)]
    require(p.shape == a.shape == g.shape == (192,) and all(np.isfinite(v).all() for v in (p, a, g)), 'finite native192')
    model_q, real_q = float(np.mean((p-g)**2)), float(np.mean((a-g)**2))
    gap = a-p
    mse = float(np.mean(gap**2)); cross = float(2*np.mean((p-g)*gap))
    np.testing.assert_allclose(real_q-model_q, mse+cross, rtol=1e-12, atol=1e-12)
    return dict(predicted_Q=model_q, actual_Q=real_q, prediction_MSE=mse, cross_term=cross)


def rank_pairs(rows, path):
    result = []
    for i, j in itertools.combinations(range(5), 2):
        actual = rows[i]['actual_Q']-rows[j]['actual_Q']
        predicted = rows[i][path]['predicted_Q']-rows[j][path]['predicted_Q']
        result.append(dict(left=rows[i]['probe'], right=rows[j]['probe'], actual_difference=actual,
                           predicted_difference=predicted, exact_sign_agreement=bool(np.sign(actual) == np.sign(predicted)),
                           actual_tie=actual == 0., prediction_tie=predicted == 0.))
    return result


def consume(directory, report_sha):
    d = Path(directory).resolve()
    require(sha(d/'report.json') == report_sha, 'external report SHA')
    r = read(d/'report.json')
    require(r['schema'] == NAME+'.report' and r['status'] == 'predictions_and_encodings_sealed' and r['errors'] == [] and r['P1_admission'] is False, 'complete raw evidence only')
    require(r['model_hash_before'] == r['model_hash_after'] == '4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546', 'original frozen model')
    require(set(f.name for f in d.iterdir()) == set(r['artifact_sha256']) | {'report.json'}, 'exact output census')
    for name, h in r['artifact_sha256'].items():
        require(Path(name).name == name and sha(d/name) == h, 'sealed artifact')
    require(sha(r['prepared']) == r['prepare_sha256'], 'external prepare SHA')
    p = read(r['prepared'])
    require(p['contract'] == r['contract'] and r['contract']['caps'] == CAPS and len(p['rows']) == len(r['rows']) == 40, 'fixed contract and denominator')
    require(read(d/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'execution binding')
    for path, h in p['bindings'].items(): require(sha(path) == h, 'prepared dependency SHA')
    def array(receipt):
        require(Path(receipt['file']).name == receipt['file'], 'flat array')
        data = (d/receipt['file']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == receipt['file_sha256'] and len(data) < 4*1024**2, 'array file seal')
        expanded = gzip.decompress(data); require(len(expanded) < 4*1024**2, 'bounded expanded array')
        f = BytesIO(expanded); a = np.load(f, allow_pickle=False)
        require(f.tell() == len(expanded) and list(a.shape) == receipt['shape'] and str(a.dtype) == receipt['dtype'] and
                a.dtype in (np.float32, np.uint8) and np.isfinite(a).all() and
                hashlib.sha256(a.tobytes()).hexdigest() == receipt['array_sha256'], 'native array bytes')
        return a
    ledger = r['ledger']
    require(len(ledger) == 320 and [e['index'] for e in ledger] == list(range(320)) and all(e['completed'] is True for e in ledger), 'all actual module returns')
    require(dict(Counter(e['resource'] for e in ledger)) == CAPS, 'module census exact')
    for e in ledger:
        for receipt in e['inputs']+[e['output']]: array(receipt)
    events = [json.loads(x) for x in gzip.decompress((d/'events.jsonl.gz').read_bytes()).splitlines()]
    attempts = [x['entry'] for x in events if x['kind'] == 'module_attempt']
    returns = [x['entry'] for x in events if x['kind'] == 'module_return']
    require(len(attempts) == len(returns) == 320 and [e['index'] for e in attempts] == list(range(320)), 'all durable attempts')
    require(sorted(returns, key=lambda e:e['index']) == ledger, 'all durable returns')
    require(all(a == {k:v for k,v in e.items() if k not in ('inputs','output','completed')} | {'completed':False}
                for a,e in zip(attempts,ledger,strict=True)), 'attempt/return identity')
    require([x['row'] for x in events if x['kind'] == 'row_sealed'] == r['rows'], 'all raw row seals before scoring')
    require(sha(NATIVE_PATH) == NATIVE_SHA, 'fixed native direction evidence')
    native = read(NATIVE_PATH)
    result_rows = []
    for i, (source, raw) in enumerate(zip(p['rows'], r['rows'], strict=True)):
        require(source['index'] == raw['index'] == i and raw['status'] == 'sealed', 'fixed row identity')
        allentries = [e for e in ledger if e['scope']['index'] == i]
        enc = [e for e in allentries if e['scope']['phase'] == 'encode']
        require([e['resource'] for e in enc] == ['encoder', 'projector'], 'own visual chain')
        pre, ein, hidden, pin, pout, actual = [array(x) for x in (raw['preprocessed'], enc[0]['inputs'][0], enc[0]['output'], enc[1]['inputs'][0], enc[1]['output'], raw['actual'])]
        require(pre.shape == (1,1,3,224,224) and ein.shape == (1,3,224,224) and hidden.shape == (1,257,192) and pin.shape == pout.shape == (1,192) and actual.shape == (192,), 'full patch14/CLS/native shapes')
        require(np.array_equal(pre.reshape(ein.shape), ein) and np.array_equal(hidden[:,0], pin) and np.array_equal(pout[0], actual), 'visual exact bridges')
        pixels = array(raw['pixels'])
        require(pixels.shape == (224,224,3) and hashlib.sha256(pixels.tobytes()).hexdigest() == source['image']['pixel_sha256'] and
                sha(d/f'i{i:02}_original.png') == source['image']['file_sha256'] == sha(source['image']['path']), 'own original PNG and pixels')
        row = {k:source[k] for k in ('index','anchor','root_slot','condition','arm','probe','task_id','selection_digest')}
        goal = np.asarray(source['goal'], np.float64)
        row['actual_Q'] = float(np.mean((actual.astype(np.float64)-goal)**2))
        for path in ('teacher', 'open'):
            entries = [e for e in allentries if e['scope'].get('path') == path]
            require([e['resource'] for e in entries] == ['action_encoder', 'predictor', 'pred_proj'], 'own exact prediction chain')
            ae, pred, proj = entries
            ain, aout = array(ae['inputs'][0]), array(ae['output'])
            hin, actions = [array(x) for x in pred['inputs']]
            predicted, projin, projout, saved = [array(x) for x in (pred['output'], proj['inputs'][0], proj['output'], raw['predictions'][path])]
            require(ain.shape == (1,3,10) and aout.shape == hin.shape == actions.shape == predicted.shape == saved.shape == (1,3,192) and projin.shape == projout.shape == (3,192), 'native module shape chain')
            require(np.array_equal(ain[0], np.asarray(source['tokens'],np.float32)) and np.array_equal(hin[0], np.asarray(source['contexts'][path],np.float32)) and np.array_equal(aout,actions) and np.array_equal(predicted.reshape(3,192),projin) and np.array_equal(projout.reshape(1,3,192),saved), 'prediction complete input/output bridges')
            row[path] = aligned(saved[0,-1], actual, goal)
        nr = native['rows'][source['anchor']]
        require((nr['root_slot'],nr['condition'],nr['arm'],nr['selection_digest']) ==
                (source['root_slot'],source['condition'],source['arm'],source['selection_digest']), 'native/latent same anchor')
        row['native_object_task_projection'] = nr['spaces']['object_xy']['finite_responses'][i%5]['task_projection']
        result_rows.append(row)
    groups = []
    for anchor in range(8):
        group = result_rows[5*anchor:5*anchor+5]
        require([x['probe'] for x in group] == ['zero','+x','-x','+y','-y'], 'all probes retained')
        for row in group:
            row['actual_minus_zero_Q'] = row['actual_Q']-group[0]['actual_Q']
            for path in ('teacher','open'):
                delta = row[path]['predicted_Q']-group[0][path]['predicted_Q']
                row[path]['predicted_minus_zero_Q'] = delta
                row[path]['zero_relative_exact_sign_agreement'] = bool(np.sign(delta) == np.sign(row['actual_minus_zero_Q']))
        groups.append(dict(anchor=anchor,root_slot=group[0]['root_slot'],condition=group[0]['condition'],arm=group[0]['arm'],
            comparisons={path:rank_pairs(group,path) for path in ('teacher','open')}))
    for path,h in p['bindings'].items(): require(sha(path) == h, 'final prepared dependency seal')
    require(sha(d/'report.json') == report_sha and sha(NATIVE_PATH) == NATIVE_SHA, 'final report seals')
    return dict(status='saved_bridges_and_rankings_verified', source_report_sha256=report_sha,
        native_analysis_sha256=NATIVE_SHA, module_counts=CAPS, module_entries=320, rows=result_rows, groups=groups,
        fixed_roots=2, fresh_roots=0, fixed_branches=40, new_neural_calls=0, new_physics_calls=0, P1_admission=False,
        limitations=['known calibration roots', 'pairwise comparisons correlated, not independent samples',
                     'latent ranking and native task direction are different axes', 'no action reselection or model training'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','report-sha','output'): parser.add_argument('--'+name,required=True)
    a = parser.parse_args(); result = consume(a.input,a.report_sha)
    with Path(a.output).open('xb') as f:
        f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode()); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],sha256=sha(a.output),rows=40,modules=320)))


if __name__ == '__main__': main()
