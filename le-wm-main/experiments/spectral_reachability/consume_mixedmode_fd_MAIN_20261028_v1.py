"""Independent saved-only census and central difference arithmetic."""
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import audit_mixedmode_fd_MAIN_20261028_v1 as producer


def analyze(p, r, events):
    require = producer.require
    require(r['status'] == 'complete_diagnostic_only' and r['errors'] == []
            and r['model_before'] == r['model_after'] == p['expected_model_hash'], 'complete frozen diagnostic')
    require(r['new_AD'] == r['new_physics'] == 0 and r['P1_admission'] is False
            and r['scientific_threshold'] is None, 'diagnostic not admission')
    rows, comparisons = r['evaluations'], r['comparisons']
    require(len(rows) == 328 and len(comparisons) == 160 and len(events) == 656, 'complete denominator')
    require(r['resources'] == {k:dict(attempted=328, completed=328)
            for k in ('predictor_module', 'action_encoder', 'pred_proj')}, 'actual module counts')
    cursor = ci = 0
    summaries = []
    for point in p['points']:
        errors = {str(e): [] for e in producer.EPS}
        def take(coordinate, epsilon, sign):
            nonlocal cursor
            row = rows[cursor]
            logits = np.zeros(10, np.float32)
            if coordinate is not None:
                logits[coordinate] = np.float32(epsilon)*sign
            request = dict(order=cursor, point=point['index'], coordinate=coordinate,
                           epsilon=epsilon, sign=sign,
                           commands=np.tanh(logits).reshape(5,2).tolist())
            require({k:row[k] for k in request} == request, 'fixed complete request order/raw')
            require(events[2*cursor] == dict(kind='attempt', request=request)
                    and events[2*cursor+1] == dict(kind='returned', row=row), 'full actual attempt-return journal')
            value = np.asarray(row['terminal'], np.float64)
            require(value.shape == (192,) and np.isfinite(value).all(), 'full finite latent')
            cursor += 1
            return value
        require(take(None, None, 0).tolist() == point['nominal'], 'original nominal exactly reproduced')
        j = np.asarray(point['jacobian'], np.float64)
        for coordinate in range(10):
            for epsilon in producer.EPS:
                minus, plus = take(coordinate, epsilon, -1), take(coordinate, epsilon, 1)
                fd = (plus-minus)/(2*float(np.float32(epsilon)))
                reference = j[:,coordinate]
                absolute = float(np.linalg.norm(fd-reference))
                relative = absolute/max(float(np.linalg.norm(fd)), float(np.linalg.norm(reference)), 1e-12)
                c = comparisons[ci]; ci += 1
                require((c['point'],c['coordinate'],c['epsilon']) == (point['index'],coordinate,epsilon)
                    and c['fd'] == fd.tolist() and c['saved_AD'] == reference.tolist()
                    and c['absolute_l2'] == absolute and c['relative_l2'] == relative,
                    'independent full-vector central difference arithmetic')
                errors[str(epsilon)].append(relative)
        summaries.append(dict(point=point['index'], root_slot=point['root_slot'], condition=point['condition'],
            label=point['label'], by_epsilon={e:dict(median_relative_l2=float(np.median(v)),
                maximum_relative_l2=max(v)) for e,v in errors.items()}))
    require(cursor == 328 and ci == 160, 'complete used census')
    return summaries


def main():
    folder = producer.source.ROOT/'results/audit_mixedmode_fd_MAIN_20261028_v1_cuda_MAIN01'
    expected = 'b2371343e58536585f90a4708e380a36b81db7c63d70e05c61182834364cf4fc'
    require = producer.require
    require(producer.sha(folder/'report.json') == expected, 'actual report seal')
    r = json.loads((folder/'report.json').read_bytes())
    binding = json.loads((folder/'binding.json').read_bytes())
    require(binding['sha256'] == r['source_binding_sha256']
            and producer.sha(binding['prepared']) == binding['sha256'], 'exact prepare binding')
    p = json.loads(Path(binding['prepared']).read_bytes())
    producer.verify_bindings(p)
    require(producer.sha(folder/'events.jsonl') == r['journal_sha256'], 'full journal seal')
    events = [json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()]
    result = dict(schema='MAIN.mixedmode_FD_independent.v1', report_sha256=expected,
                  points=analyze(p,r,events), new_neural_calls=0, new_physics_calls=0,
                  P1_admission=False, scientific_threshold=None,
                  caveat='Eight predicted contexts, diagnostic coordinate checks only; no physical/RGB accuracy proof.')
    producer.verify_bindings(p)
    require(producer.sha(folder/'report.json') == expected, 'unchanged report')
    output = producer.source.ROOT/'results/consume_mixedmode_fd_MAIN_20261028_v1_MAIN01.json'
    producer.write_new(output, result)
    print(producer.canonical(dict(output=str(output), sha256=producer.sha(output), points=result['points'])).decode())


if __name__ == '__main__':
    main()
