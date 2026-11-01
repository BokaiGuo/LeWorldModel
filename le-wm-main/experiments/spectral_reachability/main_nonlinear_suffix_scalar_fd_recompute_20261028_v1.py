"""Independent native loss-buffer decoding and scalar FD arithmetic, saved only."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import numpy as np

NAME = 'main_nonlinear_suffix_scalar_fd_recompute_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = 'a56e403c0d5b7a904b502c8044aaa4c575b667631fb98992716f1e13acd79639'
STEPS = (.01, .001, .0001)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss(blob):
    require(set(blob) == {'shape', 'dtype', 'sha256', 'data'}
            and blob['shape'] == [1] and blob['dtype'] == '<f4', 'native scalar loss layout')
    data = base64.b64decode(blob['data'], validate=True)
    require(len(data) == 4 and hashlib.sha256(data).hexdigest() == blob['sha256'], 'native loss bytes')
    value = float(np.frombuffer(data, dtype='<f4')[0])
    require(np.isfinite(value) and value >= 0, 'finite nonnegative native loss')
    return value


def vector_metrics(values, gradient):
    require(len(values) == 10 and np.asarray(gradient).shape == (10,), 'all ten raw coordinates')
    if any(x is None for x in values):
        return None
    v, g = np.asarray(values, float), np.asarray(gradient, float)
    require(np.isfinite(v).all() and np.isfinite(g).all(), 'finite full vectors')
    e, ng, nv = v-g, float(np.linalg.norm(g)), float(np.linalg.norm(v))
    return dict(L2_error=float(np.linalg.norm(e)), Linf_error=float(np.max(np.abs(e))),
                relative_L2=None if ng == 0 else float(np.linalg.norm(e))/ng,
                cosine=None if ng*nv == 0 else float(v@g)/(ng*nv), gradient_L2=ng, FD_L2=nv)


def build(directory, report_sha, receipt, receipt_sha):
    from experiments.spectral_reachability import agent_nonlinear_suffix_scalar_fd_20261028_v1 as source
    require(sha(source.__file__) == SOURCE_SHA, 'frozen FD implementation')
    directory = Path(directory).resolve(); receipt = Path(receipt).resolve()
    require(sha(directory/'report.json') == report_sha and sha(receipt) == receipt_sha, 'external actual seals')
    accepted = source.consume(directory, report_sha)
    require(accepted['complete'] is True and accepted == json.loads(receipt.read_bytes()), 'entire actual strict rebuilt')
    report = json.loads((directory/'report.json').read_bytes())
    prep = json.loads(Path(report['prepared']).read_bytes())
    require(len(report['rows']) == len(prep['points']) == len(accepted['panels']) == 4, 'all four conditions')
    rows = []
    for i, (point, row, panel) in enumerate(zip(prep['points'], report['rows'], accepted['panels'], strict=True)):
        require(point['index'] == row['index'] == panel['index'] == i
                and len(row['evaluations']) == 60, 'fixed point and sixty signed requests')
        g = np.asarray(point['raw_gradient'], float).reshape(10)
        steps = []
        for k, h in enumerate(STEPS):
            values = []
            for c in range(10):
                minus, plus = row['evaluations'][k*20+2*c:k*20+2*c+2]
                require([(e['request']['h'], e['request']['coordinate'], e['request']['sign'])
                         for e in (minus, plus)] == [(h, c, -1), (h, c, 1)], 'exact nominal signed schedule')
                fd = None
                if minus['status'] == plus['status'] == 'recorded':
                    fd = (loss(plus['objective']['arrays']['loss'])-loss(minus['objective']['arrays']['loss']))/(2*h)
                require(fd == panel['summary']['steps'][k]['pairs'][c]['nominal_FD'], 'independent scalar finite difference')
                values.append(fd)
            metrics = vector_metrics(values, g)
            require(metrics == panel['summary']['steps'][k]['metrics_nominal'], 'independent whole-vector metric')
            steps.append(dict(h=h, nominal_FD=values, metrics=metrics))
        rows.append(dict(index=i, root_slot=point['root_slot'], condition=point['condition'],
                         original_prefix_index=point['prefix_index'], steps=steps))
    require(sha(source.__file__) == SOURCE_SHA and sha(directory/'report.json') == report_sha
            and sha(receipt) == receipt_sha, 'inputs unchanged')
    return dict(schema=NAME, rows=rows, signed_denominator=240, pair_denominator=120,
                known_roots=2, full_upstream_rebuilt_here=True, new_neural_calls=0, new_physics_calls=0,
                native_loss_bytes_independently_decoded=True, GPU_reduction_reexecuted=False,
                best_h_selected=False, AD_correctness_certified=False, scientific_admission=False,
                inputs=dict(report_sha256=report_sha, receipt_sha256=receipt_sha, source_sha256=SOURCE_SHA))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('producer-output', 'report-sha', 'receipt', 'receipt-sha', 'output'):
        parser.add_argument('--'+key, required=True)
    a = parser.parse_args(); path = Path(a.output).absolute()
    require(path.resolve() == path and path.parent == ROOT/'results' and path.name.startswith(NAME+'_'), 'own output')
    value = build(a.producer_output, a.report_sha, a.receipt, a.receipt_sha)
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n'
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'whole report'); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(output=str(path), sha256=sha(path), rows=value['rows'])))


if __name__ == '__main__':
    main()
