"""MAIN independent saved-array recomputation of all80 actual-cp5 raw FD pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import main_observed_fd_scalar_cause_20261028_v1 as base

NAME = 'main_observed_raw_fd_recompute_20261028_v1'
ROOT = base.ROOT
PREP = ROOT/'results/agent_observed_raw_coordinate_fd_20261028_v1_prepare_MAIN01.json'
OUTPUT = ROOT/'results/agent_observed_raw_coordinate_fd_20261028_v1_cuda_MAIN01'
RECEIPT = ROOT/'results/agent_observed_raw_coordinate_fd_20261028_v1_consume_MAIN01.json'
PINS = ((PREP, 'a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5'),
        (OUTPUT/'report.json', 'df21e0cd0a235a6e6f43572cc02614f8c62b4704bac2aabb8ec4e2886104aa15'),
        (RECEIPT, 'c2c454cf51b11e5f28df12e2c2f9ef12c8c7cd1eba7bf99b9a9cf9c97977e2b0'))
require = base.require


def vector_metrics(fd, ad):
    fd, ad = np.asarray(fd, float), np.asarray(ad, float)
    require(fd.shape == ad.shape == (10,) and np.isfinite(fd).all() and np.isfinite(ad).all(), 'finite raw10 gradients')
    fn, an = float(np.linalg.norm(fd)), float(np.linalg.norm(ad))
    error = float(np.linalg.norm(fd-ad))
    return dict(cosine=float(fd@ad)/(fn*an) if fn*an else None,
                relative_L2=error/an if an else None, L2_error=error,
                Linf_error=float(np.max(abs(fd-ad))), FD_L2=fn, AD_L2=an)


def build():
    for f, h in PINS:
        require(base.sha(f) == h, 'pinned actual prepare/report/full strict receipt')
    p, r, u = [base.read(f) for f, _ in PINS]
    require(u['complete'] is True and u['fixed_denominator'] == 164 and u['pair_denominator'] == 80
            and u['report_sha256'] == PINS[1][1] and r['errors'] == []
            and r['model_before'] == r['model_after'], 'complete actual fixed164, unchanged model')
    authority = p['authority']
    original_path = Path(authority['output'])/'report.json'
    require(base.sha(original_path) == authority['report_sha256'], 'original observed scalar AD receipt bytes')
    original = base.read(original_path)
    panels = []
    for slot in range(4):
        jobs = p['jobs'][41*slot:41*(slot+1)]
        values = r['rows'][slot]['values']
        numeric = original['rows'][slot]['result']['numeric']
        ad = np.asarray(numeric['gradients'][0]['raw_gradient'], float).reshape(10)
        anchor = np.asarray(numeric['iterates'][0]['commands'], np.float32).reshape(10)
        require(values[0]['context'] == numeric['iterates'][0]['context']
                and len(values) == len(jobs) == 41, 'original iterate0 baseline exact, not selected new suffix')
        steps = []
        for hi, h in enumerate((.01, .001)):
            fd = []; endpoint_fd = []; float64_fd = []
            for coord in range(10):
                k = 1+20*hi+2*coord
                minus, plus = values[k:k+2]
                spans = []
                for offset, sign in enumerate((-1, 1)):
                    j = jobs[k+offset]
                    expected = anchor.copy()
                    expected[coord] = np.float32(float(anchor[coord])+sign*h)
                    actual = np.asarray(j['raw_suffix5'], np.float32).reshape(10)
                    require(actual.tobytes() == expected.tobytes() and j['coord'] == coord
                            and j['h'] == h and j['sign'] == sign, 'independent raw coordinate operation/axis order')
                    spans.append(float(actual[coord]))
                span = spans[1]-spans[0]
                require(span > 0 and span == jobs[k]['actual_pair_spacing'], 'actual float32 span, no nominal denominator')
                ym, yp = [np.asarray(v['context']['latent_history'][-1], float) for v in (minus, plus)]
                goal = np.asarray(jobs[k]['task']['goal'], float)
                g = ((yp-goal)**2).mean()-((ym-goal)**2).mean()
                float64_fd.append(float(g/span))
                native = [v['objective']['diagnostic']['native_float32_Q'] for v in (minus, plus)]
                fd.append((native[1]-native[0])/span)
                endpoint_fd.append(((yp-ym)/span).tolist())
                saved = u['panels'][slot]['steps'][hi]['pairs'][coord]
                require(saved['FD_native_Q_actual_spacing'] == fd[-1]
                        and saved['FD_float64_Q_actual_spacing'] == float64_fd[-1]
                        and saved['endpoint192_derivative_actual_spacing'] == endpoint_fd[-1]
                        and saved['AD_raw_gradient'] == ad[coord], 'all original arrays and80 saved scalar/vector FDs independently recomputed')
            metrics = vector_metrics(fd, ad)
            require(metrics == u['panels'][slot]['steps'][hi]['metrics_actual_spacing'], 'independent10dim norm/cosine')
            steps.append(dict(h=h, native_scalar_fd=fd, float64_scalar_fd=float64_fd,
                              metrics=metrics, endpoint192_secants=endpoint_fd))
        panels.append(dict(slot=slot, root_slot=jobs[0]['root_slot'], condition=jobs[0]['condition'], steps=steps))
    return dict(schema=NAME, panels=panels, fixed_requests=164, fixed_pairs=80, known_roots=2,
                bindings={str(f): h for f, h in (*PINS, (original_path, authority['report_sha256']))},
                full_upstream_rebuilt_here=False, actual_full_strict_receipt_pinned=True,
                new_neural_calls=0, new_physics_calls=0, P1_admission=False,
                scope='Independent numerical recheck of new actual-cp5 model FD only. No certification of physical response or theory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output).absolute()
    require(output.resolve() == output and output.parent == ROOT/'results'
            and output.name.startswith(NAME+'_') and output.suffix == '.json', 'exclusive own result')
    result = build()
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=base.sha(output), fixed_pairs=80)))


if __name__ == '__main__':
    main()
