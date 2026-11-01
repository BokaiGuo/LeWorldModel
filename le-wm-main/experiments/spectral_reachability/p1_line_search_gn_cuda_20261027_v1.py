"""Fixed four-update, five-scale H2 GN baseline; MAIN-only HOST CUDA.

Saved B8 tasks only, zero restart, unchanged Q/damping/feasible packet balls.
No early stopping, new image/physics/training or CPU neural fallback.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import os
from pathlib import Path
import shutil
import zlib

import numpy as np

from experiments.spectral_reachability import p1_gn_step_diagnostic_cuda_20261027_v1 as saved

NAME = 'p1_line_search_gn_cuda_20261027_v1'
ROOT, screen, core, io = saved.ROOT, saved.screen, saved.core, saved.io
require, sha, read, canonical = saved.require, saved.sha, saved.read, saved.canonical
SCALES = (1., .5, .25, .125, .0625)
UPDATES, DAMPING = 4, .002
CAP, FLOOR, FINAL = 2*1024**2, 200*1024**2, 128*1024


def budget():
    # Initial triple H2=6; each update ADforward2 + five H2 trials10 + triple replay6.
    return dict(predict=78, action_encoder=78, jacobian=4, vjp_directions=768,
                svd=4, linear_solve=4, proposal=4, selection=4)


def material():
    original = saved.material()  # raw B8 task/arm/journal, not diagnostic outcomes
    bindings = deepcopy(original['bindings_sha256'])
    for path in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(path)] = sha(path)
    for path, expected in bindings.items():
        require(sha(path) == expected, 'source/input TOCTOU: '+path)
    return dict(schema=NAME+'.prepare', bindings_sha256=bindings,
        roots=[{k:r[k] for k in ('root_index', 'root_sha256', 'task_id', 'task_path', 'component_path', 'journal_path')}
               for r in original['roots']], policy=original['policy'], expected_model_hash=original['expected_model_hash'],
        scales=list(SCALES), updates=UPDATES, damping=DAMPING, initial_logits=np.zeros((2, 10)).tolist(),
        direction='original task-Q ridge GN; project full endpoint to original packet logit balls',
        trials='float64 convex segment x+s*(projected_endpoint-x), float32 once, NumPy float32 tanh',
        selection='incumbent then all five fresh costs, stable minimum; independent triple replay every update',
        per_root_budget=budget(), total_budget={k:2*v for k,v in budget().items()},
        output_cap_bytes=CAP, floor_bytes=FLOOR, final_reserve_bytes=FINAL,
        storage='lossless canonical JSON gzip level6; Z_SYNC_FLUSH/fsync each event, raw rows AND chained J',
        new_images=0, new_physics=0, training=False, new_Q=False, independent_sample_increment=0,
        inherited_direct_resources_once=original['inherited_direct_resources_once'],
        inherited_image_resources_once=original['inherited_image_resources_once'],
        P1_admission=False, scientific_pass_threshold=None,
        scope='two existing calibration roots; repair fixed-step baseline, not equal B8 cost or physical efficacy')


def prepare(output):
    p = material(); raw = canonical(p); path = Path(output).absolute()
    require(len(raw) < 512*1024 and not path.exists(), 'exclusive small prepare')
    require(shutil.disk_usage(path.parent).free >= FLOOR+len(raw), 'prepare 200MiB floor')
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path), prepare_sha256=sha(path), bytes=len(raw))


def verify(path, expected):
    require(sha(path) == expected, 'external prepare SHA')
    p = read(path); require(p == material(), 'full source/task/model/protocol reconstruction')
    return p


class Store:
    """Flat, exclusive 2MiB total. Failed writes consume quota; never resume."""
    def __init__(self, output):
        self.output = Path(output).absolute(); self.comp = zlib.compressobj(6, zlib.DEFLATED, 31)
        self.broken = False; self.closed = False
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free >= FLOOR+CAP,
                'exclusive output, 2MiB plus200MiB reserve')
        self.output.mkdir(); self._write('events.jsonl.gz', b'')

    def _write(self, name, raw, append=False, final=False):
        require(Path(name).name == name, 'flat local file')
        used = sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw) <= CAP-(0 if final else FINAL), '2MiB total quota including partial bytes')
        require(shutil.disk_usage(self.output).free >= FLOOR+len(raw), '200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())

    def event(self, e):
        require(not self.broken and not self.closed, 'no resume on broken/closed journal')
        try:
            self._write('events.jsonl.gz', self.comp.compress(canonical(e))+self.comp.flush(zlib.Z_SYNC_FLUSH), append=True)
        except BaseException:
            self.broken = True
            raise

    def put(self, name, value, final=False):
        raw = canonical(value)
        self._write(name, gzip.compress(raw, compresslevel=6, mtime=0) if name.endswith('.gz') else raw, final=final)

    def close(self):
        require(not self.closed and not self.broken, 'no false complete gzip footer')
        self.closed = True
        self._write('events.jsonl.gz', self.comp.flush(zlib.Z_FINISH), append=True, final=True)


def recover(raw):
    """Complete durable lines plus gzip EOF; EOF false is never completion."""
    d = zlib.decompressobj(31); plain = d.decompress(raw)
    require(not d.unused_data, 'trailing data after single journal stream')
    import json
    return [json.loads(line) for line in plain.split(b'\n')[:-1]], d.eof


class Meter(screen.Meter):
    def __init__(self, sink, sync=lambda: None):
        super().__init__(budget(), sink); self.sync = sync

    def begin(self, resource):
        # Only a jacobian umbrella may contain nested forward/VJP entries.
        require(all(e['completed'] or e['resource'] == 'jacobian' for e in self.extended), 'failed work cannot be retried')
        return super().begin(resource)

    def emit(self, kind, **values):
        self.sink(dict(kind=kind, evaluation_id=self.group, **deepcopy(values)))

    def end(self, token):
        require(not token[0]['completed'], 'no repeated return')
        self.sync()  # no false completed receipt on asynchronous CUDA error
        super().end(token)

    def counts(self):
        return {k:dict(attempted=sum(e['resource'] == k for e in self.extended),
                       completed=sum(e['resource'] == k and e['completed'] for e in self.extended)) for k in self.caps}


def line_trials(x, endpoint, cap):
    x, endpoint = core.array(x, (2, 10)), core.array(endpoint, (2, 10))
    require(all(np.array_equal(a, a.astype(np.float32)) and (np.linalg.norm(a, axis=1) <= cap).all()
                for a in (x, endpoint)), 'feasible float32 endpoints')
    trials = [(x+s*(endpoint-x)).astype(np.float32).astype(float) for s in SCALES]
    require(all((np.linalg.norm(a, axis=1) <= cap).all() for a in trials), 'rounded line must remain feasible; no repair')
    return trials


def single(task, logits, backend, meter, mode):
    raw = np.tanh(np.asarray(logits, np.float32)).reshape(10, 2)
    meter.emit('evaluation_request', mode=mode, logits=np.asarray(logits).tolist(), commands=raw.astype(float).tolist())
    # native_packet events already fsync each context before returning.
    path = backend.rollout(task, raw, meter, mode)
    saved.check_path(task, raw, path)
    value = dict(task_id=core.task_id(task), logits=np.asarray(logits).tolist(), commands=raw.astype(float).tolist(),
                 contexts=path, cost=core.model_cost(task, path[-1]['latent_history'][-1]))
    meter.emit('evaluation_complete', mode=mode, prediction=value)
    return value


def verified(task, logits, backend, meter):
    values = [single(task, logits, backend, meter, mode) for mode in ('nominal', 'whole_replay', 'segmented_replay')]
    require(values[0]['contexts'] == values[1]['contexts'] == values[2]['contexts'], 'independent triple replay mismatch')
    p = dict(values[0], whole_replay=values[1]['contexts'][-1], segmented_replay=values[2]['contexts'])
    core.check_prediction(task, p)
    return p


def select(incumbent, candidates):
    require(len(candidates) == len(SCALES), 'ALL five trials before selecting; no early exit')
    values = [incumbent]+candidates
    require(all(type(p['cost']) in (float, int) and np.isfinite(p['cost']) and p['cost'] >= 0 for p in values), 'finite costs')
    return min(range(6), key=lambda i:(values[i]['cost'], i))


def optimize(task, backend, meter):
    core.validate_task(task)
    meter.group = 'initial'; incumbent = verified(task, np.zeros((2, 10)), backend, meter)
    meter.emit('initial_verified', incumbent=incumbent)
    steps = []
    for iteration in range(1, UPDATES+1):
        meter.group = f'iter_{iteration}_AD'
        terminal, j = meter.call('jacobian', lambda: backend.jacobian(task, incumbent, meter))
        terminal, j = core.array(terminal, (192,)), core.array(j, (192, 20))
        require(np.array_equal(terminal, incumbent['contexts'][-1]['latent_history'][-1]), 'fresh AD incumbent terminal mismatch')
        q = np.asarray(task['projection']); residual = q@(terminal-np.asarray(task['goal'])); a = q@j
        derivative = dict(task_id=core.task_id(task), at_logits=incumbent['logits'], at_terminal=terminal.tolist(),
                          raw_terminal_jacobian=j.tolist(), objective_gradient=(2*a.T@residual/192).tolist())
        core.check_derivative(task, derivative, incumbent)
        meter.emit('derivative_complete', derivative=derivative)
        u, s, vh = meter.call('svd', lambda: np.linalg.svd(a/np.sqrt(192), full_matrices=False))
        delta = meter.call('linear_solve', lambda: -vh.T@(s*(u.T@(residual/np.sqrt(192)))/(s*s+DAMPING)))
        endpoint = meter.call('proposal', lambda: screen.projected_logits(
            np.asarray(incumbent['logits'])+delta.reshape(2, 10), task['packet_logit_cap']))
        trials = line_trials(incumbent['logits'], endpoint, task['packet_logit_cap'])
        meter.emit('line_proposals_frozen', iteration=iteration, incumbent_logits=incumbent['logits'],
                   delta=delta.tolist(), endpoint=endpoint.tolist(), scales=list(SCALES), logits=[x.tolist() for x in trials])
        candidates = []
        for index, x in enumerate(trials):
            meter.group = f'iter_{iteration}_scale_{index}'
            candidates.append(single(task, x, backend, meter, 'whole_replay'))
        index = meter.call('selection', lambda: select(incumbent, candidates))
        chosen = ([incumbent]+candidates)[index]
        row = dict(iteration=iteration, incumbent_cost=incumbent['cost'], trial_costs=[c['cost'] for c in candidates],
                   selected_index=index, selected_scale=None if index == 0 else SCALES[index-1], selected_cost=chosen['cost'])
        meter.emit('selection_complete', **row)
        meter.group = f'iter_{iteration}_selected_replay'
        fresh = verified(task, chosen['logits'], backend, meter)
        require(fresh['contexts'] == chosen['contexts'] and fresh['cost'] == chosen['cost'], 'selected exact fresh reproduction')
        incumbent = fresh; steps.append(row)
        meter.emit('update_verified', iteration=iteration, prediction=incumbent)
    require(meter.counts() == {k:dict(attempted=v, completed=v) for k,v in budget().items()}, 'complete exact per-root budget')
    return dict(task_id=core.task_id(task), steps=steps, final_prediction=incumbent, resources=meter.counts())


def run(prepared, expected, output):
    p = verify(prepared, expected); store = Store(output)
    model = None; before = after = host = None; errors = []; rows = []; meters = []; source_ok = False
    store.put('binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    def fail(stage, exc): errors.append(saved.error(stage, exc))
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        io.frozen(model); before = io.tensor_hash(model)
        require(before == p['expected_model_hash'], 'original B8 actual frozen model')
        with sdpa_kernel(SDPBackend.MATH):
            for root in p['roots']:
                task = read(root['task_path'])['task']
                require(core.task_id(task) == root['task_id'], 'bound task identity')
                def sink(e, r=root): store.event(dict(root_index=r['root_index'], root_sha256=r['root_sha256'], **e))
                meter = Meter(sink, sync=lambda: torch.cuda.synchronize(device)); meters.append(meter)
                store.event(dict(kind='root_started', root_index=root['root_index'], task_id=root['task_id']))
                backend = screen.NativeCuda(model, device, task)
                with backend.guard(meter):
                    row = dict(root_sha256=root['root_sha256'], root_index=root['root_index'], **optimize(task, backend, meter))
                store.put(f'root_{root["root_index"]}.json.gz', row); rows.append(row)
    except BaseException as exc:
        fail('execution', exc)
        try: store.event(dict(kind='execution_error', error=errors[0]))
        except BaseException as secondary: fail('error_journal', secondary)
    finally:
        if model is not None:
            try: io.frozen(model); after = io.tensor_hash(model)
            except BaseException as exc: fail('final_model', exc)
        try: verify(prepared, expected); source_ok = True
        except BaseException as exc: fail('final_source', exc)
        try: store.close()
        except BaseException as exc: fail('journal_finalize', exc)
        counts = {k:dict(attempted=sum(m.counts()[k]['attempted'] for m in meters),
                        completed=sum(m.counts()[k]['completed'] for m in meters)) for k in budget()}
        complete = (not errors and len(rows) == 2 and source_ok and before is not None and before == after and
                    counts == {k:dict(attempted=2*v, completed=2*v) for k,v in budget().items()})
        hashes = {}
        for path in store.output.iterdir():
            if path.is_file():
                try: hashes[path.name] = sha(path)
                except BaseException as exc: fail('artifact_hash', exc); complete = False
        report = dict(schema=NAME+'.report', status='cuda_line_search_gn_completed' if complete else 'partial_or_failed',
            prepare_sha256=expected, resources=counts, errors=errors, first_error=errors[0] if errors else None,
            model_hash_before=before, model_hash_after=after, model_unchanged=before == after if before is not None else None,
            source_verified=source_ok, host=host, artifact_sha256=hashes,
            roots=[dict(root_sha256=r['root_sha256'], steps=r['steps'], final_cost=r['final_prediction']['cost']) for r in rows],
            inherited_resources_are_separate=True, new_images=0, new_physics=0, training=False, new_Q=False,
            independent_sample_increment=0, P1_admission=False, scientific_pass_threshold=None)
        try: store.put('report.json', report, final=True)
        except BaseException as exc:
            fail('final_report', exc); complete = False
            print(canonical(dict(first_error=errors[0], errors=errors)).decode(), flush=True)
    require(complete, 'partial line search preserved; no retries: '+str(errors))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha'); parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    result = prepare(a.output) if a.command == 'prepare' else (
        dict(status='verified_no_execution', roots=len(verify(a.prepared, a.prepare_sha)['roots'])) if a.command == 'verify'
        else run(a.prepared, a.prepare_sha, a.output))
    print(canonical(result).decode().strip())
