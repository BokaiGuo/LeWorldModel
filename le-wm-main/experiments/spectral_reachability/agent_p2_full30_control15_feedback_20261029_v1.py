"""Own full30 prepare/HostCUDA/saved strict; no producer globals or receipt impersonation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import inspect
from importlib.machinery import PathFinder
import io
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as native
from experiments.spectral_reachability import agent_p2_saved_program_linker_20261029_v1 as linker
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_20261029_v1_kernel as K

from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1 as P
from experiments.spectral_reachability import agent_p2_two_program_full30_rgb_encode_20261029_v1 as E

NAME, ROOT = K.NAME, native.ROOT
AUTH = 'MAIN_FULL30_CONTROL15_FEEDBACK_HOST_CUDA_ONLY'
CAP, RESERVE, FLOOR, META = P.CAP, P.RESERVE, P.FLOOR, P.META
raw, digest, require, equal, sha = K.raw, K.digest, K.require, K.equal, native.sha
read, merge, rehash = P.read, P.merge, P.rehash
PROTOCOLS = {
    str(ROOT/'innerdocs/agent_p2_two_program_full30_control15_feedback_protocol_20261029_v1.md'):
    '75e8604604bdd8f6a12eda24367747640a7318ca4525c6dba19c685fd85060d4',
    str(ROOT/'innerdocs/agent_p2_two_program_full30_control15_feedback_authority_20261029_v1.md'):
    '636c87f88d83c9a69134b1d1949e9f638c325932f295ac7edafda558c3f63895'}
PINS = {
    'prepared':dict(path=str(ROOT/'results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json'),
        sha256='20a248dc940188ff3858fe71769cdc2d726ec23d2e77e8e50783ff777db63527'),
    'report':dict(path=str(ROOT/'results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01/report.json'),
        sha256='915b6760db8bf30abe16e4bfba347604e6359c17d08c1ff3f2a7a076b7714df4'),
    'receipt':dict(path=str(ROOT/'results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json'),
        sha256='42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166')}


def files():
    return sorted(set(P.files()) | {Path(E.__file__).resolve(), Path(__file__).resolve(),
        Path(K.__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')})


def material(ep, receipt):
    require(ep['schema'] == E.NAME+'.prepare' and receipt['schema'] == E.NAME+'.consume' and
            receipt['status'] == 'all28_full30_RGB_verified_NOT_admitted' and
            receipt['fixed_denominator'] == 28 and receipt['plan_denominator'] == 4 and
            len(ep['plans']) == len(receipt['actual_context_chains']) == 4 and
            len(ep['rows']) == len(receipt['rows']) == 28, 'complete inherited RGB identity')
    for k in ('P1_admission', 'P2_admission', 'P2_mechanism', 'C3', 'physical_release'):
        require(receipt[k] is False, 'upstream scientific gates false')
    jobs, bases = [], []
    for i, wrapper in enumerate(ep['plans']):
        plan = wrapper['planner_plan']
        require(type(wrapper['index']) is int and wrapper['index'] == i and
                type(plan['root_slot']) is int and plan['root_slot'] == i//2 and
                plan['arm'] == K.parent.ARMS[i%2] and wrapper['root_slot'] == plan['root_slot'] and
                wrapper['arm'] == plan['arm'] and wrapper['task_id'] == plan['task_id'], 'fixed four base order/task')
        witness = dict(plan); h = witness.pop('witness_sha256')
        require(digest(witness) == h, 'original planner witness')
        full = K.parent.legal(plan['logits'])
        actions = K.parent.chart(full).reshape(30, 2)
        require(actions.tobytes().hex() == plan['raw_float32_bits'] and
                actions.tobytes() == np.asarray(wrapper['commands'], '<f4').tobytes() and
                plan['model_hash'] == ep['expected_model_hash'], 'original literal model/raw30')
        equal(plan['normalization'], dict(mean=native.kernel.ACTION_MEAN.tolist(),
              std=native.kernel.ACTION_STD.tolist()), 'unchanged action normalizer')
        chain = receipt['actual_context_chains'][i]
        require(type(chain['plan_index']) is int and chain['plan_index'] == i and
                chain['root_slot'] == i//2 and chain['arm'] == plan['arm'], 'chain base identity')
        require([x['step'] for x in chain['checkpoints']] == list(E.CP) and
                all(type(x['step']) is int for x in chain['checkpoints']), 'full checkpoint ordering')
        selected = []
        for j, cp in enumerate((5, 10, 15), 1):
            spec, row = ep['rows'][7*i+j], receipt['rows'][7*i+j]
            for value in (spec, row):
                require(type(value['plan_index']) is int and value['plan_index'] == i and
                        type(value['checkpoint']) is int and value['checkpoint'] == cp and
                        value['task_id'] == plan['task_id'] and value['root_slot'] == i//2 and
                        value['arm'] == plan['arm'] and value['selection_digest'] == wrapper['selection_digest'],
                        'actual prefix RGB identity, no cross-arm join')
            require(spec['absolute_step'] == cp+10 and spec['source'] is not None and
                    row['latent_missing'] is False and row['source'] == spec['source'] and
                    row['source']['step'] == cp, 'actual source/control/absolute time')
            equal(row['goal192'], plan['goal'], 'same goal')
            equal(spec['goal192'], plan['goal'], 'same prepared goal')
            warm = K.finite(plan['root_warm_raw10'], (10, 2)).astype('<f4')
            a = np.concatenate((warm, actions[:cp]))[-10:]
            require(a.tobytes().hex() == spec['causal_raw10_bits'] == row['causal_raw10_bits'],
                    'prefix causal raw10 bits')
            normalized = native.kernel.normalized_history(a[:5]).tolist()+native.kernel.normalized_history(a[5:]).tolist()
            equal(spec['normalized_action_history'], normalized, 'once normalized true history')
            z = K.finite(row['actual192'], (192,))
            require(np.array_equal(z, z.astype(np.float32).astype(float)), 'literal actual192')
            selected.append(z.tolist())
        observed = deepcopy(chain['checkpoints'][3]['context'])
        require(chain['checkpoints'][3]['context_missing'] is False, 'available observed cp15')
        K.context(observed)
        equal(observed['latent_history'], selected, 'three actual observed frames')
        equal(observed['action_history'], ep['rows'][7*i+3]['normalized_action_history'], 'two real action tokens')
        K.context(plan['contexts'][2])
        radius = K.parent.C15 if i%2 == 0 else float(np.sqrt(K.parent.C30**2-float(full[:30]@full[:30])))
        common = dict(base_index=i, root_slot=i//2, arm=plan['arm'], task_id=plan['task_id'],
                      task_sha256=plan['task_sha256'], parent_witness_sha256=h, parent_logits=full.tolist(),
                      parent_raw_bits=plan['raw_float32_bits'], prefix_raw=actions[:15].tolist(),
                      incumbent_logits=full[30:].tolist(), suffix_cap=radius, goal=deepcopy(plan['goal']))
        bases.append(dict(base_index=i, root_slot=i//2, arm=plan['arm'], policy='O', task_id=plan['task_id'],
                          parent_witness_sha256=h, parent_raw_bits=plan['raw_float32_bits'],
                          parent_commands=actions.tolist(), model_context_scope='inherited only; no new O prediction',
                          parent_model_report_sha256=E.producer.PINS['planner_report_sha']))
        for policy, c in (('I', plan['contexts'][2]), ('F', observed)):
            job = dict(deepcopy(common), policy=policy, input_context=deepcopy(c))
            K.validate(job)
            jobs.append(job)
    return bases, jobs


def build(authorities=None):
    authorities = PINS if authorities is None else authorities
    equal(authorities, PINS, 'only pinned actual RGB three authorities before input reads')
    ep, er, ec = [read(authorities[k]['path'], authorities[k]['sha256']) for k in ('prepared','report','receipt')]
    require(er['schema'] == E.NAME+'.report' and er['prepared'] == authorities['prepared']['path'] and
            er['prepare_sha256'] == ec['prepare_sha256'] == authorities['prepared']['sha256'] and
            ec['report_sha256'] == authorities['report']['sha256'] and er['errors'] == [] and
            er['model_unchanged'] is True and er['model_hash_before'] == er['model_hash_after'] == ep['expected_model_hash'],
            'actual MAIN complete receipt join')
    require(ep['arguments'] == E.ACTUAL, 'original native pins')
    bindings = dict(ep['bindings'])
    merge(bindings, {d['path']:d['sha256'] for d in authorities.values()})
    merge(bindings, {str(Path(authorities['report']['path']).parent/n):h for n,h in er['artifact_sha256'].items()})
    completion = Path(authorities['report']['path']).parent/'completion.json'
    cv = read(completion, sha(completion))
    require(cv['report_sha256'] == authorities['report']['sha256'], 'report completion binding')
    merge(bindings, {str(completion):sha(completion)})
    equal({f.name for f in completion.parent.iterdir()}, set(er['artifact_sha256'])|{'report.json','completion.json'},
          'entire inherited encoding inventory')
    for path,h in PROTOCOLS.items():
        require(sha(path) == h, 'approved protocol immutable')
    merge(bindings, PROTOCOLS)
    merge(bindings, {str(f.resolve()):sha(f) for f in files()})
    merge(bindings, P.installed_sources())
    rehash(bindings)
    bases, jobs = material(ep, ec)
    return dict(schema=NAME+'.prepare', authorities=deepcopy(authorities), contract=K.contract(),
                bases=bases, jobs=jobs, policy=ep['policy'], expected_model_hash=ep['expected_model_hash'],
                bindings_sha256=bindings, full_upstream_rebuilt_here=False,
                inherited_resources=dict(model=ep['inherited_model_resources'], native=ep['inherited_physical_counts'],
                    encoding=ec['counts'], charged_again_as_new=False),
                inherited_authority='pinned MAIN full RGB/model/native receipts; no upstream fullstrict rerun',
                generator_labels_max_control=15, **K.FLAGS)


def verify(path, h):
    p = read(path, h)
    equal(p, build(p['authorities']), 'entire own prepare reconstruction')
    return p


def write_new(path, value):
    p = Path(path).absolute()
    data = raw(value)
    require(p.name.startswith(NAME+'_') and p.parent.is_dir() and not p.is_symlink() and len(data) <= META and
            shutil.disk_usage(p.parent).free >= FLOOR+len(data), 'own exclusive bounded metadata')
    with p.open('xb') as f:
        require(f.write(data) == len(data), 'short metadata')
        f.flush()
        os.fsync(f.fileno())
    native.fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


class Store:
    """Single-LF native hashchain; inclusive clock; incremental physical byte cap."""
    def __init__(self, output, started):
        self.output = Path(output).absolute()
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and not self.output.is_symlink() and
                shutil.disk_usage(self.output.parent).free >= CAP+FLOOR, 'exclusive output and floor')
        self.output.mkdir()
        self.started, self.slot_started = started, None
        self.index, self.previous, self.used, self.closed = 0, '0'*64, 0, False
        self.metadata_used = 0
        self.comp = zlib.compressobj(6, zlib.DEFLATED, 31)
        self._write('events.jsonl.gz', b'', final=True)

    def check(self):
        now = time.monotonic()
        require(now-self.started <= 1800 and (self.slot_started is None or now-self.slot_started <= 450), 'whole/slot inclusive wall')

    def _write(self, name, data, append=False, final=False, time_guard=True):
        if not final and time_guard:
            self.check()
        require(Path(name).name == name and name not in ('', '.', '..') and not (self.output/name).is_symlink(), 'flat artifact')
        require(self.used+len(data) <= CAP-(0 if final else RESERVE) and
                shutil.disk_usage(self.output).free >= FLOOR+len(data), 'output cap/reserve/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(data) == len(data), 'short artifact')
            f.flush()
            os.fsync(f.fileno())
        self.used += len(data)
        native.fsync_dir(self.output)

    def put(self, name, value, final=False):
        data = raw(value)
        if name.endswith('.gz'):
            data = gzip.compress(data, mtime=0)
        elif name.endswith('.json'):
            require(self.metadata_used+len(data) <= 4*1024**2, 'aggregate runtime JSON metadata4MiB')
            self.metadata_used += len(data)
        self._write(name, data, final=final)

    def event(self, payload):
        require(not self.closed, 'no journal reopen')
        self.check()
        e = dict(index=self.index, previous_sha256=self.previous, payload=deepcopy(payload))
        h = digest(e)
        data = raw(dict(e, event_sha256=h))
        require(self.used+len(data)+128 <= CAP-RESERVE and
                shutil.disk_usage(self.output).free >= FLOOR+len(data)+128, 'admit complete journal record before compression')
        out = self.comp.compress(data)+self.comp.flush(zlib.Z_SYNC_FLUSH)
        self._write('events.jsonl.gz', out, append=True, time_guard=False)
        self.index += 1
        self.previous = h
        self.check()

    def close(self):
        if not self.closed:
            self._write('events.jsonl.gz', self.comp.flush(zlib.Z_FINISH), append=True, final=True)
            self.closed = True


def ad_request(index, c, u, tag):
    K.finite(u, (30,))
    require(type(index) is int and index == 0, 'one new suffix J per job')
    return dict(index=index, tag=tag, input_context_sha256=digest(c), at_logits=np.asarray(u).tolist(),
                raw=K.parent.chart(u).tolist(), coordinates='raw_incumbent', shape=[192, 30], mathematical_VJP=192,
                vectorize=True, completed=False)


class HostBackend(native.HostBackend):
    """Reuse unchanged single-packet bridge; own H3 suffix J, caps and scope."""
    def check(self):
        self.t.cuda.synchronize(self.device)
        self.store.check()
        require(self.t.cuda.max_memory_allocated(self.device) <= K.contract()['allocated_cap_bytes'] and
                self.t.cuda.max_memory_reserved(self.device) <= K.contract()['reserved_cap_bytes'], 'actual allocator peak caps')
        n, j = 27, 1
        require(len(self.row['forwards']) <= n and len(self.row['AD']) <= j, 'fixed own call caps')

    def rollout(self, c, commands, tag):
        require(len(self.row['forwards'])+len(commands) <= 27, 'no extra forward dispatch')
        return super().rollout(c, commands, tag)

    def jacobian(self, c, u, tag):
        self.check()
        n = len(self.row['AD'])
        require(self.row['policy'] in ('I', 'F') and len(self.row['forwards'])+3 <= 27, 'own AD admission')
        request = ad_request(n, c, u, tag)
        self.event(dict(kind='AD_intent', request=request))
        e = dict(request=request, completed=False)
        self.row['AD'].append(e)
        t = self.t
        # The only differentiated input is literal cast32 RAW, not logits or quantization.
        leaf = t.tensor(K.parent.chart(u).reshape(30), device=self.device, dtype=t.float32).requires_grad_(True)
        def fn(value):
            state, path = self._rollout(c, K.parent.chart(u), tag+'/forward', value.reshape(3, 5, 2))
            e['contexts'] = path
            return state[0, -1]
        with t.enable_grad():
            j = t.autograd.functional.jacobian(fn, leaf, vectorize=True)
        self.check()
        require(tuple(j.shape) == (192, 30) and j.dtype == t.float32, 'literal full raw192x30 J')
        a = j.detach().cpu().numpy().copy()
        require(np.isfinite(a).all(), 'finite full J')
        stream = io.BytesIO()
        np.save(stream, a, allow_pickle=False)
        name = f'slot{self.row["slot"]}_J{n}.npy'
        self.store._write(name, stream.getvalue())
        d = dict(file=name, sha256=sha(self.store.output/name), shape=[192, 30], dtype='float32')
        e.update(array=d, completed=True)
        self.last_ad_contexts = deepcopy(e['contexts'])
        self.event(dict(kind='AD_return', index=n, array=d, contexts=e['contexts'], completed=True))
        return a.astype(np.float64)


class SavedBackend(native.SavedBackend):
    def rollout(self, c, commands, tag):
        require(self.n+len(commands) <= 27, 'fixed saved forward cap')
        for f in self.row['forwards'][self.n:self.n+len(commands)]:
            require(set(f) == {'index', 'tag', 'request', 'input', 'raw', 'modules', 'completed', 'output'} and
                    len(f['modules']) == 3 and all(set(m) == {'index', 'resource', 'inputs', 'completed', 'output'}
                    for m in f['modules']), 'exact native full forward and module fields')
            K.context(f['input'])
            K.context(f['output'])
        return super().rollout(c, commands, tag)

    def jacobian(self, c, u, tag):
        require(self.ad < len(self.row['AD']), 'missing J')
        e = self.row['AD'][self.ad]
        request = ad_request(self.ad, c, u, tag)
        require(e['completed'] is True and set(e) == {'request', 'completed', 'contexts', 'array'} and
                e['array']['file'] == f'slot{self.row["slot"]}_J{self.ad}.npy', 'complete own full J authority')
        equal(e['request'], request, 'literal H3 suffix AD request')
        self.event(dict(kind='AD_intent', request=request))
        paths = self.rollout(c, K.parent.chart(u), tag+'/forward')
        equal(paths, e['contexts'], 'AD complete primal context path')
        j = self.load(e['array'])
        require(j.dtype == np.float32 and j.shape == (192, 30), 'strict full192x30 no reduced J')
        self.last_ad_contexts = deepcopy(paths)
        self.event(dict(kind='AD_return', index=self.ad, array=e['array'], contexts=paths, completed=True))
        self.ad += 1
        return j.astype(np.float64)

    def complete(self):
        n, j = 27, 1
        require(self.n == len(self.row['forwards']) == n and self.ad == len(self.row['AD']) == j, 'all fixed paid calls')


def classes(model, bindings):
    out = {}
    for module in model.modules():
        p = str(Path(inspect.getfile(type(module))).resolve())
        require(p in bindings and sha(p) == bindings[p], 'all actual loaded class sources presealed, no skip')
        out[p] = bindings[p]
    return out


def row_plan(p, job, row):
    result = row['result']
    plan = dict(schema=NAME+'.model_plan', base_index=job['base_index'], root_slot=job['root_slot'],
                arm=job['arm'], policy=job['policy'], task_id=job['task_id'], task_sha256=job['task_sha256'],
                parent_witness_sha256=job['parent_witness_sha256'], parent_raw_bits=job['parent_raw_bits'],
                feedback_control=15, input_context=job['input_context'], goal=job['goal'],
                suffix_cap=job['suffix_cap'], model_hash=p['expected_model_hash'],
                commands=result['commands'], raw_float32_bits=result['raw_float32_bits'],
                suffix_logits=result['suffix_logits'], new_suffix_Q=result['new_suffix_Q'],
                new_contexts=result['contexts'], selection_uses_future_labels=False,
                old_fields_scope='parent only; new suffix/context/witness are separate', **K.FLAGS)
    plan['new_witness_sha256'] = digest(plan)
    return plan


def precision(t):
    return dict(TF32_matmul=bool(t.backends.cuda.matmul.allow_tf32),
                TF32_cudnn=bool(t.backends.cudnn.allow_tf32), autocast=bool(t.is_autocast_enabled('cuda')),
                MATH=bool(t.backends.cuda.math_sdp_enabled()), flash=bool(t.backends.cuda.flash_sdp_enabled()),
                mem_efficient=bool(t.backends.cuda.mem_efficient_sdp_enabled()))


def precision_expected():
    return dict(TF32_matmul=False, TF32_cudnn=False, autocast=False, MATH=True, flash=False, mem_efficient=False)


def roster(p, summaries):
    require(len(p['bases']) == 4 and len(summaries) == 8, 'all12 even on failure')
    out=[]
    for i,base in enumerate(p['bases']):
        out.append(dict(base_index=i,policy='O',status='inherited_openloop',parent=deepcopy(base)))
        for j,pol in enumerate(('I','F')):
            s=summaries[2*i+j]
            require(s['base_index']==i and s['policy']==pol, 'stable failed denominator')
            out.append(dict(base_index=i,policy=pol,status=s['status'],model_job_slot=s['slot']))
    return out


def run_cuda(prepared, h, output, authorization):
    require(authorization == AUTH, 'explicit MAIN HostCUDA authorization before I/O')
    started = time.monotonic()  # BEFORE whole own saved verify, not after it.
    p = verify(prepared, h)
    store = Store(output, started)
    rows, errors = [], []
    model = before = after = host = loader_memory = None
    source_ok = False
    precision_before = precision_after = None
    summaries = [dict(slot=i, root_slot=j['root_slot'], base_index=j['base_index'], arm=j['arm'], policy=j['policy'],
                      status='not_started', file=None, new_suffix_Q=None, contexts=None) for i,j in enumerate(p['jobs'])]
    try:
        store.check()  # Expired verify must not import/dispatch the neural loader.
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=h))
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, before = native.loader.loaded_model(p, torch, device, store, h)
        torch.cuda.synchronize(device)
        store.check()
        loader_memory = dict(allocated_peak=torch.cuda.max_memory_allocated(device), reserved_peak=torch.cuda.max_memory_reserved(device))
        require(loader_memory['allocated_peak'] <= K.contract()['allocated_cap_bytes'] and
                loader_memory['reserved_peak'] <= K.contract()['reserved_cap_bytes'], 'loader allocator peaks before reset')
        store.put('classes.json', classes(model, p['bindings_sha256']))
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type='cuda', enabled=False):
            precision_before = precision(torch)
            equal(precision_before, precision_expected(), 'actual original precision before dispatch')
            for summary in summaries:
                job = p['jobs'][summary['slot']]
                store.slot_started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                row = dict(slot=summary['slot'], index=summary['root_slot'], arm=summary['arm'],
                           policy=summary['policy'], base_index=summary['base_index'], status='attempted', forwards=[], AD=[])
                rows.append(row)
                summary['status'] = 'attempted'
                store.event(dict(kind='slot_started', slot=row['slot'], root_slot=row['index'], arm=row['arm'], policy=row['policy'], base_index=row['base_index']))
                emit = lambda e: store.event(dict(slot=row['slot'], event=e))
                backend = HostBackend(model, device, torch, store, row, emit)
                with backend.observe():
                    row['result'] = K.execute(job, backend)
                require((len(row['forwards']), len(row['AD'])) == (27, 1), 'exact own natural counts')
                row['plan'] = row_plan(p, job, row)
                row['resources'] = K.resources(row['forwards'], row['AD'])
                row['memory'] = dict(allocated_peak=torch.cuda.max_memory_allocated(device),
                                     reserved_peak=torch.cuda.max_memory_reserved(device))
                row['status'] = 'sealed'
                name = f'slot{row["slot"]}.json.gz'
                store.put(name, row)
                store.event(dict(kind='slot_sealed', slot=row['slot'], witness=row['plan']['new_witness_sha256']))
                store.check()
                summary.update(status='sealed', file=name, new_suffix_Q=row['result']['new_suffix_Q'],
                               contexts=row['result']['contexts'], wall_seconds=time.monotonic()-store.slot_started,
                               resources=row['resources'])
                store.slot_started = None
            precision_after = precision(torch)
            equal(precision_after, precision_before, 'actual original precision unchanged')
    except BaseException as exc:
        errors.append(native.loader.enc.error('execution', exc))
    finally:
        store.slot_started = None
        for s in summaries:
            if s['status'] == 'attempted':
                s['status'] = 'failed_partial'
        if model is not None:
            try:
                native.loader.enc.frozen(model)
                after = native.loader.enc.tensor_hash(model)
            except BaseException as exc:
                errors.append(native.loader.enc.error('final_model', exc))
        try:
            rehash(p['bindings_sha256'])
            store.check()
            source_ok = True
        except BaseException as exc:
            errors.append(native.loader.enc.error('final_source_wall', exc))
        for row in rows:
            if row['status'] != 'sealed':
                try:
                    store.put(f'partial_slot{row["slot"]}.json.gz', row, final=True)
                except BaseException as exc:
                    errors.append(native.loader.enc.error('partial_persistence', exc))
        try:
            store.close()
        except BaseException as exc:
            errors.append(native.loader.enc.error('journal_close', exc))
        good = not errors and source_ok and before == after == p['expected_model_hash'] and all(s['status'] == 'sealed' for s in summaries)
        actual_resources = census(journal(store.output)) if store.closed else None
        elapsed = time.monotonic()-started
        good = good and elapsed <= 1800
        store.put('seal.json', dict(schema=NAME+'.seal', complete=good,
                  through_durable_data_seconds=elapsed,
                  data_artifacts_sha256={f.name: sha(f) for f in store.output.iterdir()}), final=True)
        report = dict(schema=NAME+'.report', status='all8_feedback_sealed_NOT_admitted' if good else 'failed_partial',
                      prepared=str(Path(prepared).absolute()), prepare_sha256=h, contract=K.contract(), rows=summaries,
                      errors=errors, first_error=errors[0] if errors else None, model_before=before, model_after=after,
                      source_verified=source_ok, host=host, wall_seconds=time.monotonic()-started,
                      actual_resources=actual_resources, inherited_resources=p['inherited_resources'], loader_memory=loader_memory,
                      policy_roster=roster(p, summaries), precision_before=precision_before, precision_after=precision_after,
                      artifact_sha256={f.name: sha(f) for f in store.output.iterdir()}, **K.FLAGS)
        store.put('report.json', report, final=True)
        # Report SHA authenticates the seal. If final publication overruns, preserve an
        # explicit non-complete marker; the strict complete inventory rejects it.
        if good:
            try:
                report_hash = sha(store.output/'report.json')
                store.check()
                completion = dict(schema=NAME+'.completion', report_sha256=report_hash,
                                  boundary='after_report_fsync_and_hash', elapsed_seconds=time.monotonic()-started)
                store.put('completion.json', completion, final=True)
                completion_hash = sha(store.output/'completion.json')
                store.check()  # Also guard the transport of the completion itself.
            except BaseException as exc:
                good = False
                store.put('terminal_failure.json', dict(reason='final_publication_wall',
                          error=native.loader.enc.error('final_publication', exc)), final=True)
    require(good, 'failed partial output retained; no retry/refund '+str(errors))
    return dict(report=str(store.output/'report.json'), sha256=report_hash,
                completion_sha256=completion_hash)


def journal(output):
    path = Path(output)/'events.jsonl.gz'
    require(path.stat().st_size <= CAP, 'bounded event gzip')
    decoder = zlib.decompressobj(31)
    data = decoder.decompress(path.read_bytes(), CAP+1)
    require(len(data) <= CAP and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
            (not data or data.endswith(b'\n')), 'single bounded complete gzip / single-LF journal')
    out, previous = [], '0'*64
    for index, line in enumerate(data.splitlines()):
        require(bool(line), 'no empty-record filtering')
        e = linker.decode(line)
        require(set(e) == {'index', 'previous_sha256', 'payload', 'event_sha256'} and
                type(e['index']) is int and e['index'] == index and e['previous_sha256'] == previous and
                raw(e) == line+b'\n', 'exact event bytes/order/fields')
        require(digest({k: v for k, v in e.items() if k != 'event_sha256'}) == e['event_sha256'], 'whole native hashchain')
        previous = e['event_sha256']
        out.append(e['payload'])
    return out


def census(events):
    """Durable intents, including pending failures; completed row lists are not attempts."""
    out = dict(forward_attempted=0, forward_returned=0, jacobian_attempted=0, jacobian_returned=0,
               predict_API_attempted=0, predict_API_returned=0,
               modules_attempted={r: 0 for r in ('action_encoder', 'predictor', 'pred_proj')},
               modules_returned={r: 0 for r in ('action_encoder', 'predictor', 'pred_proj')})
    for x in events:
        if 'event' not in x:
            continue
        e = x['event']
        key = {'forward_intent': 'forward_attempted', 'forward_return': 'forward_returned',
               'AD_intent': 'jacobian_attempted', 'AD_return': 'jacobian_returned',
               'predict_API_intent': 'predict_API_attempted', 'predict_API_return': 'predict_API_returned'}.get(e['kind'])
        if key:
            out[key] += 1
        if e['kind'] == 'module_intent':
            out['modules_attempted'][e['resource']] += 1
        elif e['kind'] == 'module_return':
            out['modules_returned'][e['entry']['resource']] += 1
    return out


def size_budget(rows, events):
    """RAM shape census: hex blobs exact width; every JSON float conservatively 32 chars."""
    def bound(v):
        if type(v) is float:
            require(np.isfinite(v) and len(raw(v))-1 <= 32, 'bounded numeric JSON representation')
            return 32
        if type(v) is list:
            return 2+max(0, len(v)-1)+sum(bound(x) for x in v)
        if type(v) is dict:
            return 2+max(0, len(v)-1)+sum(len(raw(k))-1+1+bound(x) for k, x in v.items())
        return len(raw(v))-1
    envelopes = [dict(index=i, previous_sha256='f'*64, payload=e, event_sha256='f'*64) for i, e in enumerate(events)]
    ledger = sum(len(raw(e)) for e in envelopes)
    row_bytes = sum(len(raw(r)) for r in rows)
    worst_ledger = sum(bound(e)+1 for e in envelopes)
    worst_rows = sum(bound(r)+1 for r in rows)
    matrices = 8*(192*30*4+128)  # Eight complete float32 NPY files, real v1.0 header128.
    metadata = 4*1024**2  # Enforced by Store.put across loader/classes/binding/report/seal.
    # zlib/gzip may expand incompressible bytes: include conservative member overheads.
    def compressed_bound(n):
        return n+(n>>12)+(n>>14)+(n>>25)+32
    # Store flushes EVERY record, not merely at stream end. Treat each as a
    # separately bounded compressed unit and additionally reserve 64 bytes for
    # its empty stored SYNC_FLUSH block/alignment; 128 for stream header/trailer.
    record_sizes = [bound(e)+1 for e in envelopes]
    journal_compressed_bound = 128+sum(compressed_bound(n)+64 for n in record_sizes)
    compression_overhead = journal_compressed_bound-worst_ledger + sum(
        compressed_bound(bound(r)+1)-(bound(r)+1) for r in rows)
    worst = worst_ledger+worst_rows+matrices+metadata+compression_overhead
    require(worst < CAP-RESERVE, 'full-shape JSON/NPY envelope exceeds approved output cap')
    return dict(fixture_uncompressed_ledger=ledger, fixture_uncompressed_rows=row_bytes,
                worst_uncompressed_ledger=worst_ledger, worst_uncompressed_rows=worst_rows,
                full_J_NPY_bytes=matrices, enforced_metadata_bytes=metadata,
                gzip_expansion_bound=compression_overhead,
                sync_flush_records=len(envelopes), sync_flush_bytes_per_record=64,
                journal_compressed_bound=journal_compressed_bound,
                conservative_uncompressed_total=worst, physical_output_cap=CAP, reserve=RESERVE)


def policy_slots(p, plans):
    require(len(p['bases']) == 4 and len(plans) == 8, 'whole12 denominator')
    out = []
    for i, base in enumerate(p['bases']):
        out.append(dict(base_index=i, policy='O', status='inherited_openloop', parent=deepcopy(base)))
        for j, pol in enumerate(('I','F')):
            plan = plans[2*i+j]
            require(plan['base_index'] == i and plan['policy'] == pol, 'fixed policy slot identity')
            out.append(dict(base_index=i, policy=pol, status='new_model_plan', plan=plan))
    return out


def consume(output, report_sha, completion_sha):
    output = Path(output).absolute()
    r = read(output/'report.json', report_sha)
    p = verify(r['prepared'], r['prepare_sha256'])
    require(r['schema'] == NAME+'.report' and r['status'] == 'all8_feedback_sealed_NOT_admitted' and
            r['errors'] == [] and r['first_error'] is None and r['source_verified'] is True and
            r['model_before'] == r['model_after'] == p['expected_model_hash'], 'own complete engineering authority')
    for key in K.FLAGS:
        require(r[key] is False, 'scientific gates remain false')
    equal(r['contract'], K.contract(), 'immutable approved contract')
    equal(r['precision_before'], precision_expected(), 'actual math precision before')
    equal(r['precision_after'], precision_expected(), 'actual math precision after')
    equal(r['policy_roster'], roster(p,r['rows']), 'complete12 frozen policy accounting')
    equal(r['inherited_resources'], p['inherited_resources'], 'inherited and new costs remain separate')
    require(type(r['loader_memory']) is dict and set(r['loader_memory']) == {'allocated_peak', 'reserved_peak'} and
            all(type(r['loader_memory'][k]) is int and 0 <= r['loader_memory'][k] <= cap for k, cap in
                (('allocated_peak', K.contract()['allocated_cap_bytes']), ('reserved_peak', K.contract()['reserved_cap_bytes']))),
            'actual loader memory before slot counter reset')
    require(type(r['wall_seconds']) in (int, float) and np.isfinite(r['wall_seconds']) and 0 <= r['wall_seconds'] <= 1800, 'whole wall admission')
    completion = read(output/'completion.json', completion_sha)
    require(set(completion) == {'schema','report_sha256','boundary','elapsed_seconds'} and
            completion['schema'] == NAME+'.completion' and completion['report_sha256'] == report_sha and
            completion['boundary'] == 'after_report_fsync_and_hash' and
            type(completion['elapsed_seconds']) in (int,float) and np.isfinite(completion['elapsed_seconds']) and
            r['wall_seconds'] <= completion['elapsed_seconds'] <= 1800,
            'externally pinned completion covers report fsync/hash within whole budget')
    seal = read(output/'seal.json', sha(output/'seal.json'))
    require(seal['schema'] == NAME+'.seal' and seal['complete'] is True and
            type(seal['through_durable_data_seconds']) in (int, float) and
            np.isfinite(seal['through_durable_data_seconds']) and 0 <= seal['through_durable_data_seconds'] <= r['wall_seconds'],
            'durable data clock seal, no overwall complete')
    names = {'binding.json', 'loader.json', 'classes.json', 'events.jsonl.gz', 'seal.json'} | {f'slot{i}.json.gz' for i in range(8)} | {
        f'slot{i}_J{j}.npy' for i in range(8) for j in range(1)}
    require(set(r['artifact_sha256']) == names and {f.name for f in output.iterdir()} == names | {'report.json', 'completion.json'} and
            all(f.is_file() and not f.is_symlink() for f in output.iterdir()) and
            sum(f.stat().st_size for f in output.iterdir()) <= CAP, 'exact complete bounded inventory')
    for n, h in r['artifact_sha256'].items():
        require(sha(output/n) == h, 'all actual artifact hashes')
    equal(seal['data_artifacts_sha256'], {n: h for n, h in r['artifact_sha256'].items() if n != 'seal.json'}, 'report-bound seal complete data inventory')
    equal(read(output/'binding.json', r['artifact_sha256']['binding.json']),
          dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'actual own prepare binding')
    load = read(output/'loader.json', r['artifact_sha256']['loader.json'])
    native.loader.validate_loader(load, p, r['prepare_sha256'])
    cs = read(output/'classes.json', r['artifact_sha256']['classes.json'])
    require(type(cs) is dict and bool(cs) and all(p['bindings_sha256'].get(f) == h for f, h in cs.items()), 'all reported classes presealed')
    events, cursor = journal(output), 0
    equal(r['actual_resources'], census(events), 'durable attempted/returned census including AD primal')
    actual = r['actual_resources']
    require(actual['forward_attempted'] == actual['forward_returned'] == actual['predict_API_attempted'] == actual['predict_API_returned'] == 216 and
            actual['jacobian_attempted'] == actual['jacobian_returned'] == 8 and
            all(n == 216 for v in (actual['modules_attempted'], actual['modules_returned']) for n in v.values()), 'complete fixed module/AD census')
    def accept(e):
        nonlocal cursor
        require(cursor < len(events), 'missing lifecycle event')
        equal(events[cursor], e, 'full event chronology, unknown events rejected')
        cursor += 1
    for f in load['files']:
        accept(dict(kind='load_intent', path=f['path'], expected_sha256=f['sha256']))
        accept(dict(kind='consumed_buffer_frozen', receipt=f))
    def array(d):
        require(set(d) == {'file', 'sha256', 'shape', 'dtype'} and d['file'] in names and
                r['artifact_sha256'][d['file']] == d['sha256'] and d['shape'] == [192, 30] and d['dtype'] == 'float32', 'full J descriptor')
        data = (output/d['file']).read_bytes()
        stream = io.BytesIO(data)
        require(len(data) <= 1024**2 and np.lib.format.read_magic(stream) == (1, 0), 'bounded exact NPY v1.0')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        require(shape == (192, 30) and fortran is False and dtype == np.dtype('<f4') and
                len(data)-stream.tell() == 192*30*4, 'authenticate full J header before array allocation')
        stream.seek(0)
        a = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(data) and a.dtype == np.float32 and a.shape == (192, 30) and np.isfinite(a).all(), 'full J NPY bytes/dtype/EOF')
        return a
    require(len(r['rows']) == 8, 'all eight fixed jobs')
    plans, rebuilt = [], []
    for i, ss in enumerate(r['rows']):
        job = p['jobs'][i]
        root, arm = job['root_slot'], job['arm']
        require(type(ss['slot']) is int and type(ss['root_slot']) is int and ss['slot'] == i and ss['root_slot'] == root and ss['arm'] == arm and ss['policy'] == job['policy'] and ss['base_index'] == job['base_index'] and
                ss['status'] == 'sealed' and ss['file'] == f'slot{i}.json.gz' and
                type(ss['wall_seconds']) in (int, float) and np.isfinite(ss['wall_seconds']) and 0 <= ss['wall_seconds'] <= 450, 'fixed slot identity/wall')
        row = read(output/ss['file'], r['artifact_sha256'][ss['file']], CAP, gz=True)
        require(set(row) == {'slot', 'index', 'arm', 'status', 'forwards', 'AD', 'result', 'resources', 'plan', 'memory', 'base_index', 'policy'} and
                type(row['slot']) is int and type(row['index']) is int and row['slot'] == i and row['index'] == root and
                row['arm'] == arm and row['policy'] == job['policy'] and row['base_index'] == job['base_index'] and row['status'] == 'sealed', 'actual slot native row exact ABI')
        accept(dict(kind='slot_started', slot=i, root_slot=root, arm=arm, policy=job['policy'], base_index=job['base_index']))
        backend = SavedBackend(row, lambda e: accept(dict(slot=i, event=e)), array)
        result = K.execute(job, backend)
        backend.complete()
        equal(row['result'], result, 'all contexts/evaluations/guards/selection/replays independently reconstructed')
        equal(row['resources'], K.resources(row['forwards'], row['AD']), 'actual complete paid resource census')
        require(set(row['memory']) == {'allocated_peak', 'reserved_peak'} and
                all(type(row['memory'][k]) is int and 0 <= row['memory'][k] <= cap for k, cap in
                    (('allocated_peak', K.contract()['allocated_cap_bytes']), ('reserved_peak', K.contract()['reserved_cap_bytes']))),
                'actual slot allocator peak evidence')
        equal(ss['resources'], row['resources'], 'report resource identity')
        equal(ss['contexts'], result['contexts'], 'report full contexts')
        equal(ss['new_suffix_Q'], result['new_suffix_Q'], 'report Q')
        plan = row_plan(p, job, row)
        equal(row['plan'], plan, 'new full30 plan exact action and scope authority')
        rebuilt.append(result)
        accept(dict(kind='slot_sealed', slot=i, witness=plan['new_witness_sha256']))
        plans.append(plan)
    require(cursor == len(events), 'no extra terminal events')
    rehash(p['bindings_sha256'])
    require(sha(output/'report.json') == report_sha, 'report stable through consume')
    return dict(schema=NAME+'.consume', status='control15_feedback_verified_NOT_admitted',
                report_sha256=report_sha, prepare_sha256=r['prepare_sha256'], plans=plans, policy_slots=policy_slots(p, plans),
                full_upstream_rebuilt_here=False, own_native_saved_reconstructed_here=True,
                completion_sha256=completion_sha, whole_through_report_hash_seconds=completion['elapsed_seconds'],
                actual_new=K.contract()['total'], new_consume_model_calls=0, errors=[], **K.FLAGS)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    q=sub.add_parser('prepare');q.add_argument('--output', required=True)
    q=sub.add_parser('verify');q.add_argument('--prepared', required=True);q.add_argument('--prepare-sha', required=True)
    q=sub.add_parser('run');q.add_argument('--prepared', required=True);q.add_argument('--prepare-sha', required=True)
    q.add_argument('--output', required=True);q.add_argument('--authorize-host-cuda', action='store_true')
    q=sub.add_parser('consume');q.add_argument('--output', required=True);q.add_argument('--report-sha', required=True)
    q.add_argument('--completion-sha', required=True)
    q.add_argument('--receipt', required=True)
    a=ap.parse_args()
    if a.cmd=='prepare':
        write_new(a.output, build());result=dict(path=a.output,sha256=sha(a.output))
    elif a.cmd=='verify':
        verify(a.prepared,a.prepare_sha);result=dict(verified=True)
    elif a.cmd=='run':
        result=run_cuda(a.prepared,a.prepare_sha,a.output,AUTH if a.authorize_host_cuda else None)
    else:
        result=consume(a.output,a.report_sha,a.completion_sha);write_new(a.receipt,result);result=dict(path=a.receipt,sha256=sha(a.receipt))
    print(raw(result).decode(), end='')


if __name__=='__main__':
    main()

