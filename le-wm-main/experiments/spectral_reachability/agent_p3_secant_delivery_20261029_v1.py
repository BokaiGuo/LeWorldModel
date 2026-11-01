"""Four-slot secant delivery, exact independent native replays; MAIN CUDA only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_p3_reach_secant_actual_cp5_cuda_20261029_v1 as upstream

base, tree, old, core = upstream.base, upstream.tree, upstream.old, upstream.core
signal, live, prior = upstream.signal, upstream.live, upstream.source.prior
require, canonical, digest, sha = core.require, core.canonical, core.digest, core.sha
ROOT = core.ROOT
NAME = 'agent_p3_secant_delivery_20261029_v1'
PROTOCOL = ROOT/'innerdocs/agent_p3_secant_delivery_protocol_20261029_v1.md'
PROTOCOL_SHA = '500bfc08e0b542f365c6912dfa1ec6333276035a33ac596ec3f21bf3f1b3d065'
AUTHORIZATION = 'MAIN_APPROVED_SECANT_DELIVERY_HOST_CUDA'
AUTH = {
    'prepared': dict(path=str(ROOT/'results'/(upstream.NAME+'_prepare_MAIN01.json')),
                     sha256='b3cc19fe7361e6bcc03ebf533a9a5acc115438cb6efda2f3c5655e7f680799e6'),
    'report': dict(path=str(ROOT/'results'/(upstream.NAME+'_cuda_MAIN01/report.json')),
                   sha256='191684824c95b39fcaf4cd2683ad3f3f7cd071009cdaaa076014bd9f67df91e7'),
    'receipt': dict(path=str(ROOT/'results'/(upstream.NAME+'_consume_MAIN01.json')),
                    sha256='d96f97dd076894901edea43f6223cf8cb32da4dc630968fea8fae7834b143151'),
}
PINS = {Path(upstream.__file__): 'a7ecbb3fba3957af2382aecf363fbf6dd996204f8b39aedae1a486279a0b3c08',
        Path(upstream.kernel.__file__): '8fd9e2210d3d9ed6814cf07b2e9f85501156eb9733ab29ec2bfac908fbca2468',
        Path(prior.__file__): '09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26',
        PROTOCOL: PROTOCOL_SHA}
FLAGS = {k: v for k, v in upstream.FLAGS.items() if k != 'model_delivery_verified'}
ROLE, roster = upstream.ROLE, upstream.roster
SETTLED = ('model_delivery_verified', 'no_witness')
MODES = ('whole', 'segmented')


def caps(h):
    require(type(h) is int and h in (1, 3), 'original horizons')
    return prior.verification_cost(h)


def profile():
    return dict(upstream.profile(), resource_caps={str(h): caps(h) for h in (1, 3)})


def rules():
    return dict(protocol_sha256=PROTOCOL_SHA, groups=4, candidate_denominator=16,
        selection=['immediate', 'candidate_index'], root_context_role=ROLE, actual_cp5_start=True,
        all_source_consume_rebuilt=True, per_group_selected=1, no_reselection=True,
        seal_both_replays_before_comparison=True, segmented_raw_steps={'1': [5], '3': [5, 10]},
        expected_eligible_counts=[4, 0, 0, 0], current_predict_cap=2, all_group_predict_cap=16,
        commit_raw_steps=5, suffix_is_plan=False, **FLAGS)


def raw_array(value, h):
    a = core.finite(value, (5*h, 2), True)
    require(np.max(np.abs(a)) <= 1, 'raw domain, no clipping')
    return np.asarray(a, np.dtype('<f4'))


def raw_hash(value):
    return hashlib.sha256(np.asarray(value, np.dtype('<f4')).tobytes(order='C')).hexdigest()


def ranked(rows, zero):
    require(len(rows) == 4 and [x['index'] for x in rows] == list(range(4)), 'all four candidate slots')
    ids = []
    for x in rows:
        require(x['status'] == 'complete', 'full scored candidate, not unknown')
        i, r = upstream.kernel.score(x['immediate']), upstream.kernel.score(x['reach'])
        eligible = i > upstream.kernel.score(zero['immediate']) and r < upstream.kernel.score(zero['reach'])
        require(type(x['eligible']) is bool and x['eligible'] is eligible, 'unchanged exact I/R gate')
        if eligible: ids.append(x['index'])
    return sorted(ids, key=lambda i: (rows[i]['immediate'], i))


def shape_census():
    """Bound ALL four possible deliveries; no model/arrays needed to enumerate."""
    packet = [(1, 3, 192), (1, 2, 10), (5, 2), (2,), (2,), (10,), (1, 3, 10),
              (1, 3, 10), (1, 3, 192),                 # AE IO
              (1, 3, 192), (1, 3, 192), (1, 3, 192), # outer predict IO
              (1, 3, 192), (1, 3, 192), (1, 3, 192), # predictor IO
              (3, 192), (3, 192),                    # pred_proj IO
              (1, 3, 192), (1, 2, 10)]              # boundary
    payloads = [int(np.prod(s))*4 for s in packet]
    packets = sum(2*x['horizon'] for x in roster())
    # EXACT three persisted copies: native IO group_event, group_return's
    # native_calls, report.rows.native_calls. Verification/path fields are JSON
    # lists, not blobs. binding/completion contain no blobs; there are no NPYs.
    blob_bytes = packets*sum(4*((n+2)//3)+256 for n in payloads)
    metadata = packets*64*65536 + 4*2*1024**2 + 16*1024**2
    working = 3*blob_bytes+metadata
    reserve = profile()['final_reserve_bytes']
    return dict(packet_shapes=[list(s) for s in packet], blobs_per_packet=len(packet),
        max_packets=packets, native_payload_bytes=packets*sum(payloads), blob_copy_factor=3, NPY_count=0,
        max_persisted_blobs=3*packets*len(packet),
        expanded_blob_bytes=blob_bytes, metadata_upper_bytes=metadata, working_upper_bytes=working,
        final_reserve_bytes=reserve, total_upper_bytes=working+reserve,
        output_cap_bytes=profile()['output_bytes'], full_precision=True)


def source_material(p, r, receipt, rebuilt):
    require(canonical(receipt) == canonical(rebuilt), 'FULL persisted upstream receipt equals complete saved consume')
    for obj, suffix in ((p, 'prepare'), (r, 'report'), (receipt, 'consume')):
        require(obj['schema'] == upstream.NAME+'.'+suffix and all(obj[k] is False for k in upstream.FLAGS),
                'source actual own schema/false flags, not alias status')
    require(r['status'] == 'four_groups_settled_NOT_delivery' and r['errors'] == [] and r['delivery'] is None
        and receipt['status'] == 'all4_actual_cp5_secant_groups_verified_NOT_delivery' and receipt['delivery'] is None
        and receipt['groups'] == 4 and receipt['candidate_denominator'] == 16, 'complete four groups/sixteen candidates')
    require(r['prepared'] == receipt['prepared'] == AUTH['prepared']['path']
        and r['prepare_sha256'] == receipt['prepare_sha256'] == AUTH['prepared']['sha256']
        and receipt['report_sha256'] == AUTH['report']['sha256'], 'exact upstream three-way authority')
    require(r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources'])
        and r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash']
        and r['classes_before'] == r['classes_after'] and r['classes_before'], 'upstream full model/source closure')
    require(all(p['sources'][c['path']] == c['sha256'] for c in r['classes_before']), 'pinned actual model classes')
    require(p['roster'] == roster() and len(p['tasks']) == len(p['provenance']) == 2
        and len(r['rows']) == len(receipt['arms']) == len(p['source_groups']) == 4, 'original exact complete cohort')
    require(receipt['accounting'] == r['accounting'], 'whole upstream resource vector')
    inherited = core.counters(r['accounting']['cumulative_attempted'])
    require(inherited == core.counters(r['accounting']['cumulative_completed'])
        and inherited['predict_API'] == 276 and inherited['jacobian_API'] == 56, 'actual cumulative276/56, no invented zeros')
    groups = []
    for identity, row, arm, original in zip(roster(), r['rows'], receipt['arms'], p['source_groups'], strict=True):
        require(all(type(row[k]) is int and row[k] == v and arm[k] == v and original[k] == v
                    for k, v in identity.items()), 'fixed slot/root/horizon')
        task, pro = prior.input_pair(p['tasks'][row['root']], p['provenance'][row['root']])
        result = row['result']; h = row['horizon']
        require(row['status'] == result['status'] == arm['status'] == 'all_four_scored' and arm['complete'] is True
            and arm['scored'] == 4 and result['errors'] == [] and result['delivery'] is arm['delivery'] is None
            and result['task_sha256'] == digest(task) and result['provenance_sha256'] == digest(pro)
            and result['root_context_role'] == ROLE and result['actual_cp5_start'] is True
            and result['source_group_sha256'] == digest(original)
            and result['source_closure_sha256'] == digest(p['sources'])
            and all(result[k] is False and arm[k] is False for k in upstream.FLAGS), 'whole source group binding')
        order = ranked(result['rows'], original['zero'])
        require(sorted(order) == result['eligible_indices'] == arm['eligible_indices'], 'whole source eligibility indices')
        require(result['resource_receipt'] == arm['resources'] and arm['resources']['attempted'] == arm['resources']['completed'] == upstream.caps(h),
                'whole scored group actual cost')
        items = []
        for item in result['rows']:
            a = raw_array(item['actions'], h)
            core.checked_path(task, task['root_context'], item['actions'], item['path'])
            items.append({k: deepcopy(item[k]) for k in ('index', 'status', 'immediate', 'reach', 'eligible', 'params', 'radius', 'actions', 'path')}
                | dict(row_sha256=digest(item), raw_float32_sha256=raw_hash(a),
                       origin_pointer='/rows/'+str(row['slot'])+'/result/rows/'+str(item['index'])))
        groups.append(dict(**identity, task_sha256=digest(task), provenance_sha256=digest(pro),
            source_result_sha256=digest(result), source_closure_sha256=digest(p['sources']), zero=deepcopy(original['zero']),
            items=items, eligible_order=order, selected_index=None if not order else order[0]))
    require([len(g['eligible_order']) for g in groups] == [4, 0, 0, 0], 'pinned actual4/0/0/0; never change roster to fit')
    return groups, inherited


def load_source():
    """Production MAIN only: full upstream consume, including its own closure."""
    p = upstream.read_pin(AUTH['prepared'], 16*1024**2)
    r = upstream.read_pin(AUTH['report'], profile()['final_reserve_bytes'])
    receipt = upstream.read_pin(AUTH['receipt'], 128*1024**2)
    rebuilt = upstream.consume(Path(AUTH['report']['path']).parent, AUTH['report']['sha256'],
                               AUTH['prepared']['path'], AUTH['prepared']['sha256'])
    groups, inherited = source_material(p, r, receipt, rebuilt)
    require(all(sha(v['path']) == v['sha256'] for v in AUTH.values()), 'unchanged complete original authorities')
    return p, groups, inherited


def build():
    require(all(sha(f) == h for f, h in PINS.items()), 'frozen approved protocol and exact low-level reuse')
    p, groups, inherited = load_source()
    census = shape_census()
    require(census['total_upper_bytes'] < profile()['output_bytes']
        and all(len(canonical(g)) <= 2*1024**2 for g in groups)
        and all(t['root_context']['latent_dim'] == 192 and len(canonical(t)) <= 256*1024 for t in p['tasks'])
        and len(canonical({k: p[k] for k in ('sources', 'provenance', 'cp5_tasks')})) <= 16*1024**2,
        'full shape-byte census plus bounded task/group/source metadata')
    sources = dict(p['sources'])
    files = [Path(__file__), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'), *PINS]
    for f in files:
        h = sha(f); require(str(f) not in sources or sources[str(f)] == h, 'no frozen source override'); sources[str(f)] = h
    sources.update({v['path']: v['sha256'] for v in AUTH.values()})
    require(all(sha(f) == h for f, h in sources.items()), 'all original/new source bytes')
    return dict(schema=NAME+'.prepare', tasks=p['tasks'], provenance=p['provenance'], groups=groups,
        cp5_tasks=p['cp5_tasks'], policy=p['policy'], expected_model_hash=p['expected_model_hash'], sources=sources,
        authorities=deepcopy(AUTH), upstream_source_manifest_sha256=digest(p['sources']),
        inherited_resources=inherited, roster=roster(), profile=profile(), precision=old.precision_contract(),
        shape_census=census, rules=rules(), complete_upstream_receipt_rebuilt=True, **FLAGS)


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and p.resolve() == p
        and not p.exists() and not p.is_symlink(), 'new exclusive own output')
    return p


def write_new(path, value, cap=16*1024**2):
    p = exclusive(path); b = canonical(value)
    require(len(b) <= cap and shutil.disk_usage(p.parent).free >= profile()['disk_floor_bytes']+len(b), 'metadata cap/floor')
    with p.open('xb') as f:
        require(f.write(b) == len(b), 'full metadata write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent); return dict(path=str(p), sha256=sha(p))


def prepare(output): return write_new(output, build())


def verify(prepared, expected):
    p = old.regular(prepared, 16*1024**2); require(sha(p) == expected, 'actual prepare SHA')
    value = signal.read(p); require(canonical(value) == canonical(build()) and sha(p) == expected, 'complete upstream consume and own prepare rebuild')
    return value


def blob_stats(value):
    """Actual base64 bytes, not neural math; list-valued metadata is not a blob."""
    if type(value) is dict:
        if set(value) == {'shape', 'dtype', 'sha256', 'data'}:
            a = signal.unblob(value)
            require(a.dtype == np.float32 and len(canonical(value))-len(value['data']) <= 256, 'bounded full-precision blob header')
            return 1, len(value['data'])
        children = value.values()
    elif type(value) is list: children = value
    else: return 0, 0
    counts = [blob_stats(v) for v in children]
    return sum(v[0] for v in counts), sum(v[1] for v in counts)


class Store(base.Store):
    """Same durable codec with enforced aggregate metadata/blob census quotas."""
    def __init__(self, output, limits):
        super().__init__(output, limits); self.persisted_blobs = self.blob_bytes = self.metadata_bytes = 0
    def census_guard(self, value, encoded_size):
        count, data = blob_stats(value); c = shape_census()
        require(self.persisted_blobs+count <= c['max_persisted_blobs'] and
            self.blob_bytes+data <= 3*c['expanded_blob_bytes'] and
            self.metadata_bytes+encoded_size-data <= c['metadata_upper_bytes'], 'exact persisted blob/metadata census cap')
        self.persisted_blobs += count; self.blob_bytes += data; self.metadata_bytes += encoded_size-data
    def event(self, value):
        envelope = dict(index=self.index, previous=self.tail, payload=deepcopy(value))
        envelope['sha256'] = digest(envelope)
        self.census_guard(value, len(canonical(envelope))+1); super().event(value)
    def put(self, name, value):
        b = canonical(value); self.census_guard(value, len(b))
        if name not in ('completion.json', 'completion_failure.json'): return super().put(name, value)
        self.check(len(b), final=True)
        with (self.output/name).open('xb') as f:
            require(f.write(b) == len(b), 'complete completion publication'); f.flush(); os.fsync(f.fileno())
        self.bytes += len(b); signal.fsync_dir(self.output)


def publish_report(store, report, expected, started, *, clock=time.monotonic):
    """Saved-file-only publication, exercised with the REAL Store in fixtures."""
    store.put('report.json', report)
    # Store.put fsyncs file and directory. Hash ALL published artifacts before
    # the durable completion checkpoint; never use only report's earlier wall.
    published = {n: sha(store.output/n) for n in ('binding.json', 'events.jsonl', 'report.json')}
    inclusive = clock()-started
    completion = dict(schema=NAME+'.completion', prepared=report['prepared'], prepare_sha256=expected,
        report_sha256=published['report.json'], inclusive_wall_seconds=inclusive,
        wall_scope='start through report durable publication and SHA256; completion self-publication excluded',
        status='report_published_within_cap' if inclusive <= store.limits['wall_seconds'] else 'publication_wall_failed',
        artifact_sha256=published, bytes_before_completion=sum(x.stat().st_size for x in store.output.iterdir()), **FLAGS)
    store.put('completion.json', completion); completion_sha = sha(store.output/'completion.json')
    finished = clock()-started
    # Cannot encode a timestamp after a receipt's own last fsync in that same
    # receipt. Enforce final publication too; a failed final guard adds a durable
    # extra artifact, so complete-census consumers MUST refuse this directory.
    if finished > store.limits['wall_seconds']:
        store.put('completion_failure.json', dict(schema=NAME+'.publication_failure',
            elapsed_seconds=finished, report_sha256=published['report.json'], completion_sha256=completion_sha,
            stage='completion_write_fsync_hash', **FLAGS))
    require(finished <= store.limits['wall_seconds'], 'whole publication wall failed; durable failure retained')
    return dict(output=str(store.output), report_sha256=published['report.json'], completion_sha256=completion_sha)


def same_array(a, b):
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape and a.tobytes() == b.tobytes(), 'native array exact dtype/shape/bits')


def bit_audit(call):
    """Strengthen frozen audit_native with exact bytes across every program IO."""
    require(old.audit_native(call) is True, 'original full native IO closure')
    task, req = call[0]['task'], call[0]['request']; current = deepcopy(req['parent_context'])
    body = call[1:-1]; packets = np.asarray(req['actions'], np.float32).reshape(-1, 5, 2)
    for step, raw in enumerate(packets):
        scope = [e for e in body if e.get('step') == step]
        def arrays(kind):
            es = [e for e in scope if e['kind'] == kind]
            require(len(es) == 1, 'one exact packet/boundary per step')
            return {k: signal.unblob(v) for k, v in es[0]['arrays'].items()}
        pp, boundary = arrays('program_packet'), arrays('program_boundary')
        latent = np.asarray(current['latent_history'], np.float32)[None]
        hist = np.asarray(current['action_history'], np.float32)[None]
        mean, std = (np.asarray(task[k], np.float32) for k in ('action_mean', 'action_std'))
        token = ((raw-mean)/std).reshape(10); outgoing = np.concatenate((hist, token[None, None]), axis=1)
        for k, a in dict(latent=latent, history=hist, raw=raw, mean=mean, std=std, token=token, outgoing=outgoing).items(): same_array(pp[k], a)
        def io(kind, key):
            found = [signal.unblob(e['arrays'][key]) for e in scope if e['kind'] == kind and key in e['arrays']]
            require(len(found) == 1, 'unique actual module input/output'); return found[0]
        same_array(io('io_input', 'action_encoder:0'), outgoing)
        ae = io('io_return', 'action_encoder:output')
        same_array(io('io_input', 'predict:latent'), latent); same_array(io('io_input', 'predict:act_emb'), ae)
        same_array(io('io_input', 'predictor:0'), latent); same_array(io('io_input', 'predictor:1'), ae)
        predictor = io('io_return', 'predictor:output'); same_array(io('io_input', 'pred_proj:0'), predictor.reshape(3, 192))
        full = io('io_return', 'pred_proj:output').reshape(1, 3, 192); same_array(io('io_return', 'predict:output'), full)
        next_latent = np.concatenate((latent[:, 1:], full[:, -1:]), axis=1)
        same_array(boundary['latent'], next_latent); same_array(boundary['history'], outgoing[:, 1:])
        current = dict(current, latent_history=next_latent[0].tolist(), action_history=outgoing[0, 1:].tolist())
        require(canonical(current) == canonical(call[-1]['result'][step]), 'exact full postcontext bytes')
    return True


def make_delivery(task, pro, g, verification, source_sha):
    selected = g['items'][g['selected_index']]; a = raw_array(selected['actions'], g['horizon'])
    return dict(schema=NAME+'.delivery', model_delivery_verified=True, candidate_index=selected['index'],
        eligible_order=deepcopy(g['eligible_order']), selection_key=[selected['immediate'], selected['index']],
        source_row_sha256=selected['row_sha256'], source_row_pointer=selected['origin_pointer'],
        task=deepcopy(task), task_sha256=digest(task), base_task=deepcopy(pro['original_task']),
        base_task_sha256=pro['original_task_sha256'], provenance_sha256=digest(pro),
        source_closure_sha256=source_sha, root_context_role=ROLE, raw_edge=a.tolist(), raw_edge_float32_sha256=raw_hash(a),
        commit_raw5=a[:5].tolist(), commit_float32_sha256=raw_hash(a[:5]), lookahead_raw=a[5:].tolist(),
        lookahead_float32_sha256=raw_hash(a[5:]), path=deepcopy(selected['path']), control_start=5,
        commit_control_steps=[6, 7, 8, 9, 10], prediction_end_control=5+5*g['horizon'],
        prediction_end_absolute=15+5*g['horizon'], suffix_in_plan=False, verification=deepcopy(verification), **FLAGS)


def group(task, pro, g, factory, sink, *, source_sha, permit_ram=False, guard=lambda: None,
          saved_calls=None, clock=time.monotonic_ns):
    task, pro = prior.input_pair(task, pro); h = g['horizon']
    require(g['task_sha256'] == digest(task) and g['provenance_sha256'] == digest(pro)
        and ranked(g['items'], g['zero']) == g['eligible_order']
        and g['selected_index'] == (g['eligible_order'][0] if g['eligible_order'] else None), 'source identity and fixed selection, no reselect')
    kind = 'RAM_FIXTURE' if permit_ram else 'HOST_CUDA'; ledger = core.Ledger(caps(h), clock=clock)
    session = tree.Session(task, factory, sink, ledger, kind, guard, None, saved_calls); ledger.sink = session.event
    r = dict(schema=NAME+'.group', backend_kind=kind, status='not_started', source_group_sha256=digest(g),
        task_sha256=digest(task), provenance_sha256=digest(pro), eligible_order=deepcopy(g['eligible_order']),
        selected_index=g['selected_index'], model_delivery_verified=False, delivery=None, errors=[],
        verification=[dict(mode=mode, segments=[5*h] if mode == 'whole' else ([5] if h == 1 else [5, 10]),
                           status='not_requested', call_id=None, path=None) for mode in MODES], **FLAGS)
    try:
        guard(); session.event(dict(kind='delivery_selection', source_group_sha256=digest(g), order=g['eligible_order'], selected_index=g['selected_index']))
        if g['selected_index'] is None:
            for v in r['verification']: v['status'] = 'not_applicable_no_witness'
            r['status'] = 'no_witness'
        else:
            selected = g['items'][g['selected_index']]; actions = raw_array(selected['actions'], h)
            require(raw_hash(actions) == selected['raw_float32_sha256'], 'literal sealed raw reloaded, no remapping')
            session.event(dict(kind='raw_actions_reloaded', raw=actions.tolist(), raw_sha256=raw_hash(actions)))
            for v in r['verification']:
                guard(); ledger.scope = dict(phase='delivery_verification', horizon=h, mode=v['mode'], segments=v['segments'])
                v.update(status='attempted', call_id=len(session.records))
                session.event(dict(kind='delivery_verification_attempt', verification=deepcopy(v)))
                path = ledger.call('replay', lambda: session.replay(task, actions.tolist(), v['mode'] == 'segmented'))
                v.update(status='returned', path=deepcopy(path)); session.event(dict(kind='delivery_verification_return', verification=deepcopy(v)))
            session.event(dict(kind='both_predictions_sealed'))
            for v, call in zip(r['verification'], session.records, strict=True):
                bit_audit(call['transcript'])
                require(canonical(v['path']) == canonical(selected['path']), 'whole and segmented exact original selected path, no tolerance')
            r.update(status='model_delivery_verified', model_delivery_verified=True,
                delivery=make_delivery(task, pro, g, r['verification'], source_sha))
            session.event(dict(kind='model_delivery', delivery=deepcopy(r['delivery'])))
        expected = caps(h) if g['selected_index'] is not None else tree.empty_axes()
        require(ledger.attempted == ledger.completed == expected, 'all actual verification costs; no padding')
        guard()
    except Exception as ex:
        r.update(status='partial_runtime_failure', errors=[dict(type=type(ex).__name__, message=str(ex))], model_delivery_verified=False, delivery=None)
        for v in r['verification']:
            if v['status'] == 'attempted': v['status'] = 'failed'
    r.update(native_calls=deepcopy(session.records), resource_receipt=ledger.receipt())
    core.audit_ledger(r['resource_receipt']); return r


def execute(p, factory, store, *, permit_ram=False, check=lambda: None):
    rows = [dict(**x, status='not_requested', result=None) for x in roster()]; failed = False
    store.event(dict(kind='denominator', rows=deepcopy(rows), candidates=16))
    for row in rows:
        if failed: row['status'] = 'blocked_after_runtime_failure'; continue
        slot = row['slot']; started = time.monotonic(); begin = getattr(store, 'bytes', 0)
        store.event(dict(kind='group_attempt', identity=roster()[slot]))
        sink = type('Sink', (), {'event': lambda self, e: store.event(dict(kind='group_event', slot=slot, payload=deepcopy(e)))})()
        def guard():
            check(); require(time.monotonic()-started <= p['profile']['group_wall_seconds'], 'per group wall')
        try:
            result = group(p['tasks'][row['root']], p['provenance'][row['root']], p['groups'][slot], factory, sink,
                source_sha=digest(p['sources']), permit_ram=permit_ram, guard=guard)
            row.update(status=result['status'], result=result, wall_seconds=time.monotonic()-started); failed = bool(result['errors'])
            store.event(dict(kind='group_return', row=deepcopy(row)))
            require(getattr(store, 'bytes', 0)-begin+len(canonical(row)) <= p['profile']['group_output_bytes'], 'full group output cap')
            guard()
        except Exception as ex:
            failed = True; row.update(status='outer_runtime_failure', error=dict(type=type(ex).__name__, message=str(ex)))
            store.event(dict(kind='group_outer_failure', row=deepcopy(row)))
    return rows


def accounting(p, rows):
    attempted = tree.sum_axes(*[x['result']['resource_receipt']['attempted'] for x in rows if x['result'] is not None])
    completed = tree.sum_axes(*[x['result']['resource_receipt']['completed'] for x in rows if x['result'] is not None])
    current = tree.sum_axes(*[caps(g['horizon']) for g in p['groups'] if g['selected_index'] is not None])
    return dict(inherited=deepcopy(p['inherited_resources']), new_attempted=attempted, new_completed=completed,
        new_caps=current, all_group_max=tree.sum_axes(*[caps(x['horizon']) for x in roster()]),
        cumulative_attempted=tree.sum_axes(p['inherited_resources'], attempted),
        cumulative_completed=tree.sum_axes(p['inherited_resources'], completed), scalar_backward_calls=None, total_FLOPs=None)


def run(prepared, expected, output, *, authorization):
    require(authorization == AUTHORIZATION, 'MAIN audit and explicit HOST CUDA authorization')
    started = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= p['profile']['output_bytes']+p['profile']['disk_floor_bytes'], 'whole output cap plus floor')
    store = Store(output, p['profile']); store.started = started
    rows = [dict(**x, status='blocked_before_model', result=None) for x in roster()]
    model = None; errors = []; before = after = classes = classes_after = host = None; guards = []; source_ok = False
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
        store.event(dict(kind='source_before', sha256=digest(p['sources'])))
        # Reload sealed selection/raw directly from the saved prepare, no new mapping.
        reread = signal.read(prepared); require(sha(prepared) == expected and reread['groups'] == p['groups'], 'actual saved selection/raw reread')
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        store.memory = lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        store.check(); model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        before = signal.model_hash(model); require(before['original_state_dict'] == p['expected_model_hash'], 'original frozen model')
        classes = old.model_sources(model, p); store.event(dict(kind='model_before', hashes=before, classes=classes))
        forwards = [(m, 'forward' in m.__dict__, m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda', enabled=False), torch.autocast(device_type='cpu', enabled=False), sdpa_kernel(SDPBackend.MATH):
            def guard(): store.check(); old.checked_precision(torch)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_before', value=guards[-1]))
            rows = execute(p, lambda t, s: prior.NativeDelivery(model, device, t, s), store, check=guard)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_after', value=guards[-1]))
        torch.cuda.synchronize(device)
        require(all(('forward' in m.__dict__) == had and m.__dict__.get('forward') is v for m, had, v in forwards), 'all instance hooks restored')
    except BaseException as ex: errors.append(dict(stage='run', type=type(ex).__name__, message=str(ex)))
    finally:
        if model is not None:
            try:
                after = signal.model_hash(model); classes_after = old.model_sources(model, p)
                require(before == after and classes == classes_after and all(not m.training for m in model.modules())
                    and all(not x.requires_grad for x in model.parameters()), 'model/classes unchanged')
                store.event(dict(kind='model_after', hashes=after, classes=classes_after))
            except BaseException as ex: errors.append(dict(stage='model_final', message=str(ex)))
        try:
            verify(prepared, expected); store.check(); source_ok = True
            store.event(dict(kind='source_after', sha256=digest(p['sources'])))
        except BaseException as ex: errors.append(dict(stage='source_final', message=str(ex)))
        store.close(); settled = all(x['status'] in SETTLED for x in rows)
        report = dict(schema=NAME+'.report', status='all4_secant_delivery_settled_NOT_physical' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()), prepare_sha256=expected, rows=rows, errors=errors,
            model_before=before, model_after=after, classes_before=classes, classes_after=classes_after, host=host, precision=guards,
            sources_verified=source_ok, source_manifest_sha256=digest(p['sources']), wall_seconds=store.clock()-store.started,
            CUDA_peak_bytes=store.memory() if store.memory else None, journal_tail=store.tail,
            artifact_sha256={n: sha(output/n) for n in ('binding.json', 'events.jsonl') if (output/n).exists()},
            accounting=accounting(p, rows), **FLAGS)
        publication = publish_report(store, report, expected, started)
    require(report['status'] == 'all4_secant_delivery_settled_NOT_physical', 'partial saved, no retry '+repr(errors))
    return publication


def validate_group(task, pro, g, r, events, *, source_sha, permit_fixture=False):
    require(r['backend_kind'] == 'HOST_CUDA' or permit_fixture and r['backend_kind'] == 'RAM_FIXTURE', 'no RAM production')
    require(all(e['kind'] == 'io' for e in events), 'native IO envelopes only')
    payloads = [e['payload'] for e in events]
    allowed = old.RESOURCE | old.CALL_ARRAYS | {'native_call_attempt', 'native_call_return', 'native_call_failure',
        'delivery_selection', 'raw_actions_reloaded', 'delivery_verification_attempt', 'delivery_verification_return', 'both_predictions_sealed', 'model_delivery'}
    require(all(e['kind'] in allowed for e in payloads), 'no unknown event')
    require(r['resource_receipt']['caps'] == caps(g['horizon']), 'exact independent group cap')
    old.resource_census(r['resource_receipt'], payloads)
    calls = r['native_calls']; require([c['call_id'] for c in calls] == list(range(len(calls))), 'all native calls in order')
    require(all(e['call'] is None or type(e['call']) is int and 0 <= e['call'] < len(calls) for e in events), 'no orphan IO')
    for c in calls:
        require([e['payload'] for e in events if e['call'] == c['call_id']] == c['transcript'], 'complete exact native transcript')
        bit_audit(c['transcript'])
    saved = []; sink = type('Sink', (), {'event': lambda self, e: saved.append(deepcopy(e))})()
    rebuilt = group(task, pro, g, None, sink, source_sha=source_sha, permit_ram=permit_fixture,
        saved_calls=calls, clock=tree.replay_clock(r['resource_receipt'], payloads))
    require(canonical(rebuilt) == canonical(r), 'entire saved delivery reconstructed, not status trust')
    require([e for e in saved if e['call'] is None] == [e for e in events if e['call'] is None], 'all selection/seal/comparison events exact order')
    require(r['status'] in SETTLED and not r['errors'] and len(calls) == (2 if g['selected_index'] is not None else 0), 'all dual calls or explicit no witness')
    return dict(slot=g['slot'], root=g['root'], horizon=g['horizon'], status=r['status'], complete=True,
        no_witness=r['status'] == 'no_witness', model_delivery_verified=r['model_delivery_verified'], delivery=r['delivery'],
        eligible_order=r['eligible_order'], selected_index=r['selected_index'], resources=r['resource_receipt'], **FLAGS)


def validate_rows(p, rows, events, *, permit_fixture=False):
    require(len(rows) == 4 and [(r['slot'], r['root'], r['horizon']) for r in rows] ==
        [(r['slot'], r['root'], r['horizon']) for r in roster()], 'full fixed four groups')
    top = {'denominator', 'group_attempt', 'group_return', 'source_before', 'source_after',
        'model_before', 'model_after', 'precision_before', 'precision_after'}
    require(all(e['kind'] in top | {'group_event'} for e in events), 'no unknown/outer failure event')
    require(all(type(e['slot']) is int and 0 <= e['slot'] < 4 for e in events if e['kind'] == 'group_event'), 'no orphan slot')
    require([e for e in events if e['kind'] == 'denominator'] == [dict(kind='denominator', candidates=16,
        rows=[dict(**x, status='not_requested', result=None) for x in roster()])], 'full initial16 denominator')
    out = []; position = -1
    for row, identity in zip(rows, roster(), strict=True):
        starts = [i for i, e in enumerate(events) if e == dict(kind='group_attempt', identity=identity)]
        ends = [i for i, e in enumerate(events) if e == dict(kind='group_return', row=row)]
        require(row['result'] is not None and row['status'] == row['result']['status'] in SETTLED
            and len(starts) == len(ends) == 1 and position < starts[0] < ends[0], 'all four ordered settled groups')
        es = [e['payload'] for i, e in enumerate(events) if e['kind'] == 'group_event' and e['slot'] == row['slot']]
        require(all(starts[0] < i < ends[0] for i, e in enumerate(events) if e['kind'] == 'group_event' and e['slot'] == row['slot']), 'all IO within group')
        require(type(row['wall_seconds']) in (int, float) and 0 <= row['wall_seconds'] <= p['profile']['group_wall_seconds'], 'group wall')
        position = ends[0]
        out.append(validate_group(p['tasks'][row['root']], p['provenance'][row['root']], p['groups'][row['slot']], row['result'], es,
            source_sha=digest(p['sources']), permit_fixture=permit_fixture))
    require(sum(e['kind'] == 'group_attempt' for e in events) == sum(e['kind'] == 'group_return' for e in events) == 4, 'no hidden dispatch')
    return out


def consume(output, report_sha, prepared, prepare_sha, completion_sha, *, _fixture=False):
    output = Path(output).absolute(); require(output.resolve() == output and not output.is_symlink(), 'exact own output')
    rp = old.regular(output/'report.json', profile()['final_reserve_bytes']); require(sha(rp) == report_sha, 'actual report SHA')
    r = signal.read(rp, profile()['final_reserve_bytes']); p = verify(prepared, prepare_sha)
    require(r['schema'] == NAME+'.report' and r['status'] == 'all4_secant_delivery_settled_NOT_physical' and not r['errors']
        and r['prepared'] == str(Path(prepared).absolute()) and r['prepare_sha256'] == prepare_sha
        and all(r[k] is False for k in FLAGS) and r['sources_verified'] is True
        and r['source_manifest_sha256'] == digest(p['sources']), 'complete own source/authority/science boundary')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash']
        and r['classes_before'] == r['classes_after'] and r['classes_before'], 'frozen model/classes')
    require(all(p['sources'][c['path']] == c['sha256'] for c in r['classes_before']), 'actual class source pins')
    require(r['precision'] == [old.precision_contract()]*2 and r['host']['device'] == 'cuda:0'
        and r['host']['allocation_sum'] == 4. and r['host']['visible_nvidia_nodes'], 'HOST CUDA and observed precision guards')
    require(set(r['artifact_sha256']) == {'binding.json', 'events.jsonl'} and
        {x.name for x in output.iterdir()} == {'binding.json', 'events.jsonl', 'report.json', 'completion.json'}, 'all and only actual artifacts')
    cp = old.regular(output/'completion.json', 1024**2); core.hash_value(completion_sha)
    require(sha(cp) == completion_sha, 'external completion SHA')
    completion = signal.read(cp, 1024**2)
    require(completion['schema'] == NAME+'.completion' and completion['prepared'] == r['prepared']
        and completion['prepare_sha256'] == prepare_sha and completion['report_sha256'] == report_sha
        and completion['status'] == 'report_published_within_cap' and all(completion[k] is False for k in FLAGS)
        and completion['wall_scope'] == 'start through report durable publication and SHA256; completion self-publication excluded'
        and type(completion['inclusive_wall_seconds']) in (int, float)
        and 0 <= r['wall_seconds'] <= completion['inclusive_wall_seconds'] <= p['profile']['wall_seconds'],
        'post-report durable/hash inclusive wall completion binding')
    require(set(completion['artifact_sha256']) == {'binding.json', 'events.jsonl', 'report.json'} and
        all(sha(old.regular(output/n, profile()['output_bytes'])) == h for n, h in completion['artifact_sha256'].items())
        and completion['bytes_before_completion'] == sum((output/n).stat().st_size for n in completion['artifact_sha256']), 'complete completion census')
    for n, h in r['artifact_sha256'].items(): require(sha(old.regular(output/n, profile()['output_bytes'])) == h, 'complete saved artifact SHA')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=prepare_sha), 'persisted binding')
    sizes = {}; events = base.journal(output, r, sizes)
    with (output/'events.jsonl').open('rb') as stream:
        for line in stream:
            require(line == canonical(json.loads(line, object_pairs_hook=signal.unique_pairs))+b'\n', 'canonical exactly-one-newline codec')
    for phase in ('before', 'after'):
        require([e for e in events if e['kind'] == 'source_'+phase] == [dict(kind='source_'+phase, sha256=digest(p['sources']))], 'source events')
        require([e for e in events if e['kind'] == 'model_'+phase] == [dict(kind='model_'+phase, hashes=r['model_'+phase], classes=r['classes_'+phase])], 'model events')
        require([e['value'] for e in events if e['kind'] == 'precision_'+phase] == [old.precision_contract()], 'precision events')
    arms = validate_rows(p, r['rows'], events, permit_fixture=_fixture)
    require(r['accounting'] == accounting(p, r['rows']), 'all27 resources, inherited/new/cumulative')
    require(all(r['accounting']['new_attempted'][k] == r['accounting']['new_completed'][k] == r['accounting']['new_caps'][k] for k in core.AXES), 'all actual dual replay charges returned')
    require(0 <= r['wall_seconds'] <= p['profile']['wall_seconds'] and sum(x.stat().st_size for x in output.iterdir()) <= p['profile']['output_bytes'], 'whole byte/wall cap')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][k]
        for v, k in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'memory caps')
    require(all(sizes[x['slot']]+len(canonical(x)) <= p['profile']['group_output_bytes'] for x in r['rows']), 'group byte caps')
    count = data = metadata = 0
    for n in ('binding.json', 'report.json', 'completion.json'):
        obj = signal.read(output/n, profile()['final_reserve_bytes']); c, b = blob_stats(obj)
        count += c; data += b; metadata += (output/n).stat().st_size-b
    with (output/'events.jsonl').open('rb') as stream:
        for line in stream:
            c, b = blob_stats(json.loads(line)); count += c; data += b; metadata += len(line)-b
    census = shape_census()
    require(count <= census['max_persisted_blobs'] and data <= 3*census['expanded_blob_bytes']
        and metadata <= census['metadata_upper_bytes'], 'complete actual serialization census')
    require(sha(rp) == report_sha and sha(cp) == completion_sha and all(sha(f) == h for f, h in p['sources'].items()), 'unchanged final complete closure')
    return dict(schema=NAME+('.RAM_FIXTURE.consume' if _fixture else '.consume'),
        status='RAM_FIXTURE_ONLY_NOT_PRODUCTION' if _fixture else 'all4_secant_model_deliveries_verified_NOT_physical',
        prepared=r['prepared'], prepare_sha256=prepare_sha, report_sha256=report_sha, protocol_sha256=PROTOCOL_SHA,
        completion_sha256=completion_sha, completion=completion, serialized_census=dict(blobs=count, base64_bytes=data, metadata_bytes=metadata, NPY_count=0),
        authorities=deepcopy(AUTH), groups=4, candidate_denominator=16, inherited_candidates=32, inherited_controls=4,
        arms=arms, accounting=r['accounting'], cp5_tasks=p['cp5_tasks'], root_context_role=ROLE, **FLAGS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--output', required=True)
    for cmd in ('verify', 'run', 'consume'):
        q = sub.add_parser(cmd); q.add_argument('--prepared', required=True); q.add_argument('--prepare-sha', required=True)
        if cmd in ('run', 'consume'): q.add_argument('--output', required=True)
        if cmd == 'run': q.add_argument('--authorization', required=True)
        if cmd == 'consume':
            q.add_argument('--report-sha', required=True); q.add_argument('--completion-sha', required=True); q.add_argument('--receipt', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'prepare': out = prepare(a.output)
    elif a.cmd == 'verify': out = dict(verified=True, schema=verify(a.prepared, a.prepare_sha)['schema'])
    elif a.cmd == 'run': out = run(a.prepared, a.prepare_sha, a.output, authorization=a.authorization)
    else: out = write_new(a.receipt, consume(a.output, a.report_sha, a.prepared, a.prepare_sha, a.completion_sha), 128*1024**2)
    print(canonical(out).decode())


if __name__ == '__main__': main()
