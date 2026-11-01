"""Saved-only P4 rolling/visual seam check, supplementary to MAIN's strict audit.

No producer import, torch, model, renderer, environment, or command execution.
External report/prepare SHAs are mandatory. Stdout only; no result writes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np

NAME = 'agent_p4_rolling_saved_consume_20261028_v1'
PRODUCER = 'agent_p4_moving_multiboundary_20261028_v1'
PRODUCER_SHA = '2776029cc98f8e7d80bb566183ecdd5b44d8e4a0cd25e3f85fff23056b4bb68a'
ARMS = ('line_search_GN', 'full10_conditional_D2P')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def object_sha(value):
    # Original source.digest: spaced sorted JSON, NO newline.
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def packet_sha(value):
    # Original physical.packet_digest: compact sorted JSON WITH newline.
    return digest((canonical(value)+'\n').encode())


def pairs(items):
    out = {}
    for key, value in items:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out


def decode(payload):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)


class Reader:
    """Bounded, read-once snapshot; binds every file actually used, no writes."""
    def __init__(self):
        self.inputs = {}

    def blob(self, path, expected=None, cap=64*1024**2):
        path = Path(path).absolute()
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'no symlink input')
        require(path.is_file() and path.stat().st_size <= cap, 'bounded existing input: '+str(path))
        payload = path.read_bytes()
        require(len(payload) <= cap, 'bounded input after read')
        h = digest(payload)
        require(expected is None or h == expected, 'input SHA: '+str(path))
        require(str(path) not in self.inputs or self.inputs[str(path)] == h, 'input changed during read')
        self.inputs[str(path)] = h
        return payload

    def json(self, path, expected=None):
        payload = self.blob(path, expected)
        if str(path).endswith('.gz'):
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
                payload = stream.read(8*1024**2+1)
            require(len(payload) <= 8*1024**2, 'bounded decoded JSON')
        return decode(payload)


def array(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite exact array shape')
    f = a.astype(np.float32)
    require(np.array_equal(a, f), 'literal float32 representable values')
    return f


def exact(actual, expected, message):
    b = np.asarray(expected, dtype=np.float32)
    a = array(actual, b.shape)
    require(a.tobytes() == b.tobytes(), message)


def normalize(raw10, task):
    a = array(raw10, (10, 2))
    mean = array(task['action_mean'], (2,))
    std = array(task['action_std'], (2,))
    require((std > 0).all(), 'positive original normalizer')
    return ((a-mean)/std).reshape(2, 10).tolist()


def rolled(context, latent, raw10, task):
    out = deepcopy(context)
    out['latent_history'] = deepcopy(context['latent_history'][1:])+[array(latent, (192,)).tolist()]
    out['action_history'] = normalize(raw10, task)
    return out


def same_context(actual, expected):
    require(set(actual) == set(expected), 'exact context fields')
    for key in expected:
        if key in ('latent_history', 'action_history'):
            exact(actual[key], expected[key], 'own rolling context: '+key)
        else:
            require(actual[key] == expected[key], 'unchanged context layout')


def one(events, kind, **identity):
    found = [e for e in events if e['kind'] == kind and all(e.get(k) == v for k, v in identity.items())]
    require(len(found) == 1, 'exactly one '+kind+' '+str(identity))
    return found[0]


def encoding(reader, output, row, step, layer, models, tensor_sha, events):
    """Check only rolling-boundary alias/encoding linkage, not all native PNGs."""
    eid = row['episode_id']; ns = eid+f'_b{step:02}'
    receipt = reader.json(output/(ns+'_encoding.json'))
    entries = [e for e in row['encodings'] if e['step'] == step]
    require(len(entries) == 1 and entries[0]['receipt_sha256'] == object_sha(receipt), 'row encoding SHA')
    # Observation is identified by its own phase/step, never by nearest event.
    observations = [e for e in events if e['kind'] == 'fresh_observation' and e.get('episode_id') == eid
                    and e['receipt']['phase'] == 'plan' and e['receipt']['step'] == step]
    require(len(observations) == 1, 'own actual boundary observation')
    observe = observations[0]; obs = observe['receipt']
    require(obs['observed_event_index'] == observe['index'], 'observation receipt/event index')
    original = {k: obs[k] for k in ('path', 'file_sha256', 'pixel_sha256')}
    require(original['path'] == str(output/(eid+f'_plan_{step:02}.png')), 'actual boundary PNG namespace')
    mapping = dict(schema=PRODUCER+'.visual_bridge', episode_id=eid, actual_step=step,
                   original_png=original, legacy_episode_id=ns, legacy_step=5,
                   alias_png=str(output/(ns+'_plan_05.png')))
    require(receipt['actual_step'] == step and receipt['mapping'] == mapping and
            reader.json(output/(ns+'_visual_bridge.json')) == mapping, 'explicit actual/legacy alias mapping')
    png = reader.blob(original['path'], original['file_sha256'], 1024**2)
    require(reader.blob(mapping['alias_png'], original['file_sha256'], 1024**2) == png, 'alias exact original PNG bytes')
    # PNG decoding is saved arithmetic, not render or model execution.
    from PIL import Image
    with Image.open(io.BytesIO(png)) as image:
        pixels = np.asarray(image).copy()
    require(pixels.dtype == np.uint8 and pixels.shape == (224, 224, 3) and
            digest(pixels.tobytes()) == original['pixel_sha256'], 'actual encoding pixel binding')
    request = dict(schema='p1_moving_live_feedback_physical_20261027_v1.encode_request',
                   episode_id=ns, observed_step=5,
                   png=dict(path=mapping['alias_png'], file_sha256=original['file_sha256'],
                            pixel_sha256=original['pixel_sha256']),
                   task_export=layer['task_export'], model_binding_sha256=object_sha(models))
    request['request_sha256'] = object_sha(request)
    require(receipt['request'] == request, 'exact encoding input identity')
    response = receipt['response']; latent = response['latent']
    require(reader.json(output/(ns+'_encode/receipt.json')) == response and entries[0]['latent'] == latent,
            'durable encoder response/row identity')
    require(response['schema'] == 'p1_moving_live_feedback_physical_20261027_v1.encode_response' and
            response['request_sha256'] == request['request_sha256'] and response['episode_id'] == ns and
            response['observed_step'] == 5 and response['device_type'] == 'cuda' and
            response['source_verified'] is response['model_unchanged'] is True and
            response['model_hash_before'] == response['model_hash_after'] == tensor_sha,
            'saved encoder receipt binding')
    require(latent['path'] == str(output/(ns+'_encode/native192.npy')) and
            latent['shape'] == [192] and latent['dtype'] == 'float32', 'own literal encoded array')
    payload = reader.blob(latent['path'], latent['file_sha256'], 1024**2)
    stream = io.BytesIO(payload); z = np.load(stream, allow_pickle=False)
    require(stream.tell() == len(payload) and z.shape == (192,) and z.dtype == np.float32 and
            np.isfinite(z).all() and digest(z.tobytes()) == latent['array_sha256'], 'full encoded output bytes')
    encoded = one(events, 'boundary_encoded', episode_id=eid, actual_step=step)
    require(encoded['encoding_sha256'] == object_sha(receipt) and encoded['mapping_sha256'] == object_sha(mapping)
            and observe['index'] < encoded['index'], 'causal alias/output event binding')
    observation = dict(boundary=step, png_sha256=original['file_sha256'], pixel_sha256=original['pixel_sha256'],
                       encoding_sha256=object_sha(receipt), observed_index=observe['index'],
                       encoded_index=encoded['index'], observed_latent_sha256=packet_sha(z.tolist()))
    return z, observation


def check_episode(row, layer, task, warm10, requests, responses, latents):
    """Independent chronological reconstruction; also used by RAM mutations."""
    executed = array(row['executed_raw'], (row['control_returned'], 2))
    require((abs(executed) <= 1).all(), 'actual raw domain')
    n = len(executed)
    expected = [b for b in (5, 10, 15) if b < n]
    require(sorted(requests) == sorted(responses) == sorted(latents) == expected, 'all reached boundaries, no dropped slot')
    exact(executed[:min(n, 5)], np.asarray(layer['commands'], np.float32)[:min(n, 5)], 'fixed original prefix')
    observed = deepcopy(task['root_context']); imagined = None; audited = []
    for boundary in expected:
        request = requests[boundary]; response = responses[boundary]
        raw10 = np.concatenate((array(warm10, (10, 2)), executed[:boundary]))[-10:]
        exact(request['actualrawhistory10'], raw10, 'warm10 + own executed prefix last10')
        observed = rolled(observed, latents[boundary], raw10, task)
        if imagined is None:
            imagined = rolled(task['root_context'], layer['imagined_prefix'], raw10, task)
        context = observed if row['mode'] == 'R' else imagined
        same_context(request['context3'], context)
        exact(request['goal'], task['goal'], 'same original latent goal')
        exact(request['diagnosticraw5'], np.asarray(layer['commands'], np.float32)[5:], 'fixed original diagnostic only')
        commands = array(response['dispatch_raw5'], (5, 2))
        take = min(5, n-boundary)
        exact(executed[boundary:boundary+take], commands[:take], 'own sealed actions actually returned')
        imagined = deepcopy(response['predicted_successor'])
        # No model prediction here: bind the exact previous saved successor;
        # validate its action history against the same actual full packet.
        if take == 5:
            exact(imagined['action_history'], normalize(executed[boundary-5:boundary+5], task),
                  'saved successor actual action tokens')
        audited.append(dict(boundary=boundary,history_sha256=digest(raw10.tobytes()),
                            context_sha256=object_sha(context),executed_in_packet=take))
    return audited


def consume(output, report_sha, prepared, prepare_sha):
    output = Path(output).absolute(); reader = Reader()
    p = reader.json(prepared, prepare_sha); report = reader.json(output/'report.json', report_sha)
    require(p['schema'] == PRODUCER+'.prepare' and report['schema'] == PRODUCER+'.report', 'producer schemas')
    producer = Path(__file__).with_name(PRODUCER+'.py').resolve()
    require(p['bindings'][str(producer)] == PRODUCER_SHA, 'frozen producer expected closure')
    reader.blob(producer, PRODUCER_SHA)
    require(reader.json(output/'execution_binding.json') ==
            dict(prepared=str(Path(prepared).absolute()), prepare_sha256=prepare_sha), 'execution prepare binding')
    require(report['status'] == 'known_moving_multiboundary_recorded' and not report['errors'] and not report['pending'],
            'terminal clean input only; partial/failure never promoted')
    require(report['fixed_denominator'] == 8 and report['independent_tasks'] == 2 and
            report['P1_admission'] is report['P4_admission'] is False, 'full8 denominator/no admission')
    payload = reader.blob(output/'journal.jsonl', report['journal_sha256'])
    require(payload.endswith(b'\n'), 'terminal complete journal bytes')
    # Only join relevant records. MAIN already independently audited the full
    # chain/native budget; this is not a replacement for that strict consumer.
    events = [decode(line) for line in payload.splitlines()]
    require(events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
            events[-1]['event_sha256'] == report['journal_chain_tail'], 'terminal event identity')
    manifest_path = p['original_manifest']
    manifest = reader.json(manifest_path, p['bindings'][manifest_path])
    pack = reader.json(manifest['sourcepack'], manifest['sourcepack_sha256'])
    expected_ids = [f'e{i:02}_{m}' for i in range(4) for m in ('I', 'R')]
    require([r['episode_id'] for r in report['rows']] == expected_ids, 'fixed eight episode roster')
    require([(s['episode_id'], s['boundary']) for s in report['compile_slots']] ==
            [(e, b) for e in expected_ids for b in (5, 10, 15)], 'fixed24 compile slots')
    results = []
    for index, row in enumerate(report['rows']):
        slot = index//4; arm = ARMS[(index//2) % 2]; mode = ('I', 'R')[index % 2]; eid = expected_ids[index]
        require((row['source_slot'], row['prefix_arm'], row['mode']) == (slot, arm, mode) and
                row['first_error'] is None and not row['secondary_errors'], 'fixed own layer without suppressed failure')
        n = row['control_returned']
        require(type(n) is int and 0 < n <= 20 and row['warmup_returned'] == 10 and row['initial_exact'] is True,
                'completed exact warmup/control count')
        require((row['status'] == 'completed20' and n == 20 and row['full_horizon'] is True) or
                (row['status'] == 'native_stopped_control' and row['full_horizon'] is False and
                 (row['terminated'] is True or row['truncated'] is True)), 'retain native early stop, no padding')
        require(reader.json(output/(eid+'_episode.json')) == row, 'durable episode identity')
        layer = p['layers'][index//2]
        require((layer['source_slot'], layer['arm']) == (slot, arm), 'prepared layer order')
        exported = reader.json(layer['task_export'], layer['task_sha256']); task = exported['task']
        require(exported['source_slot'] == slot and exported['task_id'] == layer['task_id'], 'same source task')
        source_row = pack['rows'][slot]
        require(source_row['slot']['index'] == slot, 'source slot order')
        blind = reader.blob(Path(manifest['sourcepack']).parent/source_row['blind_payload'],
                            source_row['blind_payload_sha256'], 2*1024**2)
        require(exported['blind_sha256'] == source_row['blind_payload_sha256'], 'paired original blind input')
        with np.load(io.BytesIO(blind), allow_pickle=False) as z:
            warm10 = z['raw_action_history'].copy()
        require(warm10.dtype == np.float32 and warm10.shape == (10, 2) and
                digest(warm10.tobytes()) == source_row['raw_history_sha256'], 'sealed real warmup10')
        exact(task['root_raw_action_history'], warm10, 'task/raw source consistency')
        exact(task['action_mean'], p['modelbinding']['action_mean'], 'same original normalizer mean')
        exact(task['action_std'], p['modelbinding']['action_std'], 'same original normalizer std')
        warm_returns = [e for e in events if e['kind'] == 'warmup_return' and e.get('episode_id') == eid]
        require([e['step'] for e in warm_returns] == list(range(1, 11)), 'own exact ten warmup returns')
        exact([e['command'] for e in warm_returns], warm10, 'actual returned warmup matches original sealed source')
        returns = [e for e in events if e['kind'] == 'control_return' and e.get('episode_id') == eid]
        require([e['step'] for e in returns] == list(range(1, n+1)), 'own chronological action returns')
        exact(row['executed_raw'], [e['command'] for e in returns], 'row/actual action return identity')
        steps = [b for b in (5, 10, 15) if b < n]
        encoding_steps = steps+([20] if row['status'] == 'completed20' else [])
        require([e['step'] for e in row['encodings']] == encoding_steps, 'full reached encoding roster')
        requests = {}; responses = {}; latents = {}
        for step in encoding_steps:
            latent, observation = encoding(reader, output, row, step, layer,
                                           exported['task_contract']['model_bindings'],
                                           p['modelbinding']['tensor_sha256'], events)
            if step == 20:
                continue  # Terminal encoding is never a fourth feedback slot.
            ns = eid+f'_b{step:02}'
            request = reader.json(output/(ns+'_compile_request.json'))
            response = reader.json(output/(ns+'_compile_response.json'))
            require((request['schema'], request['episode_id'], request['source_slot'], request['prefix_arm'],
                     request['mode'], request['boundary']) == (PRODUCER+'.compile', eid, slot, arm, mode, step),
                    'own boundary request identity')
            require(request['observation'] == observation and request['modelbinding'] == p['modelbinding'],
                    'actual encoding output -> compiler input binding')
            require(response['schema'] == PRODUCER+'.compiled' and response['request_sha256'] == packet_sha(request)
                    and response['release_sha256'] == packet_sha({k:v for k,v in response.items() if k != 'release_sha256'})
                    and response['predicted_successor'] == response['row']['evaluations'][1]['context']
                    and response['model_hash_before'] == response['model_hash_after'] == p['modelbinding']['tensor_sha256'],
                    'exact saved response/successor release')
            seal = reader.json(output/(ns+'_dispatch_seal.json'))
            require(seal['request_sha256'] == packet_sha(request) and seal['response_sha256'] == packet_sha(response)
                    and seal['seal_sha256'] == object_sha({k:v for k,v in seal.items() if k != 'seal_sha256'})
                    and seal['commands'] == response['dispatch_raw5'], 'sealed request/response/actions')
            compile_slot = one([dict(s,kind='slot') for s in report['compile_slots']], 'slot', episode_id=eid, boundary=step)
            require(compile_slot['status'] == 'sealed' and compile_slot['seal_sha256'] == seal['seal_sha256'], 'retained sealed slot')
            requests[step] = request; responses[step] = response; latents[step] = latent
        for step in (5, 10, 15):
            if step not in steps:
                entry = one([dict(s,kind='slot') for s in report['compile_slots']], 'slot', episode_id=eid, boundary=step)
                require(entry['status'] == 'not_reached' and entry['seal_sha256'] is None, 'early stop retains unreached denominator')
        audited = check_episode(row, layer, task, warm10, requests, responses, latents)
        results.append(dict(episode_id=eid,status=row['status'],control_returned=n,boundaries=audited))
    root = Path(__file__).resolve().parents[2]
    artifacts = [Path(__file__).resolve(), root/'tests'/(NAME+'_tests.py'), root/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.receipt', status='saved_rolling_visual_seams_verified_ONLY',
                consumer_artifact_sha256={str(f):digest(reader.blob(f)) for f in artifacts},
                inputs=reader.inputs, rows=results,
                fixed_denominator=8,compile_denominator=24,independent_tasks=2,
                reached_boundaries=sum(len(r['boundaries']) for r in results),
                full_native_or_budget_reaudit=False,neural_replay=False,physics_execution=False,
                P1_admission=False,P4_admission=False,scientific_pass=False,physical_release=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('output', 'report-sha', 'prepared', 'prepare-sha'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    print(canonical(consume(a.output, a.report_sha, a.prepared, a.prepare_sha)))


if __name__ == '__main__':
    main()
