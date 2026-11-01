"""Independent saved-only constructive-source audit. No producer/env/model imports."""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image

NAME = 'p1_constructive_reachable_saved_audit_20261027_v1'
SOURCE = 'p1_constructive_reachable_source_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
PREPARE01_SHA = '98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086'
PINS = {
    'experiments/spectral_reachability/'+SOURCE+'.py': 'c32b9f169adaed31d07ecef5c507ce409c2ecec5743f8c6fde4fab928b92a878',
    'tests/test_'+SOURCE+'.py': 'e60fbdf60bd6874a5d4e32a2a4e5e0bdc84c50fea06e3f4c38ee65c2372c454f',
    'innerdocs/'+SOURCE+'.md': '7263745f4d0c72a9a25364abd301009daa92d0556de8a0313fc4724bad4f4c1e',
    'experiments/spectral_reachability/p1_conditional_procedural_source_20261027_v1.py': 'faddd18ae8d7e0a97c50348929cb4c860321c88a1aedfb755eedc8a0f717deb3',
    '.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py': 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2',
    '.venv/lib/python3.12/site-packages/stable_worldmodel/envs/utils.py': '2e5d98a5c13227cba6f5574bf72b6eab149ee5b3783df0481b8d240f34047f56',
    '.venv/lib/python3.12/site-packages/stable_worldmodel/spaces.py': '73079b2a66dace087abc7c417691e3cc339887866915553e21d0936a362229eb',
}
PER_SLOT = dict(reset=6, public_step=60, explicit_render=63, goal_render=6,
                set_state=12, physics_step=612, goal_bind=2)
PHASES = ('construction', 'replay0', 'replay1')
LIMITATIONS = [
    'Saved evidence audit, not a new physical execution or model test.',
    'Commit JSON is not hash-anchored in the v1 journal before dispatch; content is checked, durable temporal commitment is source-order-only.',
    'v1 goal_bind records no before/after snapshots or per-step info.goal; no independent saved proof of attachment invariance/goal RGB return.',
    'Reset seeds, original sampled goal and polygon ordering are source-bound declarations; saved audit does not reconstruct RNG/Pymunk.',
    'Global freshness remains unknown_calibration_only; no independent confirmation or strict P1 admission.',
    'No saved timestamps independently establish wall/fsync timing; resource counts are checked, real-time dispatch order follows sealed source plus journal order.',
    'Rejected-slot reasons are retained, not promoted to independently reproduced failures when the failing guard has no saved observation.',
]


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(x): return json.dumps(x, sort_keys=True, allow_nan=False)
def sha(b): return hashlib.sha256(b).hexdigest()
def digest(x): return sha(canonical(x).encode())


def parse(b):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    return json.loads(b, object_pairs_hook=pairs,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def state(x):
    a = np.asarray(x, dtype=np.float64)
    require(a.shape == (7,) and np.isfinite(a).all(), 'finite state7')
    return a


def equal(a, b, label): require(np.array_equal(a, b), label)


def native(goal, current):
    g, s = state(goal), state(current)
    d = abs(g[4]-s[4]); old_angle = min(d, 2*np.pi-d)
    wrapped = abs(np.arctan2(np.sin(g[4]-s[4]), np.cos(g[4]-s[4])))
    xy = float(np.linalg.norm(g[:4]-s[:4]))
    return dict(success=bool(xy < 20 and old_angle < np.pi/9),
                wrapped_success=bool(xy < 20 and wrapped < np.pi/9),
                angle_bug=bool(old_angle < 0), xy=xy, wrapped_angle=float(wrapped),
                reward=-float(np.linalg.norm(g-s)))


def witness(root, sign):
    s = state(root); delta = s[2:4]-s[:2]; length = np.linalg.norm(delta)
    require(length > 0 and sign in (-1, 1), 'nondegenerate witness')
    u = delta/length; v = np.array([-u[1], u[0]])
    logits = np.vstack([np.repeat((.5*u)[None], 5, axis=0),
                        np.repeat((.25*u+sign*.5*v)[None], 5, axis=0)]).astype(np.float32)
    require(np.all(np.linalg.norm(logits.reshape(2, 10).astype(float), axis=1) <= .75*np.sqrt(5)), 'packet balls')
    return logits, np.tanh(logits)


def decode_prepare(b, expected_sha, hash_path=file_sha):
    require(sha(b) == expected_sha and len(b) < 200_000, 'external prepare SHA/size')
    e = parse(b)
    require(set(e) == {'schema', 'plan_sha256', 'gzip_base64'} and e['schema'] == SOURCE+'.envelope', 'prepare envelope')
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(e['gzip_base64'], validate=True))) as f:
        decoded = f.read(2*1024**2+1)
    require(len(decoded) <= 2*1024**2, 'prepare decompression cap')
    p = parse(decoded); require(digest(p) == e['plan_sha256'], 'prepare semantic SHA')
    require(p['schema'] == SOURCE+'.prepare', 'prepare schema')
    slots = p['slots']; n = len(slots)
    require(n in (2, 3, 4), 'fixed denominator2--4')
    seeds = []
    for i, s in enumerate(slots):
        require(set(s) == {'index', 'root_seed', 'reset_seed', 'sign'} and s['index'] == i and
                s['sign'] == (1 if i % 2 == 0 else -1), 'fixed slot/sign order')
        seeds += [s['root_seed'], s['reset_seed']]
    require(len(set(seeds)) == 2*n and all(type(s) is int and 0 < s < 2**31 for s in seeds), 'explicit distinct seeds')
    require(p['caps'] == {k: v*n for k, v in PER_SLOT.items()}, 'independent resource caps')
    require(p['horizon'] == p['warmup'] == 10 and p['replay_episodes'] == 2 and p['with_target'] is False and
            p['environment'] == 'swm/PushT-v1' and p['packet_logit_cap'] == float(.75*np.sqrt(5)) and
            p['output_cap_bytes'] == 8*1024**2 and p['disk_floor_bytes'] == 200*1024**2 and
            type(p['wall_seconds']) is int and 0 < p['wall_seconds'] <= 600, 'fixed source contract')
    require(p['recipe'] == dict(position_jitter=16., angle_offset_range=[-np.pi, np.pi],
            edge_fraction_range=[.15, .85], pusher_surface_offset=15.), 'fixed source recipe')
    text_contract = dict(
        witness='5 repeated radial logit=.5 then5 logit=.25radial+sign*.5tangent; fixed sign by slot; world geometry only in offline builder',
        selection='one fixed witness per slot; no replacement; failed slots retained; native success is reach label',
        goal_attachment='after original10-step observed history, native _set_goal_state only + _goal=actual endpoint RGB; no state setter/physics',
        deployment_payload='only own replay RGB_history/raw_action_history/goal_RGB; certificate actions/states forbidden')
    require(all(p[k] == v for k,v in text_contract.items()), 'frozen policy contract')
    require(p['P1_admission'] is False and p['training'] is False and p['new_neural_calls'] == 0 and
            p['global_freshness'] == 'unknown_calibration_only', 'no admission/learning claim')
    expected = {str(PROJECT/k): v for k, v in PINS.items()}
    require(p['bindings'] == expected, 'exact frozen source closure')
    exclusions = p['exclusions']
    require(exclusions['physical_future_labels_read'] is False and exclusions['coverage']['status'] == 'unknown', 'metadata-only unknown freshness')
    for seed in seeds:
        require(all(seed != row['seed'] for row in exclusions['seed_families']) and
                all(not row['span'][0] <= seed < row['span'][1] for row in exclusions['reset_spans']), 'historical seed overlap')
    for path, h in {**exclusions['files'], **expected}.items():
        if path == 'metadata_authority_content_id':
            require(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h), 'metadata semantic ID'); continue
        target = Path(path); target = target if target.is_absolute() else PROJECT/target
        require(target.resolve().is_relative_to(PROJECT), 'binding outside project')
        require(hash_path(target) == h, 'bound file SHA: '+str(target))
    return p


def arrays(b, fields):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        require(len(z.infolist()) == len(fields) and sum(x.file_size for x in z.infolist()) <= 4*1024**2, 'bounded unique NPZ entries')
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        require(set(z.files) == set(fields), 'NPZ field whitelist/privilege firewall')
        return {k: z[k].copy() for k in z.files}


def audit_saved(p, pack_bytes, expected_sha, read):
    """read(name)->bytes is a flat read-only artifact provider; mockable in memory."""
    require(sha(pack_bytes) == expected_sha, 'external sourcepack SHA')
    r = parse(pack_bytes); n = len(p['slots'])
    require(r['schema'] == SOURCE+'.sourcepack' and r['prepare_id'] == digest(p), 'sourcepack/prepare identity')
    require(len(r['rows']) == n and [x['slot'] for x in r['rows']] == p['slots'], 'all fixed slots, never survivor-only')
    require(r['P1_admission'] is False and r['new_neural_calls'] == 0 and r['training'] is False and
            r['new_independent_confirmation'] is False and r['direct_difficulty_required'] is False and
            r['global_freshness'] == 'unknown_calibration_only', 'science boundary')
    lines = read('journal.jsonl').splitlines(); events = []; previous = None
    for i, line in enumerate(lines):
        event = parse(line)
        require(event['index'] == i and event['previous_sha256'] == previous and
                event['event_sha256'] == digest({k:v for k,v in event.items() if k != 'event_sha256'}), 'journal hashchain')
        previous = event['event_sha256']; events.append(event)
    require(events and events[0]['kind'] == 'run_started' and events[0]['prepare'] == p and
            previous == r['journal_chain_tail'], 'journal prepare/tail anchor')
    require(events[-1]['kind'] == 'run_finished' and events[-1]['status'] == r['status'] and
            events[-1]['errors'] == r['errors'], 'terminal journal receipt')
    counts = {k: dict(attempted=0, completed=0) for k in PER_SLOT}
    stack = []; phases = {}; current = None; receipts = {}; certificates = {}; last_order = -1
    previous_event = events[0]
    for e in events[1:-1]:
        kind = e['kind']
        if kind in ('operation_attempted', 'operation_completed'):
            resource = e['resource']; require(resource in counts, 'unknown API resource')
            payload = {k:v for k,v in e.items() if k not in ('index','previous_sha256','event_sha256','kind')}
            if kind == 'operation_attempted':
                if resource == 'reset':
                    require(not stack, 'nested reset')
                    if e['kind_of_reset'] == 'seed':
                        key = (e['slot'], e['phase']); order = key[0]*3+PHASES.index(key[1])
                        require(0 <= key[0] < n and order > last_order, 'no retries/reordered candidate phases')
                        last_order = order; current = key
                        phases[key] = dict(frames={}, public=[], returns=[], recipes=[], commits=None, binds=0, resets=[])
                    require(current == (e['slot'],e['phase']), 'reset phase identity')
                    phases[current]['resets'].append(e['kind_of_reset'])
                require(current is not None, 'operation without slot')
                allowed = {'reset': ('set_state','goal_render'), 'set_state': ('physics_step',), 'public_step': ('physics_step',)}
                if stack: require(resource in allowed.get(stack[-1][0]['resource'], ()), 'unexpected nested API')
                else: require(resource in ('reset','public_step','explicit_render','goal_bind'), 'unparented internal API')
                if resource == 'public_step':
                    q = phases[current]; step = len(q['public'])+1
                    require(e['phase'] == current[1] and e['step'] == step and step <= 20, 'contiguous own action history')
                    action = np.asarray(e['raw_action']); require(action.shape == (2,) and np.isfinite(action).all(), 'finite public action')
                    if step <= 10: equal(action, [0.,0.], 'warmup action')
                    q['public'].append(e)
                if resource == 'goal_bind':
                    require(current[1] != 'construction' and len(phases[current]['public']) == 10 and e['phase'] == current[1], 'goal only at replay boundary')
                    phases[current]['binds'] += 1
                counts[resource]['attempted'] += 1
                stack.append((payload, []))
            else:
                require(stack and payload == stack[-1][0], 'paired nested API completion')
                _, children = stack.pop()
                grammar = {'reset':['set_state','goal_render','set_state'], 'set_state':['physics_step'], 'public_step':['physics_step']*10}
                require(children == grammar.get(resource, []), 'completed internal API census')
                if stack: stack[-1][1].append(resource)
                counts[resource]['completed'] += 1
        elif kind == 'placement_recipe':
            key = (e['slot']['index'], e['phase']); require(key == current and not stack and e['slot'] == p['slots'][key[0]], 'placement recipe scope')
            state(e['original_goal']); state(e['requested']); phases[key]['recipes'].append(e)
        elif kind == 'fresh_observation':
            key = (e['slot'],e['phase']); require(key == current and not stack, 'fresh observation scope')
            q = phases[key]; step = e['step']
            require(previous_event['kind'] == 'operation_completed' and previous_event['resource'] == 'explicit_render', 'fresh frame follows actual render return')
            require(step == len(q['public']) and step not in q['frames'], 'fresh observation step')
            require(e['png'] == f's{key[0]}_{key[1]}_{step:02}.png', 'own PNG name')
            b = read(e['png']); require(sha(b) == e['file_sha256'], 'PNG file SHA')
            with Image.open(io.BytesIO(b)) as im:
                require(im.mode == 'RGB' and im.size == (224,224), 'native RGB dimensions'); rgb = np.array(im)
            require(sha(np.ascontiguousarray(rgb).tobytes()) == e['pixel_sha256'], 'PNG pixel SHA')
            s = state(e['state']); equal(s, state(e['snapshot']['state']), 'snapshot observation')
            pixels = int(np.count_nonzero(np.all(rgb == [65,105,225], axis=-1)))
            require(pixels == e['canonical_agent_pixels'] and pixels > 0 and ((s[:4] >= 0)&(s[:4] <= 512)).all(), 'fresh RGB/canvas guard')
            require(np.isfinite(e['snapshot']['block_velocity']).all() and len(e['snapshot']['block_velocity']) == 2 and
                    np.isfinite(e['snapshot']['block_angular_velocity']) and np.isfinite(e['snapshot']['block_angle']), 'finite hidden diagnostics')
            equal(s[4], e['snapshot']['block_angle'] % (2*np.pi), 'observed native wrapped angle')
            q['frames'][step] = dict(event=e, rgb=rgb)
        elif kind == 'native_return':
            key = (e['slot'],e['phase']); require(key == current and not stack, 'native return scope')
            q = phases[key]; require(e['step'] == len(q['returns'])+1 and e['step']+10 in q['frames'], 'native return order')
            require(previous_event['kind'] == 'fresh_observation' and previous_event['step'] == e['step']+10, 'native return follows own fresh frame')
            equal(state(e['state']), q['frames'][e['step']+10]['event']['state'], 'native state vs fresh frame')
            q['returns'].append(e)
        elif kind == 'construction_certificate_saved':
            require(current[1] == 'construction' and not stack and current[0] not in certificates, 'certificate scope')
            require(e['file'] == f's{current[0]}_construction_certificate.npz', 'certificate filename')
            b = read(e['file']); require(sha(b) == e['sha256'], 'certificate SHA')
            certificates[current[0]] = (e, arrays(b, ('goal','goal_rgb','witness_commands','witness_logits','future_states','observed_states','root_history','raw_action_history')))
        elif kind == 'slot_finished':
            row = e['receipt']; i = row['slot']['index']
            require(not stack and i not in receipts and row == r['rows'][i], 'fixed slot receipt identity')
            receipts[i] = row
        else: raise ValueError('unknown journal event: '+kind)
        previous_event = e
    require(counts == r['counts'] == events[-1]['counts'], 'full attempted/completed API ledger')
    require(all(0 <= x['completed'] <= x['attempted'] <= p['caps'][k] for k,x in counts.items()), 'resource cap')
    require(r['first_error'] == (r['errors'][0] if r['errors'] else None), 'first error preserved')
    clean = r['status'] == 'fixed_slots_processed' and not r['errors']
    require(r['sourcepack_complete'] is clean, 'complete flag is not survivor success')
    if clean: require(not stack and len(receipts) == n, 'complete fixed denominator and operations')
    else: require(r['status'] == 'engineering_failed' and r['errors'], 'partial requires explicit engineering error')
    rows = []; roots = set(); verified = 0; checked_commits = 0
    for i, row in enumerate(r['rows']):
        require(type(row['reachable']) is bool, 'boolean reach label')
        present = [ph for ph in PHASES if (i,ph) in phases]
        require(present == list(PHASES[:len(present)]), 'no skipped phase')
        for ph in present:
            q = phases[i,ph]
            require(q['resets'] in (['seed'], ['seed','placement']) and len(q['recipes']) <= 1, 'one fixed recipe no retry')
            if len(q['resets']) == 2: require(len(q['recipes']) == 1, 'placement metadata required')
            if ph != 'construction' and q['recipes']:
                ref = phases[i,'construction']['recipes'][0]
                equal(q['recipes'][0]['original_goal'],ref['original_goal'],'replay original goal')
                equal(q['recipes'][0]['requested'],ref['requested'],'replay requested recipe')
            frames = q['frames']
            if 10 in frames:
                root = frames[10]['event']['state']; x,a = witness(root,p['slots'][i]['sign'])
                # Rejection immediately after final warmup may precede commit.
                name = f's{i}_{ph}_commit.json'
                try: c = parse(read(name))
                except FileNotFoundError:
                    require(len(q['public']) <= 10 and row['reachable'] is False, 'future without saved commitment'); c = None
                if c is not None:
                    checked_commits += 1
                    require(set(c) == {'logits','commands','root_snapshot'}, 'commit whitelist')
                    equal(c['logits'],x.astype(float),'independent witness logits'); equal(c['commands'],a.astype(float),'independent tanh commands')
                    require(c['root_snapshot'] == frames[10]['event']['snapshot'], 'commit own boundary snapshot')
                for action in q['public'][10:]: equal(action['raw_action'],a[action['step']-11].astype(float),'executed fixed witness')
            original = q['recipes'][0]['original_goal'] if q['recipes'] else None
            for t,f in frames.items():
                if t <= 10 and original is not None and t+1 <= len(q['public']):
                    require(not native(original,f['event']['state'])['success'], 'no warmup continuation after native success')
            if ph != 'construction':
                base = phases[i,'construction']['frames']
                for t,f in frames.items():
                    require(t in base, 'replay future absent from construction')
                    equal(f['event']['state'],base[t]['event']['state'],'replay own state prefix')
                    equal(f['rgb'],base[t]['rgb'],'replay own RGB prefix')
                    if t <= 10: require(f['event']['snapshot'] == base[t]['event']['snapshot'], 'replay own dynamic history')
        if i in certificates:
            event,z = certificates[i]; q = phases[i,'construction']; frames = q['frames']
            require(set(frames) == set(range(21)) and len(q['returns']) == 10, 'full construction endpoint')
            require(not any(e['terminated'] or e['truncated'] for e in q['returns']), 'construction must not continue past terminal')
            root = state(frames[10]['event']['state']); goal = state(frames[20]['event']['state'])
            x,a = witness(root,p['slots'][i]['sign'])
            for k,v in dict(goal=goal, goal_rgb=frames[20]['rgb'], witness_logits=x, witness_commands=a,
                future_states=np.asarray([frames[t]['event']['state'] for t in range(11,21)]),
                observed_states=np.asarray([frames[t]['event']['state'] for t in (0,5,10)]),
                root_history=np.stack([frames[t]['rgb'] for t in (0,5,10)]), raw_action_history=np.zeros((10,2),np.float32)).items():
                equal(z[k],v,'certificate array: '+k)
            require(z['witness_logits'].dtype == z['witness_commands'].dtype == z['raw_action_history'].dtype == np.float32 and
                    z['root_history'].dtype == z['goal_rgb'].dtype == np.uint8, 'native certificate dtypes')
        diagnostics = []
        for ph in present:
            q = phases[i,ph]
            goal = q['recipes'][0]['original_goal'] if ph == 'construction' and q['recipes'] else (certificates[i][1]['goal'] if i in certificates else None)
            if q['returns']: require(goal is not None, 'native returns require saved goal')
            for j,e in enumerate(q['returns']):
                d = native(goal,e['state']); require(type(e['terminated']) is bool and type(e['truncated']) is bool and
                    e['terminated'] == d['success'] and e['reward'] == d['reward'], 'independent native reward/success')
                if j+1 < len(q['returns']): require(not e['terminated'] and not e['truncated'], 'no continuation after terminal')
                diagnostics.append(dict(phase=ph, step=e['step'], **d))
        if row['reachable']:
            require(row['status'] == 'reachable_native_replayed' and present == list(PHASES) and i in certificates, 'reachable evidence required')
            event,z = certificates[i]; base = phases[i,'construction']['frames']; root = state(base[10]['event']['state']); goal = state(z['goal'])
            require(not native(goal,root)['success'], 'nontrivial derived task')
            lengths = []
            for ph in PHASES[1:]:
                q = phases[i,ph]; length = len(q['returns']); lengths.append(length)
                require(q['binds'] == 1 and 1 <= length <= 10 and set(q['frames']) == set(range(11+length)) and
                        len(q['public']) == 10+length and q['returns'][-1]['terminated'] and not q['returns'][-1]['truncated'], 'two complete native replay successes')
            require(row['replay_native_success'] == [True,True] and row['executed_replay_steps'] == lengths, 'replay label census')
            root_hash = sha(np.ascontiguousarray(np.concatenate([root,goal]),dtype=np.float64).tobytes())
            require(row['root_sha256'] == root_hash and root_hash not in roots and root_hash not in p['exclusions']['root_hashes'], 'fresh distinct root identity')
            roots.add(root_hash)
            require(row['certificate'] == event['file'] and row['certificate_sha256'] == event['sha256'], 'row certificate binding')
            require(row['blind_payload'] == f's{i}_blind_inputs.npz', 'own blind payload')
            b = read(row['blind_payload']); require(sha(b) == row['blind_payload_sha256'], 'blind payload SHA')
            blind = arrays(b, ('rgb_history','raw_action_history','goal_rgb'))
            for k,v in dict(rgb_history=np.stack([phases[i,'replay0']['frames'][t]['rgb'] for t in (0,5,10)]),
                            raw_action_history=np.zeros((10,2),np.float32), goal_rgb=z['goal_rgb']).items():
                equal(blind[k],v,'blind own causal array: '+k)
                require(blind[k].dtype == v.dtype, 'blind array dtype')
            require(row['object_displacement'] == float(np.linalg.norm(goal[2:4]-root[2:4])) and
                    row['wrapped_object_rotation'] == native(goal,root)['wrapped_angle'], 'descriptive object diagnostics')
            verified += 1
        else:
            require(row['status'] in ('rejected_fixed_candidate','engineering_failed','not_started'), 'explicit partial/rejected status')
            require(not any(k in row for k in ('blind_payload','root_sha256')), 'no rejected slot admission')
            if row['status'] == 'rejected_fixed_candidate': require(i in receipts and present and row.get('reason'), 'durable rejection reason/attempt')
            if row['status'] == 'not_started': require(not present and i not in receipts and not clean, 'unstarted remains denominator')
            if row['status'] == 'engineering_failed': require(not clean and r['errors'], 'engineering error evidence')
        summary = dict(index=i,status=row['status'],reachable_saved_evidence=bool(row['reachable']),
                       rejection_reason_independently_proven=False if row['status'] == 'rejected_fixed_candidate' else None,
                       phases=present,native_diagnostics=diagnostics)
        if row['reachable']:
            summary.update(root_sha256=root_hash,object_displacement=row['object_displacement'],
                           wrapped_object_rotation=row['wrapped_object_rotation'],executed_replay_steps=lengths,
                           original_root_vs_derived_goal=native(goal,root))
        rows.append(summary)
    require(r['reachable_count'] == verified, 'reachable numerator vs fixed denominator')
    return dict(status='saved_evidence_consistent' if clean else 'saved_partial_consistent_not_admitted',
                fixed_denominator=n, reachable_saved_evidence=verified, rows=rows, counts=counts,
                journal_events=len(events), png_frames=sum(len(q['frames']) for q in phases.values()),
                certificates_checked=len(certificates), witness_commits_checked=checked_commits,
                partial_errors=r['errors'], pending_operations=[x[0] for x in stack],
                sourcepack_complete=clean, P1_admission=False, independent_confirmation=False,
                direct_difficulty_verified=False, two_stage_advantage_verified=False,
                non_greedy_capability_change_verified=False, frozen_model_blind_test_completed=False,
                durable_precommit_independently_proven=False, attachment_invariance_independently_proven=False,
                limitations=LIMITATIONS)


def audit_files(prepared, prepare_sha, sourcepack=None, sourcepack_sha=None):
    report = dict(schema=NAME+'.audit', status='unverified', P1_admission=False, errors=[],
                  prepared_path=str(Path(prepared).resolve()), sourcepack_path=str(Path(sourcepack).resolve()) if sourcepack else None,
                  prepare_sha256=prepare_sha, sourcepack_sha256=sourcepack_sha, limitations=LIMITATIONS)
    try:
        p = decode_prepare(Path(prepared).read_bytes(), prepare_sha)
        report['fixed_denominator'] = len(p['slots'])
        if sourcepack is None or not Path(sourcepack).exists():
            report['status'] = 'actual_sourcepack_missing'; return report
        require(sourcepack_sha is not None, 'external sourcepack SHA required')
        path = Path(sourcepack); directory = path.parent.resolve()
        require(not path.is_symlink() and path.stat().st_size <= 128*1024, 'small regular sourcepack')
        used = sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
        require(used <= 8*1024**2, 'original output byte cap')
        def read(name):
            target = directory/name
            require(Path(name).name == name and not target.is_symlink() and target.resolve().parent == directory, 'flat nonsymlink artifact')
            require(target.stat().st_size <= 8*1024**2, 'bounded artifact')
            return target.read_bytes()
        binding = parse(read('execution_binding.json'))
        require(binding == dict(prepared=str(Path(prepared).resolve()),prepare_file_sha256=prepare_sha), 'external execution/prepare binding')
        report.update(audit_saved(p,path.read_bytes(),sourcepack_sha,read))
        report['journal_file_sha256'] = sha(read('journal.jsonl'))
        report['execution_binding_sha256'] = sha(read('execution_binding.json'))
    except Exception as exc:
        report['status'] = 'unverified_or_inconsistent'
        report['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--prepare-sha',required=True)
    parser.add_argument('--sourcepack',type=Path)
    parser.add_argument('--sourcepack-sha')
    parser.add_argument('--output',type=Path)
    a = parser.parse_args(); r = audit_files(a.prepared,a.prepare_sha,a.sourcepack,a.sourcepack_sha)
    r['audit_source_bindings'] = {str(path):file_sha(path) for path in (
        Path(__file__).resolve(), PROJECT/'tests'/('test_'+NAME+'.py'), PROJECT/'innerdocs'/(NAME+'.md'))}
    value = (canonical(r)+'\n').encode()
    if a.output:
        require(len(value) <= 200_000 and shutil.disk_usage(a.output.parent).free >= 200*1024**2+len(value), 'small NEW report/reserve')
        with a.output.open('xb') as f: f.write(value)
    print(value.decode(),end='')
    if r['status'] in ('unverified_or_inconsistent','actual_sourcepack_missing'): raise SystemExit(2)


if __name__ == '__main__': main()
