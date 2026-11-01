"""Matched-order feedback seam. Injected callbacks only; NOT a physical runner.

The caller must authenticate environment, RGB encoder, model, source and durable
events. This seam deliberately cannot issue a production or scientific receipt.
"""
from copy import deepcopy
import hashlib
import numpy as np
from experiments.spectral_reachability import p3_matched_order_codex_20261029 as design

core = design.core
NAME = 'p3_matched_order_feedback_seam_codex_20261029'
ROSTER = [(0, 0, 'fixed_order'), (1, 0, 'signal_order'),
          (2, 1, 'fixed_order'), (3, 1, 'signal_order')]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def f32(value, shape, box=False):
    a = np.asarray(value)
    need(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), 'literal f32 shape')
    if box:
        need((np.abs(a) <= 1).all(), 'raw action domain')
    return a


def join(context, previous_raw, submitted, returned, latent, mean, std):
    """Shift at *each* real 5-step boundary, including inside H3."""
    z = f32(np.asarray(context['latent_history'], dtype=np.float32), (3, 192))
    history = f32(np.asarray(context['action_history'], dtype=np.float32), (2, 10))
    old = f32(previous_raw, (5, 2), True)
    sent = f32(submitted, (5, 2), True)
    got = f32(returned, (5, 2), True)
    mu, scale = f32(mean, (2,)), f32(std, (2,))
    new_z = f32(latent, (192,))
    need((scale > 0).all() and sent.tobytes() == got.tobytes(), 'normalizer or physical ACK')
    old_token = np.divide(np.subtract(old, mu, dtype=np.float32), scale, dtype=np.float32).reshape(10)
    need(old_token.tobytes() == history[-1].tobytes(), 'previous actual action history')
    token = np.divide(np.subtract(got, mu, dtype=np.float32), scale, dtype=np.float32).reshape(10)
    out = deepcopy(context)
    out['latent_history'] = np.concatenate((z[1:], new_z[None])).tolist()
    out['action_history'] = np.stack((history[-1], token)).tolist()
    return core.context(out)


def compile_block(root, context, policy, remaining, stage, factory, store, ledger, permit_ram=False):
    """Original matched-order candidate generation, rebound to actual context."""
    task = design.parent.parent_task(root, context)
    backend = factory(task, store)
    need(getattr(backend, 'kind', None) == 'HOST_CUDA' or
         (permit_ram and getattr(backend, 'kind', None) == 'RAM_FIXTURE'), 'backend kind')
    responses = []
    for h in (1, 3):
        ledger.scope = dict(stage=stage, phase='signal', horizon=h)
        req = core.request(task, dict(context=context), operation='compile_finite_horizon_J', horizon=h)
        r = ledger.call('signal_compile', lambda req=req: backend.signal(req, ledger))
        need(r['binding_sha256'] == req['binding_sha256'] and
             r['parent_context_sha256'] == req['parent_context_sha256'] and r['horizon'] == h, 'signal binding')
        responses.append(r)
    ledger.scope = dict(stage=stage, phase='diagnostic')
    diagnostic = design.signal.spectra(task, responses, svd=lambda a: ledger.call('array_svd', lambda: np.linalg.svd(a, full_matrices=False)))
    node, signals, _ = design.signal.runtime_signal(task, responses, diagnostic)
    ledger.scope = dict(stage=stage, phase='schedule')
    decision = ledger.call('schedule', lambda: core.choose(task, node, signals, 'adaptive'))
    h = design.choose(policy, decision, remaining, stage)
    ledger.scope = dict(stage=stage, phase='program', horizon=h)
    bank = ledger.call('program_compile', lambda: design.frozen.generate(task, responses[0 if h == 1 else 1], 'endpoints', ledger))
    candidates = []
    for index, actions in enumerate(bank['actions']):
        ledger.scope = dict(stage=stage, phase='candidate', horizon=h, index=index)
        whole = ledger.call('replay', lambda: backend.replay(context, actions, False, ledger))
        segmented = ledger.call('replay', lambda: backend.replay(context, actions, True, ledger))
        need(whole == segmented, 'whole/segmented mismatch')
        terminal = core.checked_path(task, context, actions, whole)
        cost = ledger.call('metric', lambda: core.model_cost(task, terminal))
        candidates.append(dict(index=index, actions=actions, whole=whole, segmented=segmented, cost=cost))
    need(len(candidates) == 4, 'four candidates')
    winner = min(candidates, key=lambda x: (x['cost'], x['index']))
    ledger.scope = dict(stage=stage, phase='winner', horizon=h, index=winner['index'])
    path = ledger.call('replay', lambda: backend.replay(context, winner['actions'], False, ledger))
    need(path == winner['whole'], 'winner replay mismatch')
    core.checked_path(task, context, winner['actions'], path)
    return dict(stage=stage, context_sha256=core.digest(context), responses=responses,
                diagnostic=diagnostic, decision=decision, horizon=h, bank=bank,
                candidates=candidates, winner_index=winner['index'], actions=winner['actions'])


def execute_slot(root, policy, prior_raw, session, factory, store, *, permit_ram=False):
    """Inject session.packet and session.encode; preserves incomplete native stops.

    packet(start, raw) returns raw, RGB bytes, end, stopped, episode_id.
    encode(end, RGB bytes) returns latent, input_sha256, end, episode_id.
    IDs/hashes are structural joins only, never proof of a real device or env.
    """
    root = core.validate_task(root)
    need(policy in ('fixed_order', 'signal_order'), 'fixed policy')
    context = deepcopy(root['root_context'])
    previous = f32(prior_raw, (5, 2), True).copy()
    mean = np.asarray(root['action_mean'], dtype=np.float32)
    std = np.asarray(root['action_std'], dtype=np.float32)
    episode_id = session.episode_id
    need(type(episode_id) is str and bool(episode_id), 'episode identity')
    remaining = [1, 3]
    ledger = core.Ledger(design.expected(), sink=store.event)
    result = dict(schema=NAME+'.slot', policy=policy, root_sha256=core.digest(root),
                  status='attempted', blocks=[], packets=[], errors=[], source_authenticated=False,
                  native_execution_certified=False, scientific_pass=False)
    try:
        for stage in range(2):
            block = compile_block(root, context, policy, remaining, stage, factory, store, ledger, permit_ram)
            remaining.remove(block['horizon'])
            result['blocks'].append(block)
            store.event(dict(kind='block_compiled', stage=stage, context_sha256=block['context_sha256'],
                             horizon=block['horizon'], actions=deepcopy(block['actions'])))
            commands = f32(np.asarray(block['actions'], dtype=np.float32), (5*block['horizon'], 2), True)
            for offset in range(0, len(commands), 5):
                start = len(result['packets'])*5
                sent = commands[offset:offset+5].copy()
                store.event(dict(kind='packet_intent', stage=stage, start=start, raw=sent.tolist()))
                ret = session.packet(start, sent.copy())
                need(type(ret) is dict and set(ret) == {'raw','rgb_bytes','end','stopped','episode_id'}, 'packet ABI')
                got = np.asarray(ret['raw'])
                need(got.ndim == 2 and got.shape[1:] == (2,) and 1 <= len(got) <= 5, 'returned packet length')
                got = f32(got, (len(got), 2), True)
                need(ret['episode_id'] == episode_id and ret['end'] == start+len(got) and
                     type(ret['stopped']) is bool and got.tobytes() == sent[:len(got)].tobytes() and
                     (len(got) == 5 or ret['stopped']), 'packet exact same episode ACK')
                record = dict(stage=stage, start=start, end=ret['end'], raw=got.tolist(), stopped=ret['stopped'])
                result['packets'].append(record)
                store.event(dict(kind='packet_return', **record))
                if ret['stopped']:
                    result['status'] = 'native_stopped'; return result
                rgb = ret['rgb_bytes']
                need(type(rgb) is bytes and len(rgb) > 0, 'actual RGB bytes')
                rgb_sha = hashlib.sha256(rgb).hexdigest()
                enc = session.encode(ret['end'], rgb)
                need(type(enc) is dict and set(enc) == {'latent','input_sha256','end','episode_id'} and
                     enc['episode_id'] == episode_id and enc['end'] == ret['end'] and
                     enc['input_sha256'] == rgb_sha, 'same actual RGB encoder binding')
                context = join(context, previous, sent, got, enc['latent'], mean, std)
                previous = got.copy()
                store.event(dict(kind='actual_context', stage=stage, end=ret['end'], rgb_sha256=rgb_sha,
                                 context_sha256=core.digest(context)))
        need(not remaining and ledger.attempted == ledger.completed == design.expected(), 'complete measured axes')
        result['status'] = 'callback_feedback_complete_not_certified'
    except Exception as exc:
        result['status'] = 'failed'; result['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        result['resource_receipt'] = ledger.receipt()
        store.event(dict(kind='slot_final', result=deepcopy(result)))
    return result


def execute_roster(roots, prior_raws, session_factory, backend_factory, store, *, permit_ram=False):
    """Fixed four identities, independent failure retention; callback-only seam."""
    need(len(roots) == len(prior_raws) == 2, 'two fixed root inputs')
    rows = [dict(slot=s, root=r, policy=p, status='not_started', result=None) for s,r,p in ROSTER]
    store.event(dict(kind='fixed_denominator', rows=deepcopy(rows)))
    for row in rows:
        store.event(dict(kind='slot_attempt', slot=row['slot'], root=row['root'], policy=row['policy']))
        try:
            session = session_factory(row['slot'], row['root'], row['policy'])
            result = execute_slot(roots[row['root']], row['policy'], prior_raws[row['root']], session,
                                  backend_factory, store, permit_ram=permit_ram)
            row.update(status=result['status'], result=result)
        except Exception as exc:
            row.update(status='setup_failed', error=type(exc).__name__+': '+str(exc))
        store.event(dict(kind='roster_slot_return', row=deepcopy(row)))
    return dict(schema=NAME+'.roster', rows=rows, fixed_denominator=4,
                physical_certified=False, scientific_pass=False)
