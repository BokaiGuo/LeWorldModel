"""Saved-event reconstruction of K17 guard and native RGB frames; no dispatch."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_physical_gate_20261029_v1 as gate
from experiments.spectral_reachability import agent_p3_c3_k17_replay_guard_bridge_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p3_c3_k17_live_guard_20261029_v1 as live_guard
from experiments.spectral_reachability import agent_p3_c3_k17_candidate_join_20261029_v1 as join
from experiments.spectral_reachability import p3_k17_saved_source_budget_consumer_20261029 as basic


def need(ok, message):
    if not ok:
        raise ValueError(message)


def source_files():
    h3 = outer.h3
    modules = (outer, h3, h3.native, h3.native.allmode, h3.native.allmode.base,
               h3.native.allmode.base.paired, h3.moving, h3.source, gate, bridge, live_guard, join)
    result = {}
    for module in modules:
        path = Path(module.__file__)
        need(path.is_file() and not path.is_symlink(), 'source file unavailable')
        name = path.name
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        need(name not in result or result[name] == value, 'ambiguous source basename')
        result[name] = value
    return result


def audit(directory, journal_sha256, source_sha256, material_sha256):
    """Reconstruct saved guard inputs from pinned source and durable events.

    Hash anchors do not prove that a native process generated the events. Partial
    branches remain partial; no environment, scientific or release authorization.
    """
    need(isinstance(source_sha256, dict) and source_files() == source_sha256, 'guard source pins')
    data, sources = outer.material()
    material = lambda: hashlib.sha256(basic.transport.canonical(join.snapshot(data, sources))).hexdigest()
    need(material() == material_sha256, 'guard material source pin')
    events = basic.read_events(Path(directory)/'journal.jsonl', journal_sha256)
    branch = -1
    slots = []
    current = None
    for event in events:
        kind, p = event['kind'], event['payload']
        if kind == 'operation_dispatch_intent' and p.get('receipt', {}).get('resource') == 'constructor':
            branch += 1
            need(branch < 6, 'extra branch')
            current = {'index': branch, 'frames': [], 'warm': [], 'transitions': {}, 'placement': [], 'goal': [], 'guard': None}
            slots.append(current)
        if current is None:
            continue
        if kind in ('placement_recipe', 'goal_attached', 'returned_transition', 'fresh_observation'):
            need((p.get('branch'), p.get('slot'), p.get('root_slot'), p.get('mode')) == (branch, *outer.ORDER[branch]), 'event slot binding')
        if kind == 'placement_recipe':
            current['placement'].append(p)
        elif kind == 'goal_attached':
            current['goal'].append(p)
        elif kind == 'returned_transition':
            key = (p.get('phase'), p.get('step'))
            need(key not in current['transitions'], 'duplicate transition')
            phase, step = key
            need(phase in ('warmup', 'plan') and type(step) is int and step >= 1 and
                 len(current['frames']) == (3*step if phase == 'warmup' else 36+3*(step-1)),
                 'transition/frame chronology')
            current['transitions'][key] = p
            if p.get('phase') == 'warmup':
                current['warm'].append(p)
        elif kind == 'fresh_observation':
            r = p['receipt']
            current['frames'].append({k: deepcopy(p[k]) for k in ('receipt', 'pre', 'post')})
            idx = len(current['frames']) - 1
            expected = ([('warmup', s, k) for s in range(11) for k in range(3)] +
                        [('plan', s, k) for s in range(6) for k in range(3)])
            need(idx < len(expected) and (r.get('phase'), r.get('step'), r.get('overlay_index')) == expected[idx], 'frame chronology')
            need(r.get('condition') == ('on' if r['overlay_index'] == 1 else 'off'), 'frame overlay condition')
            need(p['pre'] == p['post'] and r['snapshot'] == p['post']['snapshot'] and r['state'] == r['snapshot']['state'], 'frame native invariant')
            if r['step'] > 0:
                transition = current['transitions'].get((r['phase'], r['step']))
                need(transition is not None and transition['state'] == r['state'] and
                     transition['snapshot'] == r['snapshot'], 'frame/transition state')
            if r['phase'] == 'plan':
                need(len(current['goal']) == 1 and p['pre']['goal_state'] == current['goal'][0]['goal'], 'frame goal binding')
            name = r.get('png')
            need(type(name) is str and Path(name).name == name and name.endswith('.png'), 'frame path')
            raw = basic.file_bytes(Path(directory)/name)
            need(len(raw) <= 1024*1024 and hashlib.sha256(raw).hexdigest() == r['file_sha256'], 'frame file SHA')
            with Image.open(io.BytesIO(raw)) as image:
                need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'native RGB224 PNG')
                pixels = np.asarray(image).copy()
            need(outer.h3.moving.image_sha(pixels) == r['pixel_sha256'], 'frame pixel SHA')
            outer.h3.moving.guard(pixels, outer.h3.source.vector(r['state']))
            if idx % 3 == 2:
                first = current['frames'][idx-2]['receipt']
                need(first['pixel_sha256'] == r['pixel_sha256'] and first['state'] == r['state'] and
                     first['snapshot'] == r['snapshot'] and current['frames'][idx-2]['pre'] ==
                     current['frames'][idx-1]['pre'] == p['pre'], 'off/on/off replay mismatch')
        elif kind == 'k17_guard':
            need(current['guard'] is None and p.get('accepted') is True and len(current['frames']) == 36, 'guard placement')
            item, task, reference = sources[branch]
            need(len(current['placement']) == len(current['goal']) == 1 and len(current['warm']) == 10, 'guard source event completeness')
            placement, attached = current['placement'][0], current['goal'][0]
            need(placement['reset_identity'] == reference['reset_identity'] and attached['goal'] == reference['goal_state'], 'original reset/goal identity')
            first = current['frames'][33]
            bound = first['pre']
            need(attached['snapshot'] == bound['snapshot'] and attached['goal_pose'] == bound['goal_pose'] and
                 attached['goal_rgb_sha256'] == bound['goal_rgb_sha256'], 'goal binding event')
            warm = []
            for step, transition in enumerate(current['warm'], 1):
                need(transition['step'] == step, 'warm return ordering')
                warm.append({'step': step, 'status': 'returned', 'actual_dtype': transition['actual_dtype'],
                             'requested_f32_hex': transition['requested_bits'], 'actual_f32_hex': transition['actual_bits'],
                             'terminated': transition['terminated'], 'truncated': transition['truncated']})
            reconstructed = {'reset_identity': placement['reset_identity'], 'warm_frames': current['frames'][:33],
                             'warm_returns': warm, 'root_state': current['warm'][-1]['state'],
                             'goal_state': attached['goal'], 'bound_goal': bound, 'goal_frame': first}
            proof = gate.capture_check(item, task, reference, reconstructed, directory)
            need(basic.transport.canonical(proof) == basic.transport.canonical(p.get('proof')), 'saved guard proof differs from recomputation')
            current['guard'] = proof
    if source_files() != source_sha256 or material() != material_sha256:
        raise ValueError('guard source drift')
    return {'schema': 'p3_k17_saved_native_guard_frames.v1', 'journal_sha256': journal_sha256,
            'source_files_sha256': source_sha256, 'branches': len(slots),
            'material_sha256': material_sha256,
            'frames_checked': sum(len(s['frames']) for s in slots),
            'guards_recomputed': sum(s['guard'] is not None for s in slots),
            'native_provenance_verified': False, 'scientific_pass': False}
