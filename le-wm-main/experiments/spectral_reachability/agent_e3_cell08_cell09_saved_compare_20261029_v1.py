"""Reconstruct one fixed known-root, same-goal H1 feedback pair; no execution."""
import argparse
import math
from pathlib import Path

from experiments.spectral_reachability import agent_e3_cell00_cell01_saved_compare_20261029_v1 as old
from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as roster
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal

NAME = 'agent_e3_cell08_cell09_saved_compare_20261029_v1'
ROOT = old.ROOT
base = old.base
need = old.need
SPECS = {
    8: dict(report='04d11c71116212f158beab0c3b6be534109eada4fa3f47576d168740cfb7abbd', completion='577cab704f45e97adc1b10c8c9762c0bd375a8c78fe50fec2226d179f82fa665', authority='5319f6c942c2eefe08ed92811c7b3c6f646c3b01af6d29ee90f6afda716468e2', audit='2d5bcb40bf62bbe097142cef0ee91109a0d99878dc106111bd61c4a4e6dc8430', receipt='a49fb1f7d2cac273a86151f967585119b90c73985a6afb7c5be9c0aa5f04314a'),
    9: dict(report='bf55b30581fc70200432801e1e835959936e8935a39b3f6d4d9858d463b26d2e', completion='9c4649eca9ccbd3cbba2d21d5934b39ff608d5294f872a0eda0cee2b6b208c2d', authority='d06ce5207d600de135b19d0ab7cbcb43d6354fe2fb419712a5d0bdfd42137994', audit='404e3012056177fef0863a63e1757e6ff11056a4c54f392d84a699b1aba230e7', receipt='dc6588fb3649225c73952f1743df4fcc4fde83a1bc5ff96e804a6836ccfc9583'),
}


def path(cell):
    return ROOT / f'results/agent_e3_chunked_identity_host_20261029_v1_CELL{cell:02d}_MAIN01'


def check_pair(left, right):
    need(left['cell'] == 8 and right['cell'] == 9, 'fixed cells')
    g0, g1 = left['guard'], right['guard']
    if g0 is None or g1 is None:
        return dict(aligned=False, reason='missing root guard', checkpoints=[])
    for key in ('original_root_sha256', 'original_goal', 'new_goal', 'new_goal_pixel_sha256', 'history_pixels'):
        need(g0[key] == g1[key], 'root/goal/history mismatch: ' + key)
    a, b = g0['root_observation'], g1['root_observation']
    need(all(a[k] == b[k] for k in ('snapshot', 'state', 'pixel_sha256')), 'root snapshot mismatch')
    c0, c1 = left['cp5'], right['cp5']
    aligned_cp5 = c0 is not None and c1 is not None
    if aligned_cp5:
        need(c0['prefix_raw_bits'] == c1['prefix_raw_bits'], 'prefix raw mismatch')
        need(all(c0['observation'][k] == c1['observation'][k] for k in ('snapshot', 'state', 'pixel_sha256')), 'cp5 mismatch')
    checkpoints = []
    for step in (0, 5, 10, 15, 20):
        pair = []
        for side, guard in ((left, g0), (right, g1)):
            obs = side['observations'].get(step)
            if obs is None:
                pair.append(None)
                continue
            need(obs['goal'] == guard['new_goal'] and obs['goal_rgb_sha256'] == guard['new_goal_pixel_sha256'], 'checkpoint goal mismatch')
            pair.append(dict(state=obs['state'], pixel_sha256=obs['pixel_sha256'], joint_xy_distance=old.distance(obs, guard['new_goal'])))
        delta = None if None in pair or (step >= 5 and not aligned_cp5) else pair[1]['joint_xy_distance'] - pair[0]['joint_xy_distance']
        checkpoints.append(dict(control_step=step, cell08=pair[0], cell09=pair[1], dx_minus_dxy=delta))
    return dict(aligned=True, cp5_aligned=aligned_cp5, reason=None if aligned_cp5 else 'missing cp5 guard', checkpoints=checkpoints)


def load(cell):
    s = SPECS[cell]; p = path(cell); episode = f'cell{cell:02d}'
    authority = ROOT / f'results/agent_e3_cell{cell:02d}_feedback_prepare_20261029_v1_MAIN01.json'
    saved = ROOT / f'results/agent_e3_native_saved_semantics_20261029_v1_CELL{cell:02d}_MAIN01.json'
    receipt = ROOT / f'results/agent_e3_gd_host_cuda_20261029_v1_consume_cell{cell:02d}_MAIN01.json'
    read = lambda file, cap: base.read_bytes(file, cap)
    saved_bytes = read(saved, 16*1024**2)
    need(base.fsha(saved_bytes) == s['audit'], 'saved audit hash')
    audit = old.native.build(str(p), s['report'], s['completion'], episode)
    need(base.decode(saved_bytes) == audit, 'saved audit whole reconstruction')
    authority_ref = dict(path=str(authority), sha256=s['authority'])
    side = old.inspect_saved(str(p), s['report'], s['completion'], episode, cell, authority_ref)
    inspected = host.inspect(str(p), s['report'], s['completion'], str(authority), s['authority'])
    rb = read(p/'report.json', old.native.pub.FINAL_CAP); report = base.decode(rb)
    events = [base.decode(x) for x in read(p/'events.jsonl', base.CAP).splitlines()]
    row = terminal.read(events) if audit['transport_complete'] else None
    returns = [e['detail'] for e in events if e['origin']=='physical' and e['resource']=='native_control_return']
    if cell == 9:
        for r in returns: old.validate_dx_return(r)
    receipt_bytes = read(receipt, 16*1024**2)
    need(base.fsha(receipt_bytes) == s['receipt'], 'control5 receipt hash')
    initial = base.decode(receipt_bytes)
    need(initial['cell'] == cell, 'control5 receipt cell')
    saved_base = inspected['base']['saved']
    nodes = saved_base['feedback']['nodes']
    completion = base.decode(read(p/'completion.json', old.native.pub.FINAL_CAP))
    costs = dict(control5_numerical_axes=initial['axes'], feedback_nodes=[dict(boundary=n.get('boundary'), axes=n['axes']) for n in nodes],
                 native_meter=saved_base['meter'], output_category_bytes=report['category_bytes'],
                 episode_inclusive_wall_seconds=completion['elapsed_seconds'],
                 wall_scope='run start through report fsync/hash; excludes independent completion publication and external initial control5 run',
                 control5_wall_seconds=None, FLOPs=None, gpu_time=None, peak_memory=None)
    return dict(cell=cell, episode=episode, guard=side['guard'], cp5=side['cp5'], observations=side['observations'],
                native_audit=audit, terminal=row, controls=returns, costs=costs, references=dict(**s, output_path=str(p), authority_path=str(authority), saved_path=str(saved), control5_receipt_path=str(receipt)),
                inspect_reverified=True, binding_chunks_complete=inspected['binding_chunks_complete'])


def build():
    expected = [(8, 'r1.g0.h1.dDxy', 'Dxy'), (9, 'r1.g0.h1.dDx', 'Dx')]
    for index, name, domain in expected:
        slot = roster.roster()[index]
        need((slot['cell_id'], slot['root'], slot['goal_index'], slot['horizon_packets'], slot['domain']) == (name, 1, 0, 1, domain), 'fixed roster mismatch')
    left, right = load(8), load(9)
    pair = check_pair(left, right)
    return dict(schema=NAME+'.report', cells={'cell08': left, 'cell09': right}, comparison=pair,
                action_semantics={'Dxy':'two-dimensional raw action chart and per-packet radius 0.75*sqrt(5)',
                    'Dx':'new task raw y positive float32 zero; x remains in original domain',
                    'prefix':'warmup and first five historical control actions are not Dx-masked',
                    'dx_returned_task_actions_checked':sum(r['step']>5 for r in right['controls'])},
                analysis='descriptive same-time known-root pair; no outcome-selected roots, padding, inference or causality',
                missing_policy='missing observations, outcomes, failed attempts and unmeasured costs stay null or explicitly recorded',
                inference=None, physics_replayed=False, neural_recomputed=False, freshness=False, scientific_admission=False, full_saved_strict=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command', required=True)
    for command in ('audit','verify'):
        p=sub.add_parser(command);p.add_argument('--saved',required=True)
        if command=='verify':p.add_argument('--saved-sha256',required=True)
    args=parser.parse_args(); value=build()
    if args.command=='audit':
        p=Path(args.saved);need(p.is_absolute() and p==p.resolve() and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink(), 'exclusive owned output')
        data=base.raw(value)
        import os
        with p.open('xb') as f:
            need(f.write(data)==len(data),'complete output');f.flush();os.fsync(f.fileno())
        base.syncdir(p.parent); sha=base.fsha(data)
    else:
        data=base.read_bytes(Path(args.saved),16*1024**2);sha=base.fsha(data)
        need(sha==args.saved_sha256 and data==base.raw(value),'whole comparison reconstruction')
    print(base.raw(dict(path=args.saved,sha256=sha,verified=args.command=='verify')).decode())


if __name__=='__main__': main()
