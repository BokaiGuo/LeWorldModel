"""Fixed40 saved history/action/target localization. No model or physics execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_library40_rank_error_20261028_v1 as rank
from experiments.spectral_reachability import agent_p1_teacher_cross_library_physical_20261028_v1 as physical

NAME = 'agent_p1_teacher_library40_history_localization_20261028_v1'
ROOT = rank.ROOT
RANK = ROOT/'results/agent_p1_teacher_library40_rank_error_20261028_v1_analysis_MAIN01.json'
RANK_SHA = 'e23e60588fa9766b690e605022e089656c4c80ad185853773e23a942b5e0228f'
TEACHER_PREP = ROOT/'results/agent_p1_teacher_cross_library_20261028_v1_prepare_MAIN01.json'
TEACHER_PREP_SHA = '5140f9c6baeadc1c9889e65797ebdb4bec9b68db530f06467556c5fb743058e1'
rgb, teacher = rank.rgb, rank.rgb.teacher
require, sha, digest, canonical = rank.require, rank.sha, rank.digest, rank.canonical


def bits(a):
    a = np.asarray(a, dtype='<f4')
    require(np.isfinite(a).all(), 'finite literal float32 authority')
    return a.tobytes().hex()


def equal_bits(a, b, message):
    require(np.asarray(a).shape == np.asarray(b).shape and bits(a) == bits(b), message)


def history_contract(warm, prefix, suffix, blind):
    """Public indices are 1-based; state0 is the original reset observation."""
    warm, prefix, suffix = [teacher.probe.raw(x, n) for x, n in ((warm, 10), (prefix, 5), (suffix, 5))]
    fixed = np.concatenate((np.tile(np.asarray([.1, 0.], '<f4'), (5, 1)),
                            np.tile(np.asarray([0., .1], '<f4'), (5, 1))))
    equal_bits(warm, fixed, 'original nonzero warm10, no substitute history')
    raw = np.concatenate((warm[5:], prefix))
    equal_bits(blind['raw_history'], raw, 'teacher raw history exactly absolute6..15')
    equal_bits(blind['context']['action_history'], teacher.kernel.normalized_history(raw),
               'original float32 mean/std and five-action token grouping')
    return dict(root_latent_absolute_steps=[0, 5, 10], teacher_latent_absolute_steps=[5, 10, 15],
        predictor_action_absolute_steps=[list(range(6, 11)), list(range(11, 16)), list(range(16, 21))],
        output_latent_absolute_steps=[10, 15, 20], output_action_absolute_steps=[list(range(11, 16)), list(range(16, 21))],
        teacher_raw_bits=bits(raw), prefix_raw_bits=bits(prefix), suffix_raw_bits=bits(suffix),
        full_public20_raw_bits=bits(np.concatenate((warm, prefix, suffix))),
        prefix_journal_steps=list(range(11, 16)), suffix_journal_steps=list(range(1, 6)),
        suffix_absolute_steps=list(range(16, 21)))


def raw_trace(events, job, warm, prefix, suffix):
    expected = [('warmup', i+1, x) for i, x in enumerate(warm)]
    expected += [('prefix', i+11, x) for i, x in enumerate(prefix)]
    expected += [('suffix', i+1, x) for i, x in enumerate(suffix)]
    observed = [e for e in events if e['kind'] == 'raw_return' and e['job_digest'] == job]
    require(len(observed) == len(expected) == 20, 'all20 actual public returns, including missingRGB12')
    for e, (phase, step, command) in zip(observed, expected, strict=True):
        require((e['phase'], e['step']) == (phase, step) and
                e['raw_bits'] == e['actual_bits'] == bits(command) and e['actual_dtype'] == '<f4',
                'actual raw-return bits and literal phase/order; signed zero preserved')
    return [e['event_sha256'] for e in observed]


def target_contract(task, anchor, blind, row, plan, observed):
    require(task['task_id'] == anchor['task_id'] == plan['task_id'] == row['task_id'] == observed['task_id'], 'same task identity')
    c, t = task['task_contract'], task['task']
    require((c['root_slot'], c['condition']) == (row['root_slot'], row['condition']) and
            c['blind_sha256'] == row['blind_sha256'] == anchor['blind_sha256'] == plan['blind_sha256'] == observed['blind_sha256'],
            'same root/condition/blind, no off/on pooling')
    require(t['root_context'] == anchor['root_context'] and t['goal'] == anchor['goal'] == blind['goal'] == row['goal192'], 'original latent target/context')
    require(np.array_equal(np.asarray(blind['q']), np.eye(192)) and np.array_equal(t['projection'], blind['q']), 'frozen identity192 Q')
    binding = dict(pixel_sha256=c['pixel_sha256'][3], state=observed['goal'], pose=observed['goal_pose'])
    require(row['goal_binding'] == binding and t['goal_source_sha256'] == binding['pixel_sha256'] == observed['goal_pixel_sha256'], 'original target RGB/native goal/outline binding')
    require(c['renderer_only'] is True and c['goal_pose_edited'] is False, 'renderer-only intervention, outline is not constructed taskgoal')
    return dict(task_id=row['task_id'], blind_sha256=row['blind_sha256'], goal192_digest=digest(row['goal192']),
                native_goal_state=deepcopy(observed['goal']), native_outline_pose=deepcopy(observed['goal_pose']),
                goal_pixel_sha256=binding['pixel_sha256'], outline_is_taskgoal=False, goal_or_Q_changed=False)


def locate(analysis, checks):
    require(len(checks) == 40 and [r['index'] for r in checks] == list(range(40)), 'all40 checked slots, no drops')
    rebuilt = rank.assemble(deepcopy(analysis['rows']))
    for row, checked in zip(rebuilt['rows'], checks, strict=True):
        require(checked['input_binding_status'] == 'saved_exact' and all(checked[k] == row[k] for k in
                ('index', 'root_slot', 'condition', 'role', 'library_label', 'actual_Q', 'predicted_Q')),
                'localization must use matching completed slot checks, not unrelated check labels')
    for key in rebuilt:
        require(rebuilt[key] == analysis[key], 'independent fixed40 exact-sign/matching reconstruction '+key)
    groups = []
    reversals = []
    for group in rebuilt['groups']:
        pairs = []
        for p in group['pairs']:
            item = deepcopy(p)
            item['history_binding_indices'] = [p['i'], p['j']]
            item['input_binding_verified'] = True
            pairs.append(item)
            if p['reversal'] is True:
                reversals.append(dict(item, role=group['role']))
        groups.append(dict(anchor_index=group['anchor_index'], root_slot=group['root_slot'], condition=group['condition'], role=group['role'],
            indices=group['indices'], summary=rank.pair_summary(pairs), pairs=pairs))
    return dict(fixed_slots=40, rows=checks, groups=groups, reversals=reversals,
        reversal_count=len(reversals), within_role_summary=rebuilt['within_role_summary'],
        same_word_ZS=rebuilt['same_word_ZS'], same_word_ZS_summary=rebuilt['same_word_ZS_summary'], rank_rows=rebuilt['rank_rows'])


def build():
    analysis = rgb.read(RANK, RANK_SHA)
    require(analysis == rank.build(), 'full original rank/native arrays rebuild, not hash-only acceptance')
    bindings = deepcopy(analysis['bindings']); bindings[str(RANK)] = RANK_SHA
    prepared = rgb.read(TEACHER_PREP, TEACHER_PREP_SHA); bindings[str(TEACHER_PREP)] = TEACHER_PREP_SHA
    def upstream(key):
        name, h = rgb.INPUTS[key]; return rgb.read(ROOT/name, h)
    compiled, witnessed, observed = upstream('compiler'), upstream('compiler_receipt'), upstream('physical')
    base = ROOT/Path(rgb.INPUTS['compiler'][0]).parent
    pbase = ROOT/Path(rgb.INPUTS['physical'][0]).parent
    jp = pbase/observed['journal']['file']
    require(sha(jp) == observed['journal']['sha256'], 'whole physical journal SHA')
    events = physical.read_events(jp, observed['journal_tail']); bindings[str(jp)] = sha(jp)
    arrays = {}
    def load(receipt):
        value = teacher.load_array(base, receipt); path = teacher.flat(base, receipt['file'])
        require(str(path) not in arrays or arrays[str(path)] == receipt, 'unique original array receipt')
        arrays[str(path)] = deepcopy(receipt); bindings[str(path)] = receipt['file_sha256']; return value
    require(len(prepared['anchors']) == len(compiled['images']) == len(compiled['rows']) == 8 and
            len(witnessed['plans']) == len(observed['rows']) == 40, 'fixed8 anchors/40 plans/40 actual branches')
    checks = []
    for row, plan, obs in zip(analysis['rows'], witnessed['plans'], observed['rows'], strict=True):
        i = row['index']; ai, word = divmod(i, 5)
        anchor, cr, image = prepared['anchors'][ai], compiled['rows'][ai], compiled['images'][ai]
        require(all(row[k] == plan[k] == obs[k] for k in rgb.IDENTITY if k != 'job_digest') and
                row['job_digest'] == obs['job_digest'] and obs['teacher_context_sha256'] == plan['teacher_context_sha256'], 'exact original40 identities')
        task = teacher.overlay.read(anchor['task_file']); require(sha(anchor['task_file']) == anchor['task_sha256'], 'original task export SHA')
        bindings[anchor['task_file']] = anchor['task_sha256']
        z = load(image['latent']); blind = teacher.blind_anchor(anchor, z)
        require(cr['blind_input'] == blind and digest(blind['context']) == plan['teacher_context_sha256'], 'actual z15 insertion, not predicted endpoint')
        equal_bits(blind['context']['latent_history'], anchor['root_context']['latent_history'][1:]+[z.tolist()], 'exact z5/z10/z15 rolling')
        for key, value in [('latents', blind['context']['latent_history']), ('tokens', blind['context']['action_history']), ('raw_history', blind['raw_history']), ('goal', blind['goal'])]:
            equal_bits(load(cr['arrays'][key]), value, 'saved original input array '+key)
        warm = teacher.probe.raw(task['task']['root_raw_action_history'], 10)
        prefix, suffix = teacher.probe.raw(plan['prefix_commands'], 5), teacher.probe.raw(plan['suffix_commands'], 5)
        equal_bits(prefix, anchor['prefix_commands'], 'exact original role prefix')
        if anchor['role'] == 'Z': equal_bits(prefix, np.zeros((5, 2), '<f4'), 'literal Z zero5')
        for name, value in [('commands', np.concatenate((prefix, suffix))), ('prefix_commands', prefix), ('suffix_commands', suffix)]:
            equal_bits(plan[name], value, 'plan full10 concatenation '+name); equal_bits(obs[name], value, 'physical action binding '+name)
        require(row['suffix_raw_bits'] == bits(suffix), 'rank-error suffix witness')
        h = history_contract(warm, prefix, suffix, blind)
        action_events = raw_trace(events, row['job_digest'], warm, prefix, suffix)
        target = target_contract(task, anchor, blind, row, plan, obs)
        frames = obs['trajectory']
        selected = []
        for phase, step, pixel in [('warmup', 5, task['task_contract']['pixel_sha256'][1]), ('warmup', 10, task['task_contract']['pixel_sha256'][2]), ('prefix', 5, anchor['pixel_sha256'])]:
            hits = [f for f in frames if (f['phase'], f['step']) == (phase, step)]
            require(len(hits) == 1 and hits[0]['condition'] == row['condition'] and hits[0]['pixel_sha256'] == pixel, 'true same-condition RGB state5/10/15')
            f = hits[0]; teacher.rgb(pbase/f['png'], f['file_sha256'], pixel); bindings[str(pbase/f['png'])] = f['file_sha256']
            selected.append(deepcopy(f))
        forwards = []
        for offset, stage in [(2+word, 'evaluate_'+str(word)), (7+word, 'replay_'+str(word))]:
            f, e = cr['forwards'][offset], cr['evaluations'][offset-2]
            require(f['stage'] == e['stage'] == stage and e['library_index'] == word, 'literal evaluate/replay slot')
            equal_bits(f['raw_commands'], suffix, 'forward suffix exact'); equal_bits(e['raw_commands'], suffix, 'evaluation suffix exact')
            terminal = teacher.check_forward(f, blind, load)
            teacher.kernel.validate_advance(blind['context'], suffix, e['context'])
            equal_bits(load(e['latents']), e['context']['latent_history'], 'saved advanced latent history')
            equal_bits(load(e['tokens']), e['context']['action_history'], 'saved advanced action history')
            equal_bits(e['context']['latent_history'][:2], blind['context']['latent_history'][1:], 'advance z10/z15 unchanged')
            equal_bits(e['context']['latent_history'][-1], terminal, 'actual prediction output20')
            require(e['cost'] == row['predicted_Q'], 'original Q unchanged')
            forwards.append(dict(stage=stage, expected_input_digest=digest(f['expected']), output_receipt=deepcopy(e['latents']), tokens_receipt=deepcopy(e['tokens'])))
        checks.append(dict(index=i, root_slot=row['root_slot'], condition=row['condition'], role=row['role'], library_label=row['library_label'],
            job_digest=row['job_digest'], anchor_id=row['anchor_id'], plan_digest=row['plan_digest'], history=h, target=target,
            physical_raw_return_events=action_events, RGB_history_frames=selected, forwards=forwards,
            input_binding_status='saved_exact', actual_Q=row['actual_Q'], predicted_Q=row['predicted_Q'], missing_reason=row['missing_reason'],
            causal_error_source='unknown_not_identified_by_input_alignment'))
    own = [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    for path in own: bindings[str(path)] = sha(path)
    dependencies = [Path(m.__file__).resolve() for m in (rank, rgb, teacher, physical)]
    for path in dependencies:
        require(str(path) not in bindings or bindings[str(path)] == sha(path), 'unchanged native saved-validator dependency')
        bindings[str(path)] = sha(path)
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.analysis', status='fixed40_saved_history_localized_not_admitted', bindings=bindings,
        source_files={str(p): bindings[str(p)] for p in own}, array_receipts=arrays, rank_report_sha256=RANK_SHA,
        dependency_sources={str(p): bindings[str(p)] for p in dependencies},
        P1_admission=False, full40_released=False, winner_selection=False, goal_or_Q_changed=False,
        new_neural_calls=0, new_physics_calls=0, known_roots=2, independent_roots=0, missing=[12],
        equal_FLOPs=None, physical_benefit_claim=False, **locate(analysis, checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'verify'))
    parser.add_argument('--output'); parser.add_argument('--analysis'); parser.add_argument('--analysis-sha')
    args = parser.parse_args(); a = build()
    if args.command == 'verify':
        require(rgb.read(args.analysis, args.analysis_sha, 8*1024**2) == a, 'entire deterministic saved localization rebuild')
        out = dict(verified=True, analysis_sha256=args.analysis_sha)
    elif args.output:
        p = Path(args.output).absolute(); raw = canonical(a)
        require(p.resolve() == p and p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix == '.json' and not p.exists() and len(raw) <= 8*1024**2,
                'exclusive bounded own-stem new result only')
        with p.open('xb') as f:
            require(f.write(raw) == len(raw), 'whole report write'); f.flush(); os.fsync(f.fileno())
        rgb.sync_dir(p.parent)
        out = dict(output=str(p), analysis_sha256=sha(p))
    else: out = dict(status=a['status'], fixed_slots=40, reversal_count=a['reversal_count'], groups=[{k:g[k] for k in ('root_slot','condition','role','summary')} for g in a['groups']])
    print(canonical(out).decode())


if __name__ == '__main__': main()
