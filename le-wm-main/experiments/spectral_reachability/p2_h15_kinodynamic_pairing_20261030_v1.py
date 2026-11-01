"""Saved-only physical pairing of H15 four-way evidence with H2 beam."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_h15_fourway_pairing_20261029_v1 as four
from experiments.spectral_reachability import agent_kinodynamic_matched_physical_consume_20261028_v1 as kino
from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as producer

NAME = 'p2_h15_kinodynamic_pairing_20261030_v1'
ROOT = kino.ROOT
KDIR = ROOT / 'results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01'
KSHA = '01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7'
MDIR = ROOT / 'results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01'
MSHA = 'd4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c'
need = four.need
METRICS = four.three.old.METRICS


def events(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def select(events_, root, kind, phase=None):
    return [e for e in events_ if e.get('branch') == root and e['kind'] == kind
            and (phase is None or e.get('phase') == phase)]


def pair(prior, hrow, krow, hevents, kevents, hplan, kplan):
    root = prior['root_slot']
    need(hrow['root_slot'] == krow['root_slot'] == root and kplan['root_slot'] == root,
         'fixed root identity')
    need(krow['arm'] == kino.ARM and krow['initial_exact'] is True and hrow['initial_exact'] is True,
         'authenticated branch')
    need(kplan['root_sha256'] == hplan['root_sha256'] == hrow['root_sha256'] == prior['root_sha256']
         and krow['root_state'] == hrow['root_state'] and krow['goal'] == hrow['goal'], 'root/goal mismatch')
    ha = select(hevents, root, 'goal_attached'); ka = select(kevents, root, 'goal_attached')
    need(len(ha) == len(ka) == 1 and ha[0]['snapshot'] == ka[0]['snapshot'] and
         ha[0]['goal'] == ka[0]['goal'] == hrow['goal'] and
         ha[0]['goal_rgb_sha256'] == ka[0]['goal_rgb_sha256'] == hrow['goal_pixel_sha256'],
         'full native snapshot or goal pixel mismatch')
    hw = select(hevents, root, 'returned_transition', 'warmup')
    kw = select(kevents, root, 'returned_transition', 'warmup')
    need([e['step'] for e in hw] == [e['step'] for e in kw] == list(range(1, 11)) and
         all(a['state'] == b['state'] and a['terminated'] == b['terminated'] and
             a['truncated'] == b['truncated'] for a, b in zip(hw, kw, strict=True)), 'warm history mismatch')
    transitions = select(kevents, root, 'returned_transition', 'plan')
    n = krow['own_returned']
    need(type(n) is int and 0 <= n <= 10 and [e['step'] for e in transitions] == list(range(1, n + 1)),
         'actual contiguous beam steps')
    need(all(not (e['terminated'] or e['truncated']) for e in transitions[:-1]), 'beam post-stop')
    lookup = {}
    for e in transitions:
        need(e['returned_goal_state'] == hrow['goal'] and e['returned_goal_rgb_sha256'] == hrow['goal_pixel_sha256'],
             'goal drift')
        m = kino.metric(krow['goal'], e['state'])
        lookup[e['step']] = {**{k: m[k] for k in METRICS}, 'native_success': m['native_success'],
                             'terminated': e['terminated'], 'truncated': e['truncated']}
    timeline = []
    for entry in prior['timeline']:
        beam = lookup.get(entry['step']); item = dict(entry, kinodynamic=beam)
        for method in four.METHODS:
            other = entry[method]
            item['kinodynamic_minus_' + method] = None if beam is None or other is None else {
                k: beam[k] - other[k] for k in METRICS}
        timeline.append(item)
    return dict(root_slot=root, root_sha256=prior['root_sha256'],
                task_exports={'h15_task_id': hplan['task_id'], 'h15_task_sha256': hplan['task_sha256'],
                              'beam_task_id': kplan['task_id'], 'beam_task_sha256': kplan['task_sha256']},
                status=dict(prior['status'], kinodynamic=krow['status']),
                last_step=dict(prior['last_step'], kinodynamic=n),
                common_steps=[x['step'] for x in timeline if all(x[m] is not None for m in (*four.METHODS, 'kinodynamic'))],
                timeline=timeline)


def build():
    prior = four.build()
    k = kino.consume(KDIR, KSHA)
    model = producer.consume(MDIR, MSHA)
    need(k['status'] == 'matched_domain_beam_physical_recorded_not_admitted' and
         model['status'] == 'matched_domain_witnesses_verified_not_admitted' and len(k['rows']) == len(model['plans']) == 2,
         'complete beam source')
    old = four.three.old
    h = old.h15.consume(old.H_DIR, old.H_SHA, old.H_PREP, old.H_PREP_SHA)
    hp = four.P.read(old.H_PREP, four.P.META)
    he, ke = events(old.H_DIR / 'journal.jsonl'), events(KDIR / 'journal.jsonl')
    resources = []
    pairs = []
    model_report = kino.read(MDIR / 'report.json', kino.CAP)
    for root in (0, 1):
        kp = k['prepared']['plans'][root]
        need(model['plans'][root]['source_slot'] == root and
             model['plans'][root]['selection_sha256'] == kp['compiler_selection_sha256'], 'model selection')
        pairs.append(pair(prior['pairs'][root], h['rows'][root], k['rows'][root], he, ke, hp['plans'][root], kp))
        mr = model_report['rows'][root]
        need(mr['source_slot'] == root, 'beam resource root')
        resources.append(dict(root_slot=root, fourway=prior['resources'][root],
                              beam_neural=mr['resources'], beam_predictor_samples=mr['predictor_samples'],
                              beam_physical=four.three.old.per_root_physical(k['report'], root)))
    return dict(schema=NAME+'.report', sources=dict(fourway=prior['sources'], beam_physical_sha256=KSHA,
        beam_model_sha256=MSHA, analysis_code_sha256=four.P.sha(__file__)), pairs=pairs, resources=resources,
        endpoint='same returned step after warm10, not same trajectory or wall time; beam horizon 10, no padding',
        identity='physical native root snapshot, goal state, goal pixels and warmup states checked; task exports differ',
        resource_boundary='beam H2 finite packet bank: 144 predictor samples/root, no VJP; four methods H15 132/132/117/117 and VJP 192/0/768/768; FLOPs and CPU costs not fully measured; different action support and horizon',
        equal_total_compute=False, independent_generalization=False, scientific_admission=False,
        new_neural_calls=0, new_physics_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sha256')
    a = parser.parse_args()
    need(a.output.is_absolute() and a.output.name.startswith(NAME+'_') and not a.output.is_symlink(), 'exclusive namespace')
    if a.command == 'build':
        need(a.sha256 is None, 'build arguments')
        data = (json.dumps(build(), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        with a.output.open('xb') as f:
            need(f.write(data) == len(data), 'complete write'); f.flush(); os.fsync(f.fileno())
        four.P.paired.fsync_dir(a.output.parent)
        result = {'sha256': four.P.sha(a.output)}
    else:
        need(a.sha256 and four.P.sha(a.output) == a.sha256, 'external report SHA')
        need(json.loads(a.output.read_text()) == build(), 'full rebuild')
        result = {'verified': True}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
