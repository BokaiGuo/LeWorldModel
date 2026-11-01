"""Saved sixteen-candidate audit using the existing independent stdlib scalar math."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import main_p3_R_gate_decomposition_20261029_v1 as scalar

ROOT = scalar.ROOT
NAME = 'main_p3_secant_gate_scalar_20261029_v1'
PRODUCER = 'agent_p3_reach_secant_actual_cp5_cuda_20261029_v1'
AUTH = {
    'prepared': (PRODUCER+'_prepare_MAIN01.json', 'b3cc19fe7361e6bcc03ebf533a9a5acc115438cb6efda2f3c5655e7f680799e6'),
    'report': (PRODUCER+'_cuda_MAIN01/report.json', '191684824c95b39fcaf4cd2683ad3f3f7cd071009cdaaa076014bd9f67df91e7'),
    'receipt': (PRODUCER+'_consume_MAIN01.json', 'd96f97dd076894901edea43f6223cf8cb32da4dc630968fea8fae7834b143151'),
}


def explain(p, r, c):
    scalar.need(c['schema'] == PRODUCER+'.consume' and c['groups'] == 4 and c['candidate_denominator'] == 16
                and c['status'] == 'all4_actual_cp5_secant_groups_verified_NOT_delivery', 'complete upstream receipt')
    scalar.need(r['schema'] == PRODUCER+'.report' and r['status'] == 'four_groups_settled_NOT_delivery'
                and r['errors'] == [] and r['delivery'] is None, 'complete diagnostic only')
    roster = [(0, 0, 1), (1, 0, 3), (2, 1, 1), (3, 1, 3)]
    scalar.need([(x['slot'], x['root'], x['horizon']) for x in r['rows']] == roster, 'fixed four groups')
    scalar.need(len(p['source_groups']) == len(c['arms']) == 4, 'all source and receipt groups')
    out = []
    for row, inherited, arm in zip(r['rows'], p['source_groups'], c['arms'], strict=True):
        identity = (row['slot'], row['root'], row['horizon'])
        scalar.need(identity == (inherited['slot'], inherited['root'], inherited['horizon'])
                    == (arm['slot'], arm['root'], arm['horizon']), 'literal source identity join')
        result = row['result']; task = p['tasks'][row['root']]
        scalar.need(task['projection'] == [[float(i == j) for j in range(192)] for i in range(192)], 'identity192 only')
        scalar.need(result['status'] == arm['status'] == row['status'] == 'all_four_scored'
                    and arm['complete'] is True and result['errors'] == [] and result['delivery'] is None,
                    'all four actual candidates, no delivery')
        scalar.need([x['index'] for x in result['rows']] == list(range(4)), 'fixed sixteen denominator')
        zero = inherited['zero']; values = []
        for candidate in result['rows']:
            v = scalar.scalar(candidate, task['goal'])
            scalar.need(candidate['status'] == 'complete', 'complete actual native score')
            # Compare the ORIGINAL saved float64 scores, not the recomputation
            # tolerance. The stdlib scalar bound is arithmetic-only.
            di = candidate['immediate']-zero['immediate']
            dr = candidate['reach']-zero['reach']
            eligible = di > 0 and dr < 0
            scalar.need(candidate['nongreedy'] is (di > 0) and candidate['better_reach'] is (dr < 0)
                        and candidate['eligible'] is eligible, 'original exact gate unchanged')
            values.append(dict(index=candidate['index'], radius=candidate['radius'],
                saved_delta_immediate=di, saved_delta_reach=dr, eligible=eligible,
                recomputed=v, independent_immediate_error=v['immediate']-candidate['immediate'],
                independent_reach_error=v['reach']-candidate['reach']))
        indices = [x['index'] for x in values if x['eligible']]
        scalar.need(indices == result['eligible_indices'] == arm['eligible_indices'], 'whole qualification list')
        out.append(dict(slot=row['slot'], root=row['root'], horizon=row['horizon'],
                        zero=zero, candidates=values, eligible_indices=indices))
    return dict(schema=NAME+'.analysis', groups=out, denominator=16,
        math='reuse existing independent stdlib math.fsum scalar kernel; no producer math import',
        upstream_fullstrict_rebuilt_here=False, new_neural_calls=0, new_physics_calls=0,
        scientific_admission=False, freshness=False, delivery=None, gate_tolerance_changed=False)


def build():
    docs, refs = {}, {}
    for key, (name, h) in AUTH.items():
        path = ROOT/'results'/name
        scalar.need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256*1024**2
                    and scalar.sha(path) == h, 'fixed bounded actual input '+key)
        docs[key] = json.loads(path.read_bytes()); refs[key] = dict(path=str(path), sha256=h)
    scalar.need(docs['receipt']['prepare_sha256'] == AUTH['prepared'][1]
                and docs['receipt']['report_sha256'] == AUTH['report'][1], 'exact upstream joins')
    out = explain(docs['prepared'], docs['report'], docs['receipt'])
    out['authorities'] = refs
    files = (Path(__file__).resolve(), Path(scalar.__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'))
    out['sources'] = {str(p): scalar.sha(p) for p in files}
    scalar.need(all(scalar.sha(v['path']) == v['sha256'] for v in refs.values()), 'stable inputs')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze', 'verify')); ap.add_argument('--output', required=True)
    args = ap.parse_args(); p = Path(args.output).absolute()
    scalar.need(p.parent == ROOT/'results' and p.name.startswith(NAME+'_') and p.resolve() == p, 'exclusive own result')
    b = scalar.encode(build())
    if args.command == 'verify': scalar.need(p.read_bytes() == b, 'entire saved reconstruction')
    else:
        with p.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
        fd = os.open(p.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    print(json.dumps(dict(path=str(p), sha256=scalar.sha(p))))


if __name__ == '__main__': main()
