"""Saved-only full-denominator raster audit; no rendering or neural inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_p5_static_baseline_diff_20261028_v1'
PREP = ROOT/'results/agent_p5_renderer_factorial_20261028_v3_prepare_MAIN01.json'
PREP_SHA = 'fdf76d7f0d404065bdee5abd404d4f39e0f68877263cc4862ce2dc84fe20109a'
REPORT = ROOT/'results/agent_p5_renderer_factorial_20261028_v3_render_MAIN01/report.json'
REPORT_SHA = '76f8488c852572960730f8a11082e2636636f0e0b27884766f7b7ffbd7328c93'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(actual, original):
    need(actual.shape == original.shape == (224, 224, 3), 'RGB shape')
    need(actual.dtype == original.dtype == np.uint8, 'RGB dtype')
    mask = np.any(actual != original, axis=-1)
    ys, xs = np.where(mask)
    return dict(changed_pixels=int(mask.sum()),
        bbox=None if not mask.any() else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        max_abs=int(np.abs(actual.astype(np.int16)-original.astype(np.int16)).max()),
        pixels=[dict(x=int(x), y=int(y), reconstructed=actual[y,x].tolist(), original=original[y,x].tolist())
                for y,x in zip(ys,xs)])


def analyze():
    need(sha(PREP) == PREP_SHA and sha(REPORT) == REPORT_SHA, 'actual fixed inputs')
    p, r = json.loads(PREP.read_bytes()), json.loads(REPORT.read_bytes())
    need(r['global_error'] == 'ValueError: original baseline NPY/RGB byte gate', 'actual baseline failure')
    need(r['all_five_baselines_before_toggles'] is False and r['neural_executed'] is False, 'closed factor gate')
    need(all(c['status'] == 'not_run' for c in r['cells']), 'no color factor executed')
    for name, rec in r['artifacts'].items():
        path = REPORT.parent/name
        need(path.parent == REPORT.parent and sha(path) == rec['sha256'] and path.stat().st_size == rec['bytes'], 'artifact closure')
    sources = p['native_manifest']['states']
    need([s['state_id'] for s in sources] == [s['state_id'] for s in r['rows']], 'complete five-state denominator')
    rows = []
    for source, observed in zip(sources, r['rows']):
        need(sha(source['rgb']['path']) == source['rgb']['sha256'], 'original RGB binding')
        shots = observed['shots']
        item = dict(state_id=source['state_id'], status=observed['status'], compared=False, diff=None)
        if shots:
            need(len(shots) == 1 and shots[0]['target'] == 'baseline', 'baseline only')
            shot = shots[0]
            before = REPORT.parent/shot['files']['before']['path']
            after = REPORT.parent/shot['files']['after']['path']
            need(before.read_bytes() == after.read_bytes() and shot['state_unchanged'] is True, 'public state unchanged')
            actual = REPORT.parent/shot['files']['rgb']['path']
            original = Path(source['rgb']['path'])
            item.update(compared=True, npy_bytes_equal=actual.read_bytes() == original.read_bytes(),
                diff=differences(np.load(actual, allow_pickle=False), np.load(original, allow_pickle=False)))
        rows.append(item)
    evidence = []
    for path, expected, snippets in [
        (ROOT/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py',
         '6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723',
         ['def draw_dot(', 'self.space.debug_draw(draw_options)', 'pygame.draw.circle(self.surface, color.as_int()']),
        (ROOT/'.jepa-physics-site/pymunk/space_debug_draw_options.py',
         '0693f0fb6460fba13e5cc94463108ee83bbd64ce01e115433eb974abe4a89c6a',
         ['self.collision_point_color = SpaceDebugColor(231, 76, 60, 255)', '| SpaceDebugDrawOptions.DRAW_COLLISION_POINTS'])]:
        need(sha(path) == expected, 'bound rendering source')
        lines = path.read_text().splitlines()
        evidence.append(dict(path=str(path), sha256=expected,
            lines=[dict(line=i+1, text=line.strip()) for i,line in enumerate(lines) if any(s in line for s in snippets)]))
    need(all(r['meter']['entries'][k] == 0 for k in ('space_step','reset','control_step')), 'static only')
    return dict(schema=NAME, prepare_sha256=PREP_SHA, report_sha256=REPORT_SHA, rows=rows,
        source_evidence=evidence, meter=r['meter'],
        diagnosis='public-state-identical reconstruction is not sufficient for exact RGB; contact-cache debug drawing is a source-supported hypothesis, not yet causally proven',
        required_next='original constructor/seed/reset/actions replay retaining native contact cache, all five byte gates before any factor',
        neural_calls=0, new_physics_calls=0, scientific_admission=False, threshold_relaxed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['analyze','verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = analyze()
    raw = (json.dumps(value, sort_keys=True, separators=(',',':'))+'\n').encode()
    if args.command == 'analyze':
        need(args.output.absolute().parent == ROOT/'results' and args.output.name.startswith(NAME), 'own output')
        with args.output.open('xb') as stream:
            stream.write(raw)
    else:
        need(args.output.read_bytes() == raw, 'full saved recomputation')
    print(json.dumps(dict(status=args.command, sha256=sha(args.output), rows=len(value['rows']))))


if __name__ == '__main__':
    main()
