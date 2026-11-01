"""Reconstruct observed stage cost at the fixed40 source anchors; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p1_teacher_library40_history_localization_20261028_v1 as history

ROOT = history.ROOT


def stage_cost(start, end, goal):
    values = [np.asarray(x, dtype=np.float64) for x in (start, end, goal)]
    if any(x.shape != (192,) or not np.isfinite(x).all() for x in values):
        raise ValueError('stage vectors must be finite 192-vectors')
    a, b, g = values
    before, after = float(np.mean((a-g)**2)), float(np.mean((b-g)**2))
    return dict(start_Q=before, end_Q=after, delta_Q=after-before)


def build():
    # Full original reconstruction verifies source identities, action history and RGB.
    checked = history.build()
    prepared = history.rgb.read(history.TEACHER_PREP, history.TEACHER_PREP_SHA)
    name, sha = history.rgb.INPUTS['compiler']
    compiled = history.rgb.read(ROOT/name, sha)
    folder = (ROOT/name).parent
    rows = []
    for i, (anchor, source) in enumerate(zip(prepared['anchors'], compiled['rows'], strict=True)):
        cr = source['blind_input']['context']
        start = anchor['root_context']['latent_history'][-1]
        end = cr['latent_history'][-1]
        if not np.array_equal(np.asarray(cr['latent_history'][-2], dtype='<f4'), np.asarray(start, dtype='<f4')):
            raise ValueError('observed absolute-step 10 identity mismatch')
        goal = source['blind_input']['goal']
        linked = checked['rows'][i*5:i*5+5]
        if len(linked) != 5 or any(x['anchor_id'] != anchor['anchor_id'] for x in linked):
            raise ValueError('anchor to fixed40 binding mismatch')
        # Re-read bound arrays to avoid trusting JSON alone.
        latents = history.teacher.load_array(folder, source['arrays']['latents'])
        if not np.array_equal(latents, np.asarray(cr['latent_history'], dtype='<f4')):
            raise ValueError('authenticated observed latent mismatch')
        rows.append(dict(anchor_index=i, anchor_id=anchor['anchor_id'], task_id=anchor['task_id'],
                         indices=[x['index'] for x in linked], start_step=10, end_step=15,
                         observed_stage=stage_cost(start, end, goal)))
    return dict(schema='p1_fixed40_observed_stage_20261029', rows=rows, fixed_candidates=40,
                known_roots=2, neural_calls=0, physical_calls=0, P1_admission=False,
                interpretation='Observed RGB-encoded stage cost, not model-predicted stage cost or physical success; repeated suffixes share each prefix.',
                source_bindings=checked['bindings'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if not out.is_absolute() or out.parent != ROOT/'results' or out.exists():
        raise ValueError('exclusive absolute results path required')
    print('Reconstruct observed step 10 to 15 stage cost for all fixed40 anchors; no neural or physical calls.', flush=True)
    data = json.dumps(build(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with out.open('xb') as stream:
        stream.write(data)
    print(json.dumps(dict(path=str(out), sha256=hashlib.sha256(data).hexdigest())))


if __name__ == '__main__':
    main()
