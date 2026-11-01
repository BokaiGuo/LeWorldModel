"""Prepare fixed real cell03 encoder inputs for later CUDA parity, never run a model."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host

ROOT = Path('/home/gbk/code/LeWorldModel/le-wm-main/results')
OUTPUT = ROOT / 'agent_e3_chunked_identity_host_20261029_v1_CELL03_RECOVERY01_V3'
AUTHORITY = ROOT / 'e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json'
REPORT_SHA = '7b5620d3ed7adc89932c2752519c0a2423503fcde9609a397e17a68d5f541d32'
COMPLETION_SHA = 'e5185eb3e5583674df5415983b851e6ee2dd73437a9a9e7ffeddd8b7a44e5c8f'
AUTHORITY_SHA = '1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4'
ROLES = ('encoder_hidden', 'projector_input', 'projector_output', 'latent')
SHAPES = {'encoder_hidden': (1, 257, 192), 'projector_input': (1, 192),
          'projector_output': (1, 192), 'latent': (192,)}


def compare(reference, actual):
    """Exact float32 byte equality plus descriptive error; never select a tolerance."""
    if set(reference) != set(ROLES) or set(actual) != set(ROLES):
        raise ValueError('all and only fixed roles required')
    out = {}
    for role in ROLES:
        a, b = np.asarray(reference[role]), np.asarray(actual[role])
        if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != SHAPES[role] or b.shape != SHAPES[role]:
            raise ValueError('shape or dtype mismatch: ' + role)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite: ' + role)
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        out[role] = {'bitwise_equal': a.tobytes() == b.tobytes(),
                     'max_abs_error': float(delta.max())}
    return out


def checked_array(directory, descriptor):
    name = descriptor['file']
    if Path(name).name != name:
        raise ValueError('nonflat array path')
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('array must be regular')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
        raise ValueError('array SHA drift')
    stream = io.BytesIO(data)
    value = np.load(stream, allow_pickle=False)
    if (stream.tell() != len(data) or value.dtype.str != descriptor['dtype']
            or list(value.shape) != descriptor['shape'] or not np.isfinite(value).all()):
        raise ValueError('array encoding mismatch')
    return value


def inputs():
    result = host.inspect(str(OUTPUT), REPORT_SHA, COMPLETION_SHA, str(AUTHORITY), AUTHORITY_SHA)
    if result['base']['status'] != 'completed20':
        raise ValueError('fixed completed20 input required')
    raw = (OUTPUT / 'events.jsonl').read_bytes()
    report = json.loads((OUTPUT / 'report.json').read_bytes())
    if hashlib.sha256(raw).hexdigest() != report['journal_sha256']:
        raise ValueError('journal changed after inspection')
    records = [json.loads(line)['detail'] for line in raw.splitlines()
               if b'"resource":"encoding_complete"' in line]
    if [r['boundary'] for r in records] != [10, 15]:
        raise ValueError('exact two ordered boundaries required')
    result_rows = []
    for row in records:
        frame = row['frame']
        if Path(frame['file']).name != frame['file']:
            raise ValueError('nonflat frame path')
        path = OUTPUT / frame['file']
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError('frame changed after inspection')
        for role, descriptor in row['arrays'].items():
            if role != descriptor['role']:
                raise ValueError('array role mismatch')
            checked_array(OUTPUT, descriptor)
        result_rows.append(dict(boundary=row['boundary'], frame=frame, arrays=row['arrays']))
    return dict(schema='e3_cell03_encoder_parity_inputs_v1', report_sha256=REPORT_SHA,
                completion_sha256=COMPLETION_SHA, authority_sha256=AUTHORITY_SHA,
                rows=result_rows, model_rerun=False, physical_replay=False,
                scientific_admission=False)


if __name__ == '__main__':
    print(json.dumps(inputs(), sort_keys=True))
