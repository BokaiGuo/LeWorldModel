"""Read-only closure of the existing cp10 bridge ABI; not a future authority."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_p3_same_time_capability_probe_cuda_20261029_v1 as stage
from experiments.spectral_reachability import agent_p3_future_cp10_task_projection_20261029_v1 as projection

NAME = 'agent_p3_cp10_saved_source_closure_20261029_v1'
MAX_FILE = 16 * 1024 * 1024


def need(ok, reason):
    if not ok: raise ValueError(reason)


def data(path, expected, limit=MAX_FILE):
    p = Path(path)
    need(p.is_absolute() and p.resolve() == p and not p.is_symlink() and p.is_file(), 'canonical file')
    stage.core.hash_value(expected)
    need(p.stat().st_size <= limit, 'file bound')
    body = p.read_bytes()
    need(len(body) <= limit and hashlib.sha256(body).hexdigest() == expected, 'file SHA')
    return body


def image(frame):
    body = data(frame['png'], frame['file_sha256'])
    with Image.open(io.BytesIO(body)) as im:
        need(im.format == 'PNG' and im.mode == 'RGB' and im.size == (224,224), 'actual RGB PNG')
        pixels = np.asarray(im, dtype=np.uint8).copy()
    need(hashlib.sha256(pixels.tobytes()).hexdigest() == frame['pixel_sha256'], 'pixel SHA')
    need(stage.raw(frame['state']) == stage.raw(frame['snapshot']['state']), 'history state/snapshot')
    return pixels


def array(directory, desc, shape, dtype):
    need(set(desc) == {'file','sha256','array_sha256','shape','dtype'}, 'array descriptor keys')
    name = desc['file']; need(isinstance(name,str) and Path(name).name == name, 'flat array path')
    need(desc['shape'] == list(shape) and desc['dtype'] == dtype, 'array declared shape/dtype')
    body = data(directory/name, desc['sha256'])
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as f:
        unpacked = f.read(MAX_FILE+1)
    need(len(unpacked) <= MAX_FILE, 'bounded decompression')
    stream = io.BytesIO(unpacked)
    a = np.load(stream, allow_pickle=False)
    need(stream.tell() == len(unpacked) and a.shape == shape and str(a.dtype) == dtype and a.flags.c_contiguous, 'array actual shape/dtype')
    need(hashlib.sha256(a.tobytes()).hexdigest() == desc['array_sha256'], 'array contents SHA')
    if dtype == 'float32': need(np.isfinite(a).all(), 'finite latent')
    return a


def verify_row(spec, encoded, task, physical, output):
    """Join one prepared slot to an inherited complete physical receipt row."""
    need(physical['index'] == spec['source_slot'] and physical['root_slot'] == spec['root_slot']
         and physical['arm'] == ('H1' if spec['role'] == 'C' else spec['role']), 'physical slot identity')
    need(physical['status'] == 'horizon_completed' and physical['own_returned'] == 10
         and physical['initial_exact'] is True and physical['prefix_reference_exact'] is True
         and physical['prefix51']['complete'] is True and len(physical['prefix51']['frames']) == 51
         and physical['missing_checkpoints'] == [], 'actual guarded history')
    plan = physical['plan']
    need(plan['task_id'] == spec['base_task_id'] and plan['selection_digest'] == spec['selection_digest']
         and stage.raw(plan['model_task']) == stage.raw(spec['cp5_task']), 'physical original task/plan')
    commands = np.asarray(plan['commands'], dtype='<f4')
    need(commands.shape == (10,2) and commands.tobytes().hex() == spec['causal_raw10_bits']
         and commands.tobytes() == np.asarray(spec['causal_raw10'],dtype='<f4').tobytes(), 'physical raw10')
    frames = [x for x in physical['trajectory'] if x['phase']=='plan' and x['overlay_index']==0 and x['step'] in (0,5,10)]
    need([f['step'] for f in frames] == [0,5,10] and [h['absolute_step'] for h in spec['history_sources']] == [10,15,20], 'three frames')
    for f,h in zip(frames,spec['history_sources'],strict=True):
        need(f['condition'] == 'off' and all(stage.raw(f[k]) == stage.raw(h[k]) for k in
             ('file_sha256','pixel_sha256','state','snapshot')), 'physical frame/history binding')
        image(h)
    h = spec['history_sources'][-1]
    need(all(stage.raw(h[k]) == stage.raw(spec['source'][k]) for k in ('file_sha256','pixel_sha256','state','snapshot'))
         and h['png'] == spec['source']['path'], 'endpoint source')
    need(stage.raw(encoded['source']) == stage.raw(spec['source']) and encoded['status'] == 'encoded', 'encoded source')
    pixels = image(h)
    need(np.array_equal(pixels, array(output,encoded['pixels'],(224,224,3),'uint8')), 'saved encoder pixels')
    latent = array(output,encoded['latent'],(192,),'float32')
    need(latent.tobytes() == np.asarray(task['task']['root_context']['latent_history'][-1],dtype='<f4').tobytes()
         and stage.raw(task['identity']['encoded_latent']) == stage.raw(encoded['latent']), 'latent task identity')
    return latent


def inspect(authority):
    """Historical pinned producer ABI only; inherits its MAIN full receipt."""
    p = stage.build(authority)  # reads all externally SHA-bound sources; no fullstrict/NN
    bp = stage.read(authority['prepare']['path'],authority['prepare']['sha256'])
    br = stage.read(authority['report']['path'],authority['report']['sha256'])
    bs = stage.read(authority['receipt']['path'],authority['receipt']['sha256'])
    output = Path(authority['report']['path']).parent
    latents=[]
    for spec, encoded, item in zip(bp['rows'],br['rows'],bs['tasks'],strict=True):
        report = stage.read(spec['source_report']['path'],spec['source_report']['sha256'])
        receipt = stage.read(spec['source_receipt']['path'],spec['source_receipt']['sha256'])
        need(receipt['physical_report_sha256'] == spec['source_report']['sha256']
             and stage.raw(receipt['rows']) == stage.raw(report['rows']) and receipt['errors'] == report['errors'] == [], 'physical report/receipt')
        latents.append(verify_row(spec,encoded,item,receipt['rows'][spec['source_slot']],output))
    projected=projection.project(bp['rows'],latents,[r['latent'] for r in br['rows']])
    need(all(stage.raw(x['task']['root_context']) == stage.raw(y['task']['root_context']) for x,y in
             zip(projected['endpoints'],p['endpoints'],strict=True)), 'new projection contexts')
    return dict(schema=NAME+'.inspection',status='historical_only_not_future_authority', slots=5, history_frames=15,
                physical_receipts_inherited=True, upstream_fullstrict_rebuilt=False,
                future_authority_verified=False, execution_authorized=False, trusted_timestamp_proven=False,
                fresh_roots_certified=False, scientific_admission=False)


def main():
    ap=argparse.ArgumentParser()
    for key in ('prepare','report','completion','receipt','source'):
        ap.add_argument('--'+key,required=True);ap.add_argument('--'+key+'-sha',required=True)
    a=ap.parse_args()
    refs={k:{'path':getattr(a,k),'sha256':getattr(a,k+'_sha')} for k in ('prepare','report','completion','receipt','source')}
    print(stage.raw(inspect(refs)).decode())


if __name__ == '__main__': main()
