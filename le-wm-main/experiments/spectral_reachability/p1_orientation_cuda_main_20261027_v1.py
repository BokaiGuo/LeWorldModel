"""MAIN-only frozen-CUDA diagnostic of the complete fixed RGB contrast catalog.

No Q fitting, outcome inputs, simulator, predictor or task admission. Every
duplicate is retained and charged; angle and sampler rules precede encoding.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import p1_goal_orientation_contrast_20261027_v1 as catalog
from experiments.spectral_reachability import p1_direct_cuda_calibration_main_20261027_v2 as native
from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen

NAME = 'p1_orientation_cuda_main_20261027_v1'
ROOT = native.ROOT
FLOOR = 200 * 1024**2
CAP = 4 * 1024**2
require = catalog.require


def write(path, value):
    raw = catalog.canonical(value)
    require(len(raw) < CAP, 'bounded JSON')
    require(shutil.disk_usage(Path(path).parent).free > FLOOR + len(raw), '200MiB write floor')
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def material(folder, expected):
    folder = Path(folder).absolute()
    catalog.verify(folder, expected)
    manifest = json.loads(catalog.read(folder/'manifest.json'))
    require(len(manifest['frames']) == 104, 'fixed all-frame census')
    original, _ = native.native_metadata()
    files = set(native.io.source_files() + [Path(__file__).resolve(), Path(catalog.__file__),
        Path(native.__file__), Path(screen.__file__), ROOT/'tests'/('test_'+NAME+'.py'),
        ROOT/'innerdocs'/(NAME+'.md')])
    return dict(schema=NAME+'.protocol', catalog=str(folder), catalog_prepare_sha256=expected,
        manifest_sha256=native.io.sha(folder/'manifest.json'), frames=manifest['frames'],
        policy=original['policy'], model_bindings=original['model_bindings'],
        sources_sha256={str(f): native.io.sha(f) for f in files},
        encoder_cap=104, projector_cap=104, batch=1, predict_cap=0, AD_cap=0,
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, walltime_cap_seconds=600,
        peak_allocated_cuda_cap_bytes=3*1024**3, no_retries=True,
        no_cpu_neural_fallback=True, training=False, new_Q=False, P1_admission=False,
        numerical_analysis='native192 MSE to clean/original; paired sampler MSE; exact-input repeat differences',
        interpretation='raster diagnostic only; interpolation/agent removal confounds; not physical equivalence')


def prepare(folder, expected, output):
    output = Path(output).absolute()
    require(not output.exists(), 'exclusive prepare')
    require(shutil.disk_usage(output.parent).free > FLOOR+CAP, 'full reserve')
    p = material(folder, expected)
    output.mkdir()
    write(output/'protocol.json', p)
    write(output/'prepare.json', dict(protocol_sha256=native.io.sha(output/'protocol.json')))
    return native.io.sha(output/'prepare.json')


def verify(folder, expected):
    folder = Path(folder).absolute()
    require(native.io.sha(folder/'prepare.json') == expected, 'external prepare SHA')
    receipt = json.loads((folder/'prepare.json').read_text())
    require(native.io.sha(folder/'protocol.json') == receipt['protocol_sha256'], 'protocol SHA')
    p = json.loads((folder/'protocol.json').read_text())
    require(material(p['catalog'], p['catalog_prepare_sha256']) == p, 'all source/catalog/model bindings')
    return p


def analyze(frames, values):
    z = np.asarray(values, dtype=np.float64)
    require(z.shape == (104,192) and np.isfinite(z).all(), 'complete native192 census')
    require(len(frames) == 104 and [f['order'] for f in frames] == list(range(104)), 'fixed frame order')
    rows = []
    for root in range(4):
        fs = frames[root*26:(root+1)*26]; a = z[root*26:(root+1)*26]
        require(all(f['root_index'] == root for f in fs) and
            [f['id'] for f in fs[:2]] == ['original','clean'], 'original/clean reference binding')
        contrasts = []
        for i in range(2,26,2):
            f,g = fs[i:i+2]
            require(f['angle_degrees'] == g['angle_degrees'] and
                (f['sampler'],g['sampler']) == ('nearest','bilinear'), 'paired fixed samplers')
            contrasts.append(dict(angle_degrees=f['angle_degrees'],
                nearest_to_clean_mse=float(np.mean((a[i]-a[1])**2)),
                bilinear_to_clean_mse=float(np.mean((a[i+1]-a[1])**2)),
                sampler_pair_mse=float(np.mean((a[i]-a[i+1])**2))))
        rows.append(dict(root_index=root, original_to_clean_mse=float(np.mean((a[0]-a[1])**2)), contrasts=contrasts))
    repeats = []
    first = {}
    for f,v in zip(frames,z):
        h = f['pixel_sha256']
        if h in first:
            repeats.append(float(np.max(np.abs(v-first[h]))))
        else:
            first[h] = v
    return dict(rows=rows, unique_pixel_inputs=len(first), repeated_input_pairs=len(repeats),
        maximum_exact_input_repeat_abs_difference=max(repeats, default=0.),
        independent_physical_orientations_established=False, new_Q=False, P1_admission=False)


def run(folder, expected, output):
    p = verify(folder, expected); output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free > FLOOR+CAP, 'new output/full reserve')
    output.mkdir(); values=[]; events=[]; errors=[]; model=None; before=after=None; host=None
    peak=None; started=time.monotonic()
    with (output/'image.events.jsonl').open('x') as journal:
        def event(e):
            require(sum(f.stat().st_size for f in output.iterdir()) < CAP-300000, 'bounded durable output')
            require(shutil.disk_usage(output).free > FLOOR+65536, 'journal disk floor')
            events.append(e)
            journal.write(catalog.canonical(e).decode()); journal.flush(); os.fsync(journal.fileno())
        try:
            import torch
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
            torch.manual_seed(20261027); torch.backends.cuda.matmul.allow_tf32=False
            torch.cuda.reset_peak_memory_stats(device)
            model=load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
            native.io.frozen(model); before=native.io.tensor_hash(model)
            encode=screen.native_image_encoder(model,device,p0.image_transform(),event)
            for f in p['frames']:
                require(time.monotonic()-started < p['walltime_cap_seconds'], 'walltime cap before next image')
                image,_=native.io.pixels(Path(p['catalog'])/f['file'],f['file_sha256'],f['pixel_sha256'])
                event(dict(kind='frame_started',order=f['order'],pixel_sha256=f['pixel_sha256']))
                values.append(encode(image).tolist())
                torch.cuda.synchronize(device)
                peak=torch.cuda.max_memory_allocated(device)
                require(peak <= p['peak_allocated_cuda_cap_bytes'], 'CUDA peak cap')
                event(dict(kind='frame_completed',order=f['order']))
            write(output/'analysis.json', analyze(p['frames'],values))
        except BaseException as exc:
            errors.append(dict(type=type(exc).__name__,message=str(exc)))
        finally:
            if model is not None:
                try:
                    native.io.frozen(model); after=native.io.tensor_hash(model)
                except BaseException as exc:
                    errors.append(dict(stage='final_model',type=type(exc).__name__,message=str(exc)))
            try:
                verify(folder,expected)
            except BaseException as exc:
                errors.append(dict(stage='final_source',type=type(exc).__name__,message=str(exc)))
            counts={k:dict(attempted=sum(e.get('samples',0) for e in events if
                e['kind']=='native_image_attempted' and e['resource']==k),
                completed=sum(e['counts'][k] for e in events if e['kind']=='native_image_completed'))
                for k in ('encoder','projector')}
            complete=(not errors and len(values)==104 and before is not None and before==after and
                counts=={k:dict(attempted=104,completed=104) for k in counts})
            result=dict(status='cuda_raster_diagnostic_completed' if complete else 'partial_or_failed',
                errors=errors,host=host,frames_completed=len(values),resources=counts,
                model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
                prepare_sha256=expected,peak_allocated_cuda_bytes=peak,elapsed_seconds=time.monotonic()-started,
                training=False,new_Q=False,predict_calls=0,physical_steps=0,P1_admission=False)
            write(output/'report.json',result)
    require(complete,'incomplete raster diagnostic; preserve output, no automatic retry')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--catalog',type=Path);parser.add_argument('--catalog-sha')
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha')
    parser.add_argument('--output',type=Path);a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.catalog,a.catalog_sha,a.output)
    elif a.command=='verify': result=verify(a.prepared,a.prepare_sha)['schema']
    else: result=run(a.prepared,a.prepare_sha,a.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__': main()
