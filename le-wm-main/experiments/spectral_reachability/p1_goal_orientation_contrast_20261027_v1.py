"""Goal-RGB-only raster orientation inputs; no model, pose, Q fit or simulator.

This is an image diagnostic, not renderer-faithful physical intervention data.
Angles/ROI/resamplers are fixed in source; failures refuse without fallback.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

NAME = 'p1_goal_orientation_contrast_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
ROI = (48, 48, 176, 176)  # RGB pixel coordinates; not simulator coordinates.
ANGLES = tuple(range(-180, 180, 30))
RESAMPLERS = ('nearest', 'bilinear')
OBJECT_COLOR = (119, 136, 153)
AGENT_COLOR = (65, 105, 225)
CAP = 1_000_000
RESERVE = 200*1024*1024
GOAL_DIR = ROOT/'results/pusht_goal_object_metric4_v1_20261026_cuda'
GOAL_FILES = (
    '6e78f991b65c007f3e5462c768ee65a1f7d7653a42173e18cb72fe0fcd57b5c1',
    'c97330f21e7be307300c85bdbcc380d03db76203bd623dcc577a98e8165d1ac1',
    '2981c12ebd93cbd5a45a122141ef505e71da9c5a84eb3faf07c5dc51b8f1ba2d',
    '48d6f4175eeca4a943a13ee04aea545e227b3e9016c086e120b1606f8b72736d',
)
GOAL_PIXELS = (
    'd2be1dc1db2d28228101349b41fc1b049d46e4bc586988b8b4e530e3e49128ff',
    '5a78f4555312fa07cfb8634cbad1ebd81e9bd07aaf8e20d060a07e87ce2e033f',
    '7adafdf8b005f39ef9fd928592721a1bda5002804b1f3b502bb5d574f6b3f8bf',
    '522047fc2849e13f2696637b0fb85ec505f7edff825550771a0ca1ddf7d66ec7',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < CAP, 'bounded regular input')
    return path.read_bytes()


def rgb(value):
    a = np.asarray(value)
    require(a.shape == (224, 224, 3) and a.dtype == np.uint8, '224x224 uint8 RGB only')
    return a


def white_perimeter(a):
    return all(np.all(x == 255) for x in (a[0], a[-1], a[:, 0], a[:, -1]))


def single_component(mask):
    """Reject detached foreground; never discard it to make an edit pass."""
    coords = np.argwhere(mask)
    require(len(coords) > 0, 'empty foreground')
    seen = set(); todo = [tuple(coords[0])]
    while todo:
        y, x = todo.pop()
        if (y, x) in seen:
            continue
        seen.add((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y+dy, x+dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and (yy, xx) not in seen:
                    todo.append((yy, xx))
    require(len(seen) == len(coords), 'foreign/detached ROI foreground; no inferred background fill')


def moments(patch):
    # Image-derived ink-mass centroid; not physical center of mass or pose.
    weights = (255-patch.astype(float)).sum(axis=-1)
    mass = float(weights.sum())
    require(mass > 0, 'nonempty ink')
    yy, xx = np.indices(weights.shape)
    return mass, (float((xx*weights).sum()/mass), float((yy*weights).sum()/mass))


def isolate(goal):
    goal = rgb(goal)
    agent = np.all(goal == AGENT_COLOR, axis=-1)
    obj = np.all(goal == OBJECT_COLOR, axis=-1)
    require(agent.any() and obj.any(), 'canonical visible agent/object required')
    yy, xx = np.nonzero(agent)
    x0, y0, x1, y1 = int(xx.min())-2, int(yy.min())-2, int(xx.max())+3, int(yy.max())+3
    require(0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224, 'agent margin clipped')
    sprite = goal[y0:y1, x0:x1]
    require(white_perimeter(sprite) and not obj[y0:y1, x0:x1].any(), 'agent not isolated on white; no hidden-object reconstruction')
    clean = goal.copy(); clean[y0:y1, x0:x1] = 255
    left, top, right, bottom = ROI
    require(not obj[:top].any() and not obj[bottom:].any() and not obj[:, :left].any() and not obj[:, right:].any(), 'object outside fixed ROI')
    patch = clean[top:bottom, left:right].copy()
    foreground = np.any(patch != 255, axis=-1)
    require(white_perimeter(patch), 'nonwhite ROI perimeter/background')
    single_component(foreground)
    require(np.array_equal(clean[obj], goal[obj]), 'source object changed during agent removal')
    mass, center = moments(patch)
    # PIL's center is in edge coordinates; our moments use pixel-center indices.
    pivot = (center[0]+.5, center[1]+.5)
    ys, xs = np.nonzero(foreground)
    radius = float(np.max(np.hypot(xs-center[0], ys-center[1]))) + math.sqrt(.5) + 2
    clearance = min(pivot[0], pivot[1], patch.shape[1]-pivot[0], patch.shape[0]-pivot[1])
    require(radius < clearance, 'all-angle support bound fails; no angle filtering or adaptive ROI')
    return clean, patch, dict(agent_bbox=[x0, y0, x1, y1], roi=list(ROI),
        foreground_pixels=int(foreground.sum()), canonical_object_pixels=int(obj.sum()),
        ink_mass=mass, pivot_local_edge=list(pivot), rotation_support_radius_bound=radius,
        all_angle_clearance=clearance, patch_sha256=digest(patch.tobytes()),
        support_sha256=digest(foreground.tobytes()))


def rotate(patch, angle, sampler, pivot):
    method = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR}[sampler]
    return np.asarray(Image.fromarray(patch).rotate(angle, resample=method,
        center=tuple(pivot), expand=False, fillcolor=(255, 255, 255))).copy()


def catalog(goal):
    """No optional angle/ROI knobs, root outcomes or physical-state argument."""
    goal = rgb(goal); clean, patch, info = isolate(goal)
    left, top, right, bottom = ROI
    images = [dict(id='original', role='original_goal', rgb=goal.copy()),
              dict(id='clean', role='agent_removed_reference', rgb=clean.copy())]
    _, center = moments(patch)
    for angle in ANGLES:
        for sampler in RESAMPLERS:
            rotated = rotate(patch, angle, sampler, info['pivot_local_edge'])
            require(white_perimeter(rotated), 'rotated boundary nonwhite; refuse complete catalog')
            edited = clean.copy(); edited[top:bottom, left:right] = rotated
            if angle == 0:
                require(np.array_equal(edited, clean), 'zero-degree interpolation identity')
            restored = rotate(rotated, -angle, sampler, info['pivot_local_edge'])
            mass, rotated_center = moments(rotated)
            images.append(dict(id=f'angle_{angle:+04d}_{sampler}', role='raster_orientation',
                angle_degrees=angle, sampler=sampler, rgb=edited,
                artifact=dict(changed_pixels=int(np.any(rotated != patch, axis=-1).sum()),
                    foreground_pixels=int(np.any(rotated != 255, axis=-1).sum()),
                    ink_mass_ratio=mass/info['ink_mass'],
                    ink_centroid_shift_pixels=[rotated_center[k]-center[k] for k in (0, 1)],
                    roundtrip_rgb_mse=float(np.mean((restored.astype(float)-patch)**2)))))
    sampler_differences = []
    for i, angle in enumerate(ANGLES):
        a, b = images[2+2*i]['rgb'], images[3+2*i]['rgb']
        sampler_differences.append(dict(angle_degrees=angle,
            rgb_mse=float(np.mean((a[top:bottom, left:right].astype(float)-b[top:bottom, left:right])**2))))
    return dict(images=images, isolation=info, sampler_differences=sampler_differences)


def contract():
    return dict(schema=NAME+'.contract', roi=list(ROI), angles_degrees=list(ANGLES),
        resamplers=list(RESAMPLERS), positive_angle='PIL counterclockwise displayed raster',
        pivot='all-channel ink-mass centroid of complete nonwhite isolated ROI; not physical COM',
        backgrounds='white isolated ROI; preserve complete anti-aliased foreground, no pure-color mask rotation',
        references=['original', 'clean'], images_per_goal=26, goals=4, images=104,
        future_encoder_samples=104, future_existing_projector_samples=104,
        actual_neural_calls=0, actual_physical_steps=0, training=False,
        learned_parameters=0, new_Q=False, physical_pose_input=False,
        no_angle_or_root_filtering=True, asset_cap_bytes=CAP,
        diagnostic_only=True, renderer_faithful_physical_states=False,
        metric_validity_claim=False, P1_admission=False, P3_admission=False)


def build():
    payloads = {}; goals = []; frames = []
    for root_index in range(4):
        path = GOAL_DIR/f'root_{root_index}.original_goal.png'
        raw = read(path); require(digest(raw) == GOAL_FILES[root_index], 'original goal file drift')
        with Image.open(BytesIO(raw)) as im:
            require(im.format == 'PNG' and im.mode == 'RGB', 'original RGB PNG only')
            goal = rgb(np.asarray(im).copy())
        require(digest(goal.tobytes()) == GOAL_PIXELS[root_index], 'original goal pixel drift')
        result = catalog(goal)
        goals.append(dict(root_index=root_index, source=str(path), source_sha256=GOAL_FILES[root_index],
            source_pixel_sha256=GOAL_PIXELS[root_index], **{k:v for k,v in result.items() if k != 'images'}))
        for image in result['images']:
            file = f'root_{root_index}.{image["id"]}.png'
            stream = BytesIO(); Image.fromarray(image['rgb']).save(stream, format='PNG', compress_level=9)
            raw = stream.getvalue(); payloads[file] = raw
            frames.append(dict(order=len(frames), root_index=root_index, file=file,
                **{k:v for k,v in image.items() if k != 'rgb'},
                pixel_sha256=digest(image['rgb'].tobytes()), file_sha256=digest(raw)))
    require(len(frames) == 104, 'complete fixed census')
    sources = [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
        ROOT/'experiments/spectral_reachability/p3p4_goal_patch_metric_20261027_v1.py',
        ROOT/'experiments/spectral_reachability/pusht_goal_object_tangent_metric.py']
    manifest = dict(schema=NAME+'.manifest', contract=contract(), goals=goals, frames=frames,
        unique_pixel_images=len({f['pixel_sha256'] for f in frames}),
        unique_clean_object_patches=len({g['isolation']['patch_sha256'] for g in goals}),
        versions=dict(numpy=np.__version__, pillow=PIL_VERSION),
        sources_sha256={str(p):digest(read(p)) for p in sources})
    payloads['manifest.json'] = canonical(manifest)
    payloads['prepare.json'] = canonical(dict(schema=NAME+'.prepared', status='raster_inputs_only_not_encoded',
        manifest_sha256=digest(payloads['manifest.json']), images=104, new_neural_calls=0,
        new_physical_steps=0, P1_admission=False, P3_admission=False))
    require(sum(map(len, payloads.values())) < CAP, 'entire asset output exceeds 1MB; refuse, no downselection')
    return payloads, manifest


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir(), 'exclusive new output with existing parent')
    require(shutil.disk_usage(output.parent).free > RESERVE+CAP, '200MiB storage reserve')
    payloads, manifest = build()
    output.mkdir()
    for file, raw in payloads.items():
        with (output/file).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(prepare_sha256=digest(read(output/'prepare.json')), images=len(manifest['frames']),
        bytes=sum(map(len, payloads.values())), unique_clean_object_patches=manifest['unique_clean_object_patches'],
        unique_pixel_images=manifest['unique_pixel_images'], status='raster_inputs_only_not_encoded')


def verify(output, expected):
    output = Path(output).absolute()
    require(digest(read(output/'prepare.json')) == expected, 'external prepare SHA')
    # Reconstruct every image, all source/goal hashes, order and artifact stats.
    payloads, _ = build()
    require({p.name for p in output.iterdir()} == set(payloads), 'complete output census')
    for file, raw in payloads.items():
        require(read(output/file) == raw, 'saved source/goal/image/metadata drift: '+file)
    return dict(status='verified_raster_inputs_not_encoded', files=len(payloads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-sha')
    args = parser.parse_args()
    result = prepare(args.output) if args.command == 'prepare' else verify(args.output, args.prepare_sha)
    print(canonical(result).decode().strip())


if __name__ == '__main__':
    main()
