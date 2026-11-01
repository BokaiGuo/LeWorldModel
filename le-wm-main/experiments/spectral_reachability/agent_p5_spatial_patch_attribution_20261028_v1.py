"""Pure-saved all-word/all-patch diagnostic; never a planner or model runner."""
from __future__ import annotations

import argparse
import base64
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v5 as owner

TAG = 'agent_p5_spatial_patch_attribution_20261028_v1'
ROOT = owner.ROOT
WORDS = owner.WORDS
OWNER_SHA = 'aece72c75efa24f8c5e2df12aff38a954df31307a0476edf28be2295b1320025'
MANIFEST_SHA = '79c3c2f1aa1b8f1b9223e8d87f22986f5c3437144e10bbe78a6cc556093beafd'
REPORT_SHA = '9505f332fc6c1cbb670d9a77f46290f8d2822486b00d89006ea038359898bdc5'
RECEIPT_SHA = '33f3f24735131fd8b569c1d0a87068d1b3f4e26dab31618a4afb90005cf8c927'
sha, read_json, write_new = owner.impl.sha, owner.impl.read_json, owner.impl.write_new
require = owner.require
CAP = 8 * 1024**2
MAPS = ('predicted_goal', 'real_goal', 'prediction_real', 'cross', 'real_minus_predicted_goal')


def protocol():
    return dict(schema=TAG, words=list(WORDS), roots=1, patches=256, channels=384,
                grid=[16, 16], patch_pixels=[14, 14], order='row-major; no CLS/register token',
                partition='all 16 fixed 4x4-patch tiles; no masks/selection/segmentation',
                reduction='float64 channel mean, then uniform mean of all 256 patches',
                identity='R-G=(P-G)+(R-P); real_goal=predicted_goal+prediction_real+cross',
                cross='2*mean_channel((P-G)*(R-P)), signed; not a separate physical cost',
                rgb_units='raw uint8 squared-channel units; never added to feature MSE',
                localization='token coordinate only; attention is global, not causal region attribution',
                new_encoder_API=0, new_predictor_API=0, new_forward_samples=0, physics=0,
                training=False, fitting=False, masking=False, rescoring=False, winner_output=False,
                fresh=False, prospective=False, scientific_admission=False, C2_established=False,
                task_success=None, output_cap_bytes=CAP)


def inputs(manifest, report, receipt):
    """Full exact saved consumer, not just status/SHA inspection."""
    require(sha(owner.__file__) == OWNER_SHA, 'frozen v5 source')
    require(tuple(map(sha, (manifest, report, receipt))) ==
            (MANIFEST_SHA, REPORT_SHA, RECEIPT_SHA), 'actual authority SHA')
    m = read_json(manifest, MANIFEST_SHA)
    r = read_json(report, REPORT_SHA)
    c = read_json(receipt, RECEIPT_SHA)
    require(owner.consume(m, Path(report), REPORT_SHA) == c, 'exact v5 receipt return')
    a = m['authority']
    goal_rgb = a['gm']['public_manifest']['inputs']['goal_rgb']
    rgb = {w: owner.array(a['terminals'][w]['rgb'], (224, 224, 3), np.uint8) for w in WORDS}
    goal_pixels = owner.array(goal_rgb, (224, 224, 3), np.uint8)
    g = owner.array(a['goalrec'], (256, 384))
    p = {w: owner.array(a['prediction'][w], (3, 256, 384))[-1] for w in WORDS}
    real = {w: owner.array(r['terminal_features'][w], (256, 384)) for w in WORDS}
    # Hash-only source provenance: no importing external model/transform code.
    suffixes = ('dinov2/layers/patch_embed.py', 'dinov2/models/vision_transformer.py',
                'app/plan_common/datasets/transforms.py')
    seals = a['model_spec']['loader_manifest']['seals']
    provenance = {}
    for suffix in suffixes:
        found = [(path, h) for path, h in seals.items() if path.endswith('/' + suffix)]
        require(len(found) == 1 and sha(found[0][0]) == found[0][1], 'grid/preprocess source closure')
        provenance.update(found)
    records = [a['goalrec'], goal_rgb, *a['prediction'].values(),
               *r['terminal_features'].values(), *(a['terminals'][w]['rgb'] for w in WORDS)]
    bindings = {str(Path(x).resolve()): sha(x) for x in (manifest, report, receipt)}
    bindings.update({rec['path']: rec['sha256'] for rec in records})
    bindings.update(provenance)
    for path in (Path(__file__), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md')):
        bindings[str(path.resolve())] = sha(path)
    return m, r, g, p, real, rgb, goal_pixels, bindings


def checked(x, shape, dtype):
    require(x.shape == shape and x.dtype == dtype and np.isfinite(x).all(), 'array ABI/finite')
    return x.astype(np.float64)


def pixel_map(x, y):
    x = checked(x, (224, 224, 3), np.uint8)
    y = checked(y, (224, 224, 3), np.uint8)
    return ((x-y)**2).reshape(16, 14, 16, 14, 3).mean(axis=(1, 3, 4))


def summarize(maps):
    """Every patch is present; contributions sum to original full-image metric."""
    require(all(x.shape == (16, 16) and np.isfinite(x).all() for x in maps.values()), 'all patch maps')
    tiles = []
    for row in range(4):
        for col in range(4):
            tiles.append(dict(tile=[row, col], patch_bounds=[4*row, 4*(row+1), 4*col, 4*(col+1)],
                pixel_bounds=[56*row, 56*(row+1), 56*col, 56*(col+1)], patches=16,
                full_metric_contribution={k: float(x[row*4:row*4+4, col*4:col*4+4].sum()/256)
                                          for k, x in maps.items()}))
    return dict(patch_channel_mean={k: x.tolist() for k, x in maps.items()},
                patch_full_metric_contribution={k: (x/256).tolist() for k, x in maps.items()},
                full_mean={k: float(x.mean()) for k, x in maps.items()}, tiles=tiles)


def decompose(goal, predictions, real, rgb, goal_rgb):
    require(tuple(predictions) == tuple(real) == tuple(rgb) == WORDS, 'all four fixed-order words')
    g = checked(goal, (256, 384), np.float32)
    rows = []
    for w in WORDS:
        p = checked(predictions[w], (256, 384), np.float32)
        r = checked(real[w], (256, 384), np.float32)
        d, e = p-g, r-p
        maps = dict(predicted_goal=np.mean(d*d, axis=1).reshape(16, 16),
                    real_goal=np.mean((r-g)**2, axis=1).reshape(16, 16),
                    prediction_real=np.mean(e*e, axis=1).reshape(16, 16),
                    cross=(2*np.mean(d*e, axis=1)).reshape(16, 16))
        maps['real_minus_predicted_goal'] = maps['real_goal']-maps['predicted_goal']
        residual = maps['real_goal']-maps['predicted_goal']-maps['prediction_real']-maps['cross']
        require(np.allclose(residual, 0, rtol=0, atol=1e-10), 'per-patch identity')
        maps['raw_rgb_goal'] = pixel_map(rgb[w], goal_rgb)
        rows.append(dict(word=w, status='diagnostic', feature_units='squared latent units/channel',
                         rgb_units='uint8 squared units/channel', identity_max_abs_residual=float(abs(residual).max()),
                         **summarize(maps)))
    pairs = []
    for left, right in (('xy', 'yx'), ('x_minus_x', 'zero_zero')):
        i, j = WORDS.index(left), WORDS.index(right)
        maps = {k: np.array(rows[i]['patch_channel_mean'][k])-np.array(rows[j]['patch_channel_mean'][k])
                for k in (*MAPS, 'raw_rgb_goal')}
        maps['real_feature_pair'] = np.mean((real[left].astype(np.float64)-real[right].astype(np.float64))**2,
                                           axis=1).reshape(16, 16)
        maps['raw_rgb_pair'] = pixel_map(rgb[left], rgb[right])
        pairs.append(dict(left=left, right=right, direction='left minus right except nonnegative pair distances',
                          **summarize(maps)))
    return dict(rows=rows, pairs=pairs)


def png_uri(rgb):
    """Lossless original RGB embedding, standard library only; no image decoder/model."""
    checked(rgb, (224, 224, 3), np.uint8)
    def chunk(kind, payload):
        return struct.pack('!I', len(payload))+kind+payload+struct.pack('!I', zlib.crc32(kind+payload))
    scan = b''.join(b'\0'+row.tobytes() for row in rgb)
    png = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!IIBBBBB', 224, 224, 8, 2, 0, 0, 0))
           +chunk(b'IDAT', zlib.compress(scan))+chunk(b'IEND', b''))
    return 'data:image/png;base64,'+base64.b64encode(png).decode('ascii')


def figure(data, rgb, goal_rgb):
    """All words and all patches; shared scale per metric, no selected ROI."""
    keys = ('predicted_goal', 'real_goal', 'prediction_real', 'raw_rgb_goal')
    scales = {k: max(float(np.max(row['patch_channel_mean'][k])) for row in data['rows']) for k in keys}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="1340" viewBox="0 0 1320 1340">',
           '<rect width="1320" height="1340" fill="white"/>',
           '<text x="10" y="20">All 4 words / 256 patches. Token coordinates are NOT causal object regions.</text>',
           '<text x="10" y="42">Goal RGB (lossless); 4x4 tiles fixed before analysis. Feature and RGB units remain separate.</text>',
           f'<image x="10" y="55" width="224" height="224" href="{png_uri(goal_rgb)}"/>']
    for i, row in enumerate(data['rows']):
        y = 320+250*i
        svg.append(f'<text x="10" y="{y-10}">{row["word"]}</text>')
        svg.append(f'<image x="10" y="{y}" width="224" height="224" href="{png_uri(rgb[row["word"]])}"/>')
        for j, key in enumerate(keys):
            x = 260+260*j
            svg.append(f'<text x="{x}" y="{y-10}">{key}; max={scales[key]:.6g}</text>')
            for r in range(16):
                for c in range(16):
                    v = row['patch_channel_mean'][key][r][c]
                    level = round(255*(1-(v/scales[key] if scales[key] else 0)))
                    svg.append(f'<rect x="{x+14*c}" y="{y+14*r}" width="14" height="14" fill="rgb(255,{level},{level})"><title>r={r},c={c},value={v:.17g}</title></rect>')
        for x in (10, 260, 520, 780, 1040):
            for t in range(5):
                svg.append(f'<path d="M{x+56*t} {y}v224 M{x} {y+56*t}h224" stroke="black" stroke-opacity="0.4" fill="none"/>')
    svg.append('</svg>')
    return '\n'.join(svg)


def build(manifest, report, receipt):
    m, r, g, p, real, rgb, goal_rgb, bindings = inputs(manifest, report, receipt)
    data = decompose(g, p, real, rgb, goal_rgb)
    for row, sealed in zip(data['rows'], r['words']):
        require(row['word'] == sealed['word'], 'sealed word alignment')
        for k, old in (('predicted_goal', 'predicted_goal_mse'), ('real_goal', 'real_terminal_goal_mse'),
                       ('prediction_real', 'prediction_to_real_feature_mse')):
            require(np.isclose(row['full_mean'][k], sealed[old], rtol=1e-12, atol=1e-12), 'sealed full score reconstruction')
    value = dict(schema=TAG, status='all4_saved_spatial_diagnostics', protocol=protocol(), bindings=bindings,
                 authority=dict(manifest=str(Path(manifest).resolve()), report=str(Path(report).resolve()),
                                receipt=str(Path(receipt).resolve())), **data,
                 scientific_admission=False, C2_established=False, task_success=None,
                 semantic_regions=None, causal_attribution=None,
                 interpretation='global attention/agent/goal raster confounding unresolved; not task benefit',
                 reused_meter=r['meter'], new_neural_API=0, new_physics=0)
    return value, figure(data, rgb, goal_rgb)


def verify(report, svg):
    value = read_json(report)
    wanted, picture = build(**value['authority'])
    require(value == wanted and Path(svg).read_text() == picture, 'full saved recomputation')
    require(Path(report).stat().st_size+Path(svg).stat().st_size <= CAP, 'output cap')
    return dict(status='all4_spatial_diagnostics_verified', scientific_admission=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('analyze', 'verify'))
    p.add_argument('--manifest', type=Path)
    p.add_argument('--report', type=Path, required=True, help='new analysis JSON')
    p.add_argument('--svg', type=Path, required=True)
    p.add_argument('--producer-report', type=Path)
    p.add_argument('--receipt', type=Path)
    a = p.parse_args(argv)
    if a.command == 'verify':
        result = verify(a.report, a.svg)
    else:
        require(a.manifest and a.producer_report and a.receipt, 'three actual saved authorities required')
        require(a.report.resolve() != a.svg.resolve() and not a.report.exists() and not a.svg.exists(), 'exclusive new outputs')
        value, picture = build(a.manifest, a.producer_report, a.receipt)
        require(len(json.dumps(value).encode())*2+len(picture.encode()) <= CAP, 'output cap before writes')
        write_new(a.report, value)
        with a.svg.open('x') as f:
            f.write(picture)
        result = verify(a.report, a.svg)
    print(json.dumps(result if a.command == 'verify' else {'status': result['status'], 'scientific_admission': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
