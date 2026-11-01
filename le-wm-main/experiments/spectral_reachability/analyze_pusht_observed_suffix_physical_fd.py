"""Saved physical FD and image observability diagnostic, no policy changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def cosine(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    norm = np.linalg.norm(left)*np.linalg.norm(right)
    return float(np.sum(left*right)/norm) if norm > 0 else None


def royal_blue_pixels(image):
    """Canonical default renderer only; absence is NOT a general RGB detector."""
    image = np.asarray(image)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError('official uint8 RGB required')
    return int(np.count_nonzero(np.all(image == [65, 105, 225], axis=-1)))


def summarize(report, image_reader):
    if report['status'] != 'completed' or not report['gates'] or not all(v is True for v in report['gates'].values()):
        raise ValueError('completed physical FD required')
    roots = report['rows']
    if len(roots) != 4 or len({r['root_sha256'] for r in roots}) != 4:
        raise ValueError('four distinct roots required')
    rows = []
    for root in roots:
        if len(root['branches']) != 2 or len({b['candidate_id'] for b in root['branches']}) != 2:
            raise ValueError('two fixed distinct prefixes required')
        for branch in root['branches']:
            probes = branch['probes']
            if len(probes) != 9 or any(p['status'] != 'executed' for p in probes) or len({p['id'] for p in probes}) != 9:
                raise ValueError('all nine declared probes required')
            j = np.asarray(branch['saved_model_jacobian'], float)
            counts = [royal_blue_pixels(image_reader(p)) for p in probes]
            comparisons = []
            for comparison in branch['comparisons']:
                if comparison['status'] != 'compared':
                    raise ValueError('complete derivative comparison required')
                physical = np.asarray(comparison['physical_jacobian'], float)
                comparisons.append({'epsilon': comparison['epsilon'], 'raw_jacobian_cosine': cosine(j, physical),
                    'model_norm': float(np.linalg.norm(j)), 'physical_norm': float(np.linalg.norm(physical)),
                    'relative_error': float(np.linalg.norm(j-physical)/np.linalg.norm(physical)) if np.linalg.norm(physical) else None})
            states = {tuple(p['state']) for p in probes}
            hashes = {p['rgb_sha256'] for p in probes}
            rows.append({'root_index': root['root_index'], 'root_sha256': root['root_sha256'], 'candidate_id': branch['candidate_id'],
                         'canonical_blue_pixel_counts': counts, 'distinct_states': len(states), 'distinct_rgb': len(hashes),
                         'all_images_identical_but_states_differ': len(hashes) == 1 and len(states) > 1,
                         'canonical_agent_color_absent_all_probes': all(n == 0 for n in counts), 'comparisons': comparisons})
    return {'schema': 'pusht_physical_fd_observability.v1', 'rows': rows, 'P1_admission': False,
            'selector_changed': False, 'dropped_branches': 0,
            'scope': 'known-root local physical derivative and default-color observability diagnostic; no universal detector or theory verdict'}


if __name__ == '__main__':
    from PIL import Image
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = json.loads(args.report.read_text())
    def read(probe):
        if digest(probe['rgb_path']) != probe['rgb_file_sha256']:
            raise ValueError('saved physical RGB file changed')
        value = np.asarray(Image.open(probe['rgb_path']).convert('RGB'))
        if hashlib.sha256(value.tobytes()).hexdigest() != probe['rgb_sha256']:
            raise ValueError('decoded RGB differs from captured RGB')
        return value
    result = summarize(report, read)
    result['input'] = {'path': str(args.report.resolve()), 'sha256': digest(args.report)}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    for row in result['rows']:
        print(json.dumps({k: row[k] for k in ('root_index', 'candidate_id', 'canonical_blue_pixel_counts', 'distinct_states', 'distinct_rgb', 'all_images_identical_but_states_differ')}))
