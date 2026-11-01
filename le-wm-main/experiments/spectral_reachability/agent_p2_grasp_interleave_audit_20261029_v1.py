"""Source-only GRASP interleave audit; never imports torch or runs a model."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'original': ('experiments/spectral_reachability/agent_p2_history_joint30_grasp_cuda_20261029_v1.py', '6c44630f75259ce107b5b001a8dfc7f50b60c4f7b6ab920371e2994cde1ca9a1'),
    'host': ('experiments/spectral_reachability/agent_p2_intersection_host_cuda_20261029_v1.py', 'a7d572d9dd7b6fea8aa5b29cf76c69036fb19b8d0d704280bed7cb1581d3c596'),
    'common': ('experiments/spectral_reachability/agent_p2_common_resource_grasp_cuda_20261029_v1.py', 'ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d'),
    'installed': ('.venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py', 'ccc2ac2880880f85ea71a5baa2152ba97e3f6dc5cb546d27570795aa6020d3d2'),
}


def method(source, cls, name):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    if len(classes) != 1:
        raise ValueError('class identity')
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(methods) != 1:
        raise ValueError('method identity')
    return methods[0]


def inspect_sources(root=ROOT, sources=SOURCES):
    texts = {}
    for name, (path, expected) in sources.items():
        data = (root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source SHA mismatch: ' + name)
        texts[name] = data.decode('utf-8')
    original = method(texts['original'], 'Native', 'soft')
    host = method(texts['host'], 'GraspBackend', 'soft')
    original_loop = next(n for n in original.body if isinstance(n, ast.For))
    host_loop = next(n for n in host.body if isinstance(n, ast.For))
    def assignments(loop):
        return [n.targets[0].id for n in loop.body if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)]
    if 'dyn' not in assignments(original_loop) or 'dense' not in assignments(original_loop):
        raise ValueError('original loss not in prediction loop')
    if 'dyn' in assignments(host_loop) or 'dense' in assignments(host_loop):
        raise ValueError('host unexpected loss in prediction loop')
    nested = [n for n in host.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate']
    if len(nested) != 1 or not any(isinstance(n, ast.For) for n in nested[0].body):
        raise ValueError('host deferred evaluation missing')
    return {'source_sha256': {k: v[1] for k, v in sources.items()},
            'original': 'interleaved_prediction_loss', 'host': 'all_predictions_then_loss',
            'cuda_parity': 'not_tested', 'production_ready': False}


if __name__ == '__main__':
    print(json.dumps(inspect_sources(), sort_keys=True, indent=2))
