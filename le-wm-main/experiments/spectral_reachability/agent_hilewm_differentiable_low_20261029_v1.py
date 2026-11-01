"""Caller-grad-owned CUDA low facade over the frozen trusted data graph.

No construction, parameter copy, old node dispatch, model import or runner.
RAM tests use symbolic tensors, not a CPU neural fallback.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

TAG = 'agent_hilewm_differentiable_low_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py':
        'df313393f1ead36a512dc549f8883a1da85dd3e7ffa057f690d39627a1e7b411',
    'module.py': '0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620',
    'results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__models__jepa.py.source.txt':
        'bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2',
}
CALLABLE = frozenset(('Linear', 'NonDynamicallyQuantizableLinear', 'LayerNorm',
    'BatchNorm1d', 'GELU', 'SiLU', 'Identity', 'Dropout', 'Sequential', 'MLP',
    'FeedForward', 'Attention', 'ConditionalBlock', 'Transformer', 'ARPredictor', 'Embedder'))
DATA_ONLY = frozenset(('ModuleList', 'Conv1d', 'Softmax'))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def source_closure(t=None):
    """Bounded source-only hashes; never reads weights or upstream reports."""
    out = {}
    paths = {ROOT/k: v for k, v in PINS.items()}
    paths.update({ROOT/'experiments/spectral_reachability'/(TAG+'.py'): None,
                  ROOT/'tests'/('test_'+TAG+'.py'): None,
                  ROOT/'innerdocs'/(TAG+'.md'): None})
    if t is not None:
        folder = Path(t.__file__).resolve().parent
        paths.update({folder/p: None for p in ('nn/functional.py', 'autograd/grad_mode.py',
                     'nn/modules/module.py', '_tensor.py')})
    for p, expected in paths.items():
        need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 4*1024**2,
             'bounded regular kernel source: '+str(p))
        b = p.read_bytes()
        need(len(b) <= 4*1024**2, 'source cap')
        h = hashlib.sha256(b).hexdigest()
        need(expected is None or h == expected, 'pinned source: '+str(p))
        out[str(p)] = h
    return out


def protocol():
    return dict(schema=TAG+'.protocol', device='MAIN HOST CUDA only',
        state='same original parameters/buffers; no registration or copies',
        gradient='caller owns grad mode; inference mode rejected; disabled grad rejected',
        scope=['action_encoder', 'low_predictor', 'low_pred_proj'],
        guard='all graph eval/frozen CUDA tensors; identity/version/layout/literal attributes',
        content_hash_guard='caller must persist authenticated all-weight before/after byte hashes',
        original_class_parity=False, CUDA_verified=False, gradient_verified=False,
        P2_pairing=False, raw_inverse=False, scientific_admission=False)


class _Guard:
    def __init__(self, root, t):
        self.root, self.t = root, t
        self.initial = None
        self.initial = self.snapshot()

    def snapshot(self):
        t = self.t
        need(t.cuda.is_available(), 'HOST CUDA required')
        need(not t.is_inference_mode_enabled() and t.is_grad_enabled(),
             'caller must enable ordinary autograd')
        modules = tuple(self.root.modules())
        need(modules and all(m.training is False for m in modules), 'all original graph eval')
        tensors = tuple(self.root.parameters()) + tuple(self.root.buffers())
        need(tensors, 'nonempty original frozen graph')
        device = tensors[0].device
        need(device.type == 'cuda', 'no CPU neural fallback')
        rows = []
        for x in tensors:
            need(t.is_tensor(x) and x.device == device and not x.requires_grad
                 and not t.is_inference(x), 'original frozen non-inference CUDA weights/buffers')
            rows.append((id(x), x._version, tuple(x.shape), tuple(x.stride()),
                         x.storage_offset(), x.dtype, x.device))
        attrs = tuple((id(m), deepcopy(m.declared_class_not_imported),
                       deepcopy(m.literal_source_attributes),
                       tuple((k, id(v)) for k, v in m._modules.items())) for m in modules)
        return device, tuple(rows), attrs

    def check(self):
        now = self.snapshot()
        need(self.initial is None or now == self.initial, 'frozen graph identity/version/attributes changed')
        return now[0]

    def input(self, x):
        t = self.t
        need(t.is_tensor(x) and x.device == self.initial[0] and x.dtype == t.float32
             and not t.is_inference(x), 'original CUDA float32 non-inference input')


class _Node:
    """Not nn.Module: forwards only via these owned equations, never the owner."""
    def __init__(self, owner, guard):
        self._owner, self._guard = owner, guard
        self._kind = owner.declared_class_not_imported[1]
        need(self._kind in CALLABLE | DATA_ONLY, 'uncertified low node: '+self._kind)
        self._children = {k: _Node(v, guard) for k, v in owner._modules.items()}

    @property
    def _modules(self):
        return dict(self._children)

    def __getattr__(self, name):
        owner = self._owner
        if name in self._children:
            return self._children[name]
        if name in owner._parameters:
            return owner._parameters[name]
        if name in owner._buffers:
            return owner._buffers[name]
        if name in owner.literal_source_attributes:
            return deepcopy(owner.literal_source_attributes[name])
        raise AttributeError('no delegated executable/implicit attribute: '+name)

    def __call__(self, *args, **kwargs):
        g = self._guard
        g.check()
        need(not kwargs and len(args) in (1, 2), 'explicit low positional API')
        for x in args:
            g.input(x)
        x = args[0]
        k, a, f = self._kind, self._owner.literal_source_attributes, g.t.nn.functional
        need(len(args) == (2 if k in ('ConditionalBlock', 'Transformer', 'ARPredictor') else 1),
             'exact node arity')
        try:
            if k in ('Linear', 'NonDynamicallyQuantizableLinear'):
                return f.linear(x, self.weight, self.bias)
            if k == 'LayerNorm':
                return f.layer_norm(x, a['normalized_shape'], self.weight, self.bias, a['eps'])
            if k == 'BatchNorm1d':
                need(a['track_running_stats'] is True, 'stored BN running stats required')
                return f.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                    training=False, momentum=a['momentum'], eps=a['eps'])
            if k == 'GELU':
                return f.gelu(x, approximate=a['approximate'])
            if k == 'SiLU':
                return f.silu(x)
            if k in ('Identity', 'Dropout'):
                return x
            if k == 'Sequential':
                for child in self._children.values():
                    x = child(x)
                return x
            if k in ('MLP', 'FeedForward'):
                return self.net(x)
            if k == 'Attention':
                norm = self.norm(x)
                q, key, v = self.to_qkv(norm).chunk(3, dim=-1)
                heads = a['heads']
                def split(value):
                    return value.reshape(value.shape[0], value.shape[1], heads, -1).transpose(1, 2)
                need(a['scale'] == (q.shape[-1]//heads)**-0.5, 'explicit source attention scale')
                y = f.scaled_dot_product_attention(split(q), split(key), split(v),
                                                   dropout_p=0.0, is_causal=True)
                return self.to_out(y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))
            if k == 'ConditionalBlock':
                sm, cm, gm, sf, cf, gf = self.adaLN_modulation(args[1]).chunk(6, dim=-1)
                x = x + gm*self.attn(self.norm1(x)*(1+cm)+sm)
                return x + gf*self.mlp(self.norm2(x)*(1+cf)+sf)
            if k == 'Transformer':
                x, c = self.input_proj(x), self.cond_proj(args[1])
                for block in self.layers._modules.values():
                    x = block(x, c)
                return self.output_proj(self.norm(x))
            if k == 'ARPredictor':
                need(x.shape[1] <= self.pos_embedding.shape[1], 'predictor context bound')
                return self.transformer(self.dropout(x+self.pos_embedding[:, :x.shape[1]]), args[1])
            if k == 'Embedder':
                n = self.patch_embed
                need(tuple(n.weight.shape) == (10, 10, 1) and tuple(n.stride) == (1,)
                     and tuple(n.padding) == (0,) and tuple(n.dilation) == (1,) and n.groups == 1,
                     'literal original action Conv1d')
                return self.embed(f.conv1d(x.float().permute(0, 2, 1), n.weight, n.bias).permute(0, 2, 1))
            raise ValueError('data-only node not callable: '+k)
        finally:
            g.check()


class LowFacade:
    def __init__(self, root, t):
        self._root = root
        self._guard = _Guard(root, t)
        for name in ('action_encoder', 'low_predictor', 'low_pred_proj'):
            setattr(self, name, _Node(getattr(root, name), self._guard))

    def parameters(self, *args, **kwargs):
        return self._root.parameters(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self._root.buffers(*args, **kwargs)

    def modules(self):
        return self._root.modules()

    def named_parameters(self, *args, **kwargs):
        return self._root.named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self._root.named_buffers(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._root.state_dict(*args, **kwargs)

    def assert_frozen(self):
        self._guard.check()
        return dict(identity_version_eval_unchanged=True, content_hash_certified=False)


def wrap(graph, torch, *, main_authorized=False):
    """Use only on MAIN's authenticated constructed graph, in enabled grad mode.

    Pass the facade to the normalized seam AND independent low reference.
    This does not authenticate a graph/receipt, prove gradients, or authorize a run.
    """
    need(main_authorized is True, 'MAIN authorization before graph/runtime access')
    source_closure(torch)
    return LowFacade(graph, torch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('protocol', 'verify-sources'))
    args = p.parse_args()
    print(json.dumps(protocol() if args.command == 'protocol' else source_closure(), sort_keys=True))


if __name__ == '__main__':
    main()
