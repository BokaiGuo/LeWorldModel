"""Symbolic RAM equations only: no torch, weights, arrays or neural execution."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.spectral_reachability import agent_hilewm_differentiable_low_20261029_v1 as m


DEVICE = NS(type='cuda')


class Symbol:
    def __init__(self, name='x', shape=(2, 3, 4), grad=True):
        self.name, self.shape, self.requires_grad = name, shape, grad
        self.device, self.dtype, self._version = DEVICE, 'float32', 0

    def stride(self):
        return tuple(1 for _ in self.shape)

    def storage_offset(self):
        return 0

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            size = 1
            known = 1
            for n in self.shape: size *= n
            for n in shape:
                if n != -1: known *= n
            shape[shape.index(-1)] = size//known
        return Symbol('reshape('+self.name+')', tuple(shape), self.requires_grad)

    def transpose(self, a, b):
        shape = list(self.shape); shape[a], shape[b] = shape[b], shape[a]
        return Symbol('transpose('+self.name+')', tuple(shape), self.requires_grad)

    def permute(self, *order):
        return Symbol('permute('+self.name+')', tuple(self.shape[i] for i in order), self.requires_grad)

    def float(self):
        return self

    def chunk(self, n, dim=-1):
        shape = list(self.shape); shape[dim] //= n
        return tuple(Symbol(f'chunk{i}('+self.name+')', tuple(shape), self.requires_grad) for i in range(n))

    def __getitem__(self, index):
        return Symbol('slice('+self.name+')', self.shape, self.requires_grad)

    def _op(self, other, sign):
        return Symbol('('+self.name+sign+(other.name if isinstance(other, Symbol) else str(other))+')',
                      self.shape, self.requires_grad or getattr(other, 'requires_grad', False))

    def __add__(self, other): return self._op(other, '+')
    def __radd__(self, other): return self._op(other, '+')
    def __mul__(self, other): return self._op(other, '*')
    def __rmul__(self, other): return self._op(other, '*')


def weight(name='w', shape=(4, 4)):
    return Symbol(name, shape, False)


class Owner:
    def __init__(self, kind, children=None, attrs=None, params=None, buffers=None):
        self.declared_class_not_imported = ('module', kind)
        self.literal_source_attributes = attrs or {}
        self._modules, self._parameters, self._buffers = children or {}, params or {}, buffers or {}
        self.training = False

    def __getattr__(self, name):
        for values in (self._modules, self._parameters, self._buffers):
            if name in values: return values[name]
        raise AttributeError(name)

    def __call__(self, *args, **kw):
        raise AssertionError('old inference-only forward must never be called')

    def modules(self):
        yield self
        for child in self._modules.values(): yield from child.modules()

    def parameters(self):
        for node in self.modules(): yield from (x for x in node._parameters.values() if x is not None)

    def buffers(self):
        for node in self.modules(): yield from (x for x in node._buffers.values() if x is not None)

    def named_parameters(self): return iter((str(i), x) for i, x in enumerate(self.parameters()))
    def named_buffers(self): return iter((str(i), x) for i, x in enumerate(self.buffers()))
    def state_dict(self): return dict(self.named_parameters())


class Functions:
    def __init__(self): self.calls = []

    def __getattr__(self, name):
        def call(x, *args, **kw):
            self.calls.append((name, x, args, kw))
            shape = x.shape
            if name == 'linear': shape = (*shape[:-1], args[0].shape[0])
            if name == 'conv1d': shape = (shape[0], args[0].shape[0], shape[-1])
            return Symbol(name+'('+x.name+')', shape,
                          x.requires_grad or any(getattr(a, 'requires_grad', False) for a in args))
        return call


def runtime():
    f = Functions()
    return NS(cuda=NS(is_available=lambda: True), is_inference_mode_enabled=lambda: False,
              is_grad_enabled=lambda: True, is_tensor=lambda x: isinstance(x, Symbol),
              is_inference=lambda x: False, float32='float32', nn=NS(functional=f))


def node(kind='Linear', attrs=None, children=None, params=None, buffers=None):
    if params is None and kind in ('Linear', 'NonDynamicallyQuantizableLinear', 'LayerNorm', 'BatchNorm1d'):
        params = dict(weight=weight(), bias=weight('b', (4,)))
    return Owner(kind, children, attrs, params, buffers)


def graph(child):
    return Owner('HiJEPA', dict(action_encoder=child, low_predictor=node(), low_pred_proj=node()))


def facade(child):
    root, t = graph(child), runtime()
    return m.LowFacade(root, t), root, t


@pytest.mark.parametrize('kind', ['Linear', 'NonDynamicallyQuantizableLinear', 'LayerNorm',
                                   'BatchNorm1d', 'GELU', 'SiLU', 'Identity', 'Dropout'])
def test_primitives_symbolic(kind):
    attrs = dict(normalized_shape=[4], eps=1e-6, track_running_stats=True,
                 momentum=.1, approximate='none', p=.7)
    buffers = dict(running_mean=weight('mean', (4,)), running_var=weight('var', (4,))) if kind == 'BatchNorm1d' else {}
    r, owner, t = facade(node(kind, attrs, buffers=buffers))
    x = Symbol(); out = r.action_encoder(x)
    assert out.requires_grad
    if kind in ('Identity', 'Dropout'): assert out is x
    else:
        call = t.nn.functional.calls[0]
        if kind == 'BatchNorm1d': assert call[3] == dict(training=False, momentum=.1, eps=1e-6)
        if kind == 'GELU': assert call[3] == dict(approximate='none')
    assert r.assert_frozen()['content_hash_certified'] is False


@pytest.mark.parametrize('kind', ['Sequential', 'MLP', 'FeedForward'])
def test_container_order(kind):
    seq = node('Sequential', children={'0': node('Linear'), '1': node('GELU', {'approximate': 'tanh'}), '2': node('Linear')})
    child = seq if kind == 'Sequential' else node(kind, children={'net': seq})
    r, _, t = facade(child)
    assert r.action_encoder(Symbol()).requires_grad
    assert [c[0] for c in t.nn.functional.calls] == ['linear', 'gelu', 'linear']


def attention(scale=2**-.5):
    return node('Attention', {'heads': 2, 'scale': scale}, {
        'norm': node('Identity'), 'to_qkv': node('Linear', params={'weight': weight(shape=(12, 4)), 'bias': None}),
        'to_out': node('Identity'), 'attend': node('Softmax')})


def test_attention_layout_causal_and_grad():
    r, _, t = facade(attention())
    out = r.action_encoder(Symbol())
    assert out.shape == (2, 3, 4) and out.requires_grad
    sdp = next(c for c in t.nn.functional.calls if c[0] == 'scaled_dot_product_attention')
    assert sdp[1].shape == (2, 2, 3, 2)
    assert sdp[3] == dict(dropout_p=0., is_causal=True)


def block():
    return node('ConditionalBlock', children={
        'norm1': node('Identity'), 'norm2': node('Identity'), 'attn': attention(),
        'mlp': node('Identity'), 'adaLN_modulation': node('Linear', params={'weight': weight(shape=(24, 4)), 'bias': None})})


def test_conditioned_equations_and_predictor():
    transformer = node('Transformer', children={'input_proj': node('Identity'), 'cond_proj': node('Identity'),
        'layers': node('ModuleList', children={'0': block()}), 'norm': node('Identity'), 'output_proj': node('Identity')})
    predictor = node('ARPredictor', children={'transformer': transformer, 'dropout': node('Dropout')},
                     params={'pos_embedding': weight('pos', (1, 3, 4))})
    r, _, t = facade(predictor)
    y = r.action_encoder(Symbol('z'), Symbol('c'))
    assert y.requires_grad and 'slice(pos)' in y.name and 'chunk2' in y.name and 'chunk5' in y.name
    assert len(t.nn.functional.calls) == 3


def test_embedder_uses_original_conv_weights_and_no_dispatch():
    conv = node('Conv1d', dict(stride=(1,), padding=(0,), dilation=(1,), groups=1),
                params={'weight': weight(shape=(10, 10, 1)), 'bias': weight(shape=(10,))})
    child = node('Embedder', children={'patch_embed': conv, 'embed': node('Identity')})
    r, root, t = facade(child)
    assert r.action_encoder.patch_embed.weight is conv.weight
    assert r.action_encoder.patch_embed.stride == (1,)
    out = r.action_encoder(Symbol(shape=(2, 3, 10)))
    assert out.shape == (2, 3, 10) and out.requires_grad
    assert t.nn.functional.calls[0][2][0] is conv.weight
    assert list(r.parameters()) == list(root.parameters())
    assert list(r.modules()) == list(root.modules())
    assert r.state_dict() == root.state_dict()


@pytest.mark.parametrize('mutation', ['training', 'weight_grad', 'version', 'attribute', 'cpu', 'identity', 'gradmode', 'inference'])
def test_mutations_fail_closed(mutation):
    r, root, t = facade(node())
    if mutation == 'training': root.action_encoder.training = True
    if mutation == 'weight_grad': root.action_encoder.weight.requires_grad = True
    if mutation == 'version': root.action_encoder.weight._version += 1
    if mutation == 'attribute': root.action_encoder.literal_source_attributes['new'] = 1
    if mutation == 'cpu': root.action_encoder.weight.device = NS(type='cpu')
    if mutation == 'identity': root.action_encoder._parameters['weight'] = weight()
    if mutation == 'gradmode': t.is_grad_enabled = lambda: False
    if mutation == 'inference': t.is_inference_mode_enabled = lambda: True
    with pytest.raises(ValueError): r.action_encoder(Symbol())
    assert not t.nn.functional.calls


def test_authorization_before_any_root_or_runtime_access():
    with pytest.raises(ValueError, match='MAIN authorization'): m.wrap(None, None)


def test_wrap_closure_and_delegation():
    root, t = graph(node()), runtime()
    # File identity only; this is a fake runtime, not a CUDA execution test.
    t.__file__ = str(m.ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py')
    r = m.wrap(graph=root, torch=t, main_authorized=True)
    assert r.low_predictor.weight is root.low_predictor.weight
    assert list(r.named_parameters()) == list(root.named_parameters())
    assert list(r.named_buffers()) == list(root.named_buffers())


def test_bad_attention_scale():
    r, _, _ = facade(attention(scale=1.))
    with pytest.raises(ValueError, match='scale'): r.action_encoder(Symbol())


def test_no_unlisted_delegation_or_data_node_dispatch():
    r, _, _ = facade(node('Identity'))
    with pytest.raises(AttributeError): _ = r.action_encoder.forward
    r, _, _ = facade(node('Conv1d'))
    with pytest.raises(ValueError, match='data-only'): r.action_encoder(Symbol())
    with pytest.raises(ValueError, match='uncertified'): facade(node('ViTModel'))


def test_input_and_arity_rejected():
    r, _, _ = facade(node())
    with pytest.raises(ValueError, match='positional'): r.action_encoder(Symbol(), x=Symbol())
    with pytest.raises(ValueError, match='arity'): r.action_encoder(Symbol(), Symbol())
    x = Symbol(); x.dtype = 'float64'
    with pytest.raises(ValueError, match='float32'): r.action_encoder(x)


def test_source_closure_and_no_graph_severing_or_torch_import():
    paths = m.source_closure()
    assert len(paths) == 6
    source = Path(m.__file__).read_text(); tree = ast.parse(source)
    forbidden = {'inference_mode', 'no_grad', 'detach', 'clone', 'copy_', 'forward'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in forbidden]
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) and
                'torch' in ast.get_source_segment(source, n)]
    assert m.protocol()['CUDA_verified'] is False
