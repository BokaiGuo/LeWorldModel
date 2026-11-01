"""Metadata-only execution of bridge definitions: NO torch/model/CUDA imports.

AST extraction uses the production class/helper bodies, replacing torch only at
the boundary with NumPy metadata doubles. This checks orchestration, not CUDA
kernel fidelity, real device transfer, neural objectives or physical efficacy.
Export failures must remain failed, never completed candidate costs.
"""
import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability.d2p_sampling_baselines import (
    CostEvaluationError, mppi, random_shooting,
)


@dataclass(frozen=True)
class Device:
    type: str = 'cuda'  # metadata label only, never a real device

    def __str__(self):
        return 'metadata-only:cuda'


class Tensor:
    def __init__(self, value, *, dtype=None, device=None, fail_export=False):
        with np.errstate(over='ignore'):
            self.data = np.asarray(value, dtype=dtype)
        self.device = device or Device()
        self.dtype = self.data.dtype
        self.requires_grad = False
        self.fail_export = fail_export

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __lt__(self, other):
        return self.data < other.data

    def is_floating_point(self):
        return self.dtype.kind == 'f'

    def detach(self):
        return self

    def cpu(self):
        if self.fail_export:
            raise RuntimeError('injected host export failure; no CUDA executed')
        return self

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data.tolist()


class ModelMetadata:
    training = False

    def modules(self):
        return [self]

    def parameters(self):
        return []


@pytest.fixture
def bridge_type():
    path = Path(__file__).resolve().parents[1] / 'experiments/spectral_reachability/d2p_cuda_sampling_cost.py'
    parsed = ast.parse(path.read_text())
    # Do not import production imports: P0 pulls neural/runtime dependencies.
    definitions = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    assert any(isinstance(n, ast.ClassDef) and n.name == 'CudaSamplingCost' for n in definitions)
    fake_torch = SimpleNamespace(
        Tensor=Tensor,
        as_tensor=lambda value, *, device, dtype: Tensor(value, device=device, dtype=dtype),
        isfinite=lambda value: np.isfinite(value.data),
        all=np.all, no_grad=nullcontext,
    )
    namespace = {'np': np, 'torch': fake_torch, '_cost_fn': None}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CudaSamplingCost']


def make_bridge(bridge_type, cost_fn):
    return bridge_type(
        ModelMetadata(), Tensor(np.zeros((1, 3, 2))), Tensor(np.zeros((1, 2))),
        Tensor(np.zeros((1, 3, 2)), dtype=np.float32),
        action_lower=Tensor([-2., -3.]), action_upper=Tensor([2., 3.]),
        future_horizon=2, cost_fn=cost_fn,
    )


def solve(solver, bridge, **overrides):
    kwargs = dict(initial=np.zeros((2, 2)), lower=-4., upper=4., samples=7, seed=9)
    if solver is mppi:
        kwargs.update(iterations=2, temperature=.5, noise_scale=.4)
    kwargs.update(overrides)
    return solver(bridge, **kwargs)


@pytest.mark.parametrize('solver', [random_shooting, mppi])
def test_success_counts_and_parameter_domains_are_separate(bridge_type, solver):
    seen = []

    def cost(model, pixels, goal, history, values, **bounds):
        seen.append(values.data.copy())
        return Tensor(np.square(values.data).sum(axis=(1, 2)))

    bridge = make_bridge(bridge_type, cost)
    _, kernel = solve(solver, bridge)
    metadata = bridge.receipt()
    assert kernel['schema'] == 'd2p_sampling_baseline_receipt.v2'
    assert kernel['completed'] is True
    expected = 8 if solver is random_shooting else 17
    assert kernel['cost_evaluations'] == metadata['completed_candidate_cost_samples'] == expected
    assert kernel['cost_calls'] == metadata['completed_candidate_cost_calls'] == len(seen)
    assert metadata['candidate_cost_samples'] == kernel['cost_evaluations_attempted']
    assert metadata['failed_candidate_cost_calls'] == kernel['cost_calls_failed'] == 0
    assert kernel['parameter_domain']['shape'] == [2, 2]
    assert kernel['parameter_domain']['lower'] == [[-4., -4.], [-4., -4.]]
    assert metadata['normalized_action_lower'] == [-2., -3.]
    assert metadata['normalized_action_upper'] == [2., 3.]
    assert 'no_action_transform' in kernel['parameter_domain']['coordinate']
    assert 'tanh_mapped' in metadata['coordinate']
    assert 'not predict' in metadata['scope']
    if solver is mppi:
        assert kernel['contract']['importance_correction'] == 'none'
    assert kernel['contract']['control_cost'] == 'caller_only'


@pytest.mark.parametrize('kind', ['nan', 'shape', 'complex', 'bool', 'exception'])
@pytest.mark.parametrize('solver', [random_shooting, mppi])
def test_delegated_failure_never_claims_kernel_completed(bridge_type, solver, kind):
    calls = 0

    def cost(model, pixels, goal, history, values, **bounds):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Tensor(np.zeros(len(values.data)))
        if kind == 'exception':
            raise RuntimeError('injected callable failure')
        n = len(values.data)
        return Tensor({'nan': lambda: np.full(n, np.nan),
                       'shape': lambda: np.zeros((n, 1)),
                       'complex': lambda: np.full(n, 1+2j),
                       'bool': lambda: np.ones(n, dtype=bool)}[kind]())

    bridge = make_bridge(bridge_type, cost)
    with pytest.raises(CostEvaluationError) as failure:
        solve(solver, bridge)
    kernel, metadata = failure.value.receipt, bridge.receipt()
    assert kernel['completed'] is False
    assert 'action' not in kernel
    assert kernel['cost_calls'] == metadata['completed_candidate_cost_calls'] == 1
    assert kernel['cost_evaluations'] == metadata['completed_candidate_cost_samples'] == 1
    assert kernel['cost_calls_failed'] == metadata['failed_candidate_cost_calls'] == 1
    assert kernel['cost_calls_attempted'] == metadata['candidate_cost_calls'] == 2
    assert kernel['cost_evaluations_attempted'] == metadata['candidate_cost_samples'] == 8
    assert kernel['cost_evaluations_failed'] == 7


def test_parameter_cast_overflow_is_attempted_but_not_submitted(bridge_type):
    def forbidden(*args, **kwargs):
        pytest.fail('overflowed proposals must not reach cost')

    bridge = make_bridge(bridge_type, forbidden)
    with pytest.raises(CostEvaluationError, match='conversion overflow') as failure:
        solve(random_shooting, bridge, initial=np.full((2, 2), 1e100),
              lower=-1e101, upper=1e101)
    kernel, metadata = failure.value.receipt, bridge.receipt()
    assert kernel['completed'] is False
    assert kernel['cost_calls_attempted'] == kernel['cost_calls_failed'] == 1
    assert kernel['cost_calls'] == kernel['cost_evaluations'] == 0
    assert metadata['candidate_cost_calls'] == metadata['completed_candidate_cost_calls'] == 0
    assert metadata['failed_candidate_cost_calls'] == 0  # submission never occurred
    assert 'after CUDA conversion' in metadata['counter_semantics']


def test_cost_domain_overflow_fails_after_successful_parameter_conversion(bridge_type):
    def overflow_cost(model, pixels, goal, history, values, **bounds):
        with np.errstate(over='ignore'):
            # Finite float64 source cost becomes inf in delegated float32 dtype.
            return Tensor(np.full(len(values.data), 1e100), dtype=np.float32)

    bridge = make_bridge(bridge_type, overflow_cost)
    with pytest.raises(CostEvaluationError) as failure:
        solve(random_shooting, bridge)
    kernel, metadata = failure.value.receipt, bridge.receipt()
    assert kernel['completed'] is False
    assert kernel['cost_calls'] == metadata['completed_candidate_cost_calls'] == 0
    assert metadata['candidate_cost_calls'] == metadata['failed_candidate_cost_calls'] == 1


def test_export_failure_cannot_increment_completed_counter(bridge_type):
    """Valid cost tensor with failed export must agree with kernel failure."""
    def valid_cost_with_failed_export(model, pixels, goal, history, values, **bounds):
        return Tensor(np.zeros(len(values.data)), fail_export=True)

    bridge = make_bridge(bridge_type, valid_cost_with_failed_export)
    with pytest.raises(CostEvaluationError, match='host export failure') as failure:
        solve(random_shooting, bridge)
    kernel, metadata = failure.value.receipt, bridge.receipt()
    assert kernel['completed'] is False
    assert kernel['cost_calls'] == 0
    assert kernel['cost_calls_failed'] == 1
    assert metadata['completed_candidate_cost_calls'] == 0
    assert metadata['completed_candidate_cost_samples'] == 0
    assert metadata['failed_candidate_cost_calls'] == 1
