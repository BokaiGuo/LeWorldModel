"""Non-neural coordinate algebra only; no checkpoint, model or simulator."""
import numpy as np
import pytest

from experiments.spectral_reachability import pusht_full_packet_chart_20261027 as c


def finite_difference(function, point, epsilon=1e-6):
    point = np.asarray(point, dtype=float)
    columns = []
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset.flat[index] = epsilon
        columns.append((np.asarray(function(point + offset)).ravel() -
                        np.asarray(function(point - offset)).ravel()) / (2 * epsilon))
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize('kind,width', [('constant', 2), ('full', 10)])
@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_orthonormal_columns_and_horizon_layout(kind, width, horizon):
    basis = c.chart_matrix(kind, horizon=horizon)
    assert basis.shape == (10*horizon, width*horizon)
    np.testing.assert_allclose(basis.T @ basis, np.eye(width*horizon), atol=3e-16, rtol=0)
    if horizon == 1:
        np.testing.assert_array_equal(c.basis(kind), basis)
    parameters = np.arange(horizon*width, dtype=float).reshape(horizon, width) / 10
    full = c.params_to_full(parameters, kind)
    np.testing.assert_allclose(full.ravel(), basis @ parameters.ravel(), atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(full), np.linalg.norm(parameters), rtol=3e-16)


def test_constant_is_tiled_pairs_not_repeated_coordinates_and_returns_fresh_basis():
    expected = np.tile(np.eye(2), (5, 1)) / np.sqrt(5)
    np.testing.assert_array_equal(c.basis('constant'), expected)
    basis = c.basis('constant')
    basis[:] = 999
    np.testing.assert_array_equal(c.basis('constant'), expected)
    coefficients = np.array([.25, -.75])
    np.testing.assert_allclose(c.raw_packet(coefficients, 'constant'),
                               np.tile(np.tanh(coefficients/np.sqrt(5)), (5, 1)))


def test_full_packet_can_change_each_of_five_commands():
    coefficients = np.arange(10, dtype=float) / 20
    np.testing.assert_array_equal(c.raw_packet(coefficients, 'full'),
                                  np.tanh(coefficients).reshape(5, 2))


@pytest.mark.parametrize('kind,width', [('constant', 2), ('full', 10)])
@pytest.mark.parametrize('shape', [(), (3,), (2, 3)])
def test_raw_roundtrip_logit_roundtrip_and_no_mutation(kind, width, shape):
    coefficients = np.linspace(-.8, .9, int(np.prod(shape))*width).reshape(*shape, width)
    original = coefficients.copy()
    logits = c.params_to_full(coefficients, kind)
    np.testing.assert_allclose(c.full_to_params(logits, kind), coefficients, atol=6e-16)
    raw = c.params_to_raw_packets(coefficients, kind)
    assert raw.shape == (*shape, 5, 2)
    np.testing.assert_allclose(c.raw_packets_to_params(raw, kind), coefficients, atol=6e-16)
    assert c.atanh_l2_norm(coefficients, kind) == pytest.approx(np.linalg.norm(np.arctanh(raw)))
    np.testing.assert_array_equal(coefficients, original)


def test_constant_and_embedded_full_have_same_commands_norm_and_projection():
    coefficients = np.array([[.4, -.6], [.8, .2]])
    full = c.constant_to_full(coefficients)
    np.testing.assert_allclose(c.params_to_raw_packets(coefficients, 'constant'),
                               c.params_to_raw_packets(full, 'full'), atol=1e-16)
    assert c.atanh_l2_norm(coefficients, 'constant') == pytest.approx(c.atanh_l2_norm(full, 'full'))
    projected = c.project_params(coefficients, 'constant', radius=.75)
    full_projected = c.project_params(full, 'full', radius=.75)
    np.testing.assert_allclose(c.constant_to_full(projected), full_projected, atol=1e-16)
    assert np.linalg.norm(projected) == pytest.approx(.75)


def test_norm_is_total_over_suffix_not_per_packet_and_mapping_never_silently_clips():
    coefficients = np.array([[.6, 0], [.6, 0]])
    with pytest.raises(ValueError, match='budget exceeded'):
        c.params_to_raw_packets(coefficients, 'constant', radius=.75)
    projected = c.project_params(coefficients, 'constant', radius=.75)
    c.params_to_raw_packets(projected, 'constant', radius=.75)
    np.testing.assert_array_equal(c.project_params(coefficients, 'constant', radius=0), np.zeros((2, 2)))
    copy = c.project_params(coefficients, 'constant', radius=10)
    copy[0, 0] = 999
    assert coefficients[0, 0] == .6


@pytest.mark.parametrize('kind,width', [('constant', 2), ('full', 10)])
def test_projected_boundary_can_be_used_with_the_exact_same_budget(kind, width):
    rng = np.random.default_rng(27)
    for _ in range(100):
        coefficients = rng.normal(size=(3, width))
        radius = np.sqrt(5)*.75
        projected = c.project_params(coefficients, kind, radius=radius)
        assert c.atanh_l2_norm(projected, kind) <= radius
        c.params_to_raw_packets(projected, kind, radius=radius)


def test_new_sqrt5_coordinates_and_damping_recover_old_constant_physical_scale_only():
    legacy = np.array([.3, -.4])
    coefficients = np.sqrt(5)*legacy
    np.testing.assert_allclose(c.raw_packet(coefficients, 'constant'), np.tile(np.tanh(legacy), (5, 1)), atol=1e-16)
    assert .002*np.linalg.norm(coefficients)**2 == pytest.approx(.01*np.linalg.norm(legacy)**2)
    for radius in (.25, .75):
        direction = legacy/np.linalg.norm(legacy)
        raw = c.raw_packet(np.sqrt(5)*radius*direction, 'constant')
        np.testing.assert_allclose(raw, np.tile(np.tanh(radius*direction), (5, 1)), atol=1e-16)


@pytest.mark.parametrize('kind,width', [('constant', 2), ('full', 10)])
def test_raw_mapping_chain_rule_at_nonzero_coordinates(kind, width):
    point = np.linspace(-.9, .6, width)
    analytical = c.raw_mapping_jacobian(point, kind)
    numerical = finite_difference(lambda u: c.raw_packet(u, kind), point)
    assert analytical.shape == (10, width)
    np.testing.assert_allclose(analytical, numerical, atol=9e-11, rtol=2e-9)
    batch = c.raw_mapping_jacobian(np.stack((point, point)), kind)
    assert batch.shape == (2, 10, width)
    np.testing.assert_array_equal(batch[0], analytical)


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_full_jacobian_times_C_equals_direct_constant_and_coordinate_gradient(horizon):
    rng = np.random.default_rng(27+horizon)
    linear = rng.normal(size=(7, 10*horizon))
    point = rng.normal(scale=.3, size=2*horizon)
    matrix = c.chart_matrix('constant', horizon=horizon)
    logits = matrix @ point
    def full_output(z):
        return np.sin(linear @ np.tanh(z))
    full_jacobian = np.cos(linear @ np.tanh(logits))[:, None] * linear * (1-np.tanh(logits)**2)
    constant_jacobian = c.pullback_jacobian(full_jacobian, 'constant', horizon=horizon)
    numerical = finite_difference(lambda u: full_output(matrix @ u), point)
    np.testing.assert_allclose(constant_jacobian, numerical, atol=3e-10, rtol=3e-8)
    target = rng.normal(size=7)
    residual = full_output(logits)-target
    full_gradient = full_jacobian.T @ residual
    pulled = c.pullback_gradient(full_gradient, 'constant', horizon=horizon)
    numerical_gradient = finite_difference(lambda u: .5*np.sum((full_output(matrix@u)-target)**2), point)[0]
    np.testing.assert_allclose(pulled, numerical_gradient, atol=2e-9, rtol=1e-7)
    np.testing.assert_allclose(pulled, constant_jacobian.T @ residual, atol=1e-14)
    np.testing.assert_array_equal(c.pullback_jacobian(full_jacobian, 'full', horizon=horizon), full_jacobian)


@pytest.mark.parametrize('kind,width', [('constant', 2), ('full', 10)])
@pytest.mark.parametrize('horizon', [1, 2])
@pytest.mark.parametrize('with_current', [False, True])
def test_ridge_normal_equations_and_restricted_full_objective(kind, width, horizon, with_current):
    rng = np.random.default_rng(310)
    j_full = rng.normal(size=(11, 10*horizon))
    residual = rng.normal(size=11)  # TARGET minus CURRENT output.
    current = rng.normal(scale=.2, size=width*horizon) if with_current else np.zeros(width*horizon)
    damping = .002
    matrix = c.chart_matrix(kind, horizon=horizon)
    jacobian = j_full @ matrix
    expected = np.linalg.solve(jacobian.T@jacobian+damping*np.eye(width*horizon),
                               jacobian.T@residual-damping*current)
    kwargs = {'current': current} if with_current else {}
    step = c.chart_ridge_step(j_full, residual, kind, damping=damping, horizon=horizon, **kwargs)
    np.testing.assert_allclose(step, expected, atol=2e-11, rtol=1e-10)
    gradient = jacobian.T@(jacobian@step-residual)+damping*(current+step)
    np.testing.assert_allclose(gradient, 0, atol=4e-13)
    def restricted_objective(delta):
        return np.sum((j_full@(matrix@delta)-residual)**2)+damping*np.linalg.norm(matrix@(current+delta))**2
    np.testing.assert_allclose(finite_difference(restricted_objective, step)[0], 0, atol=2e-8)


def test_projecting_full_ridge_optimum_is_not_constant_restricted_solve():
    rng = np.random.default_rng(17)
    jacobian = rng.normal(size=(8, 10))
    residual = rng.normal(size=8)
    unrestricted = c.ridge_step(jacobian, residual, damping=.002)
    restricted = c.chart_ridge_step(jacobian, residual, 'constant', damping=.002)
    assert np.linalg.norm(c.basis('constant').T@unrestricted-restricted) > .01


def test_zero_rank_ridge_and_nonzero_current_penalty():
    jacobian = np.zeros((4, 10))
    np.testing.assert_array_equal(c.ridge_step(jacobian, np.ones(4), damping=.002), np.zeros(10))
    current = np.linspace(-1, 1, 10)
    np.testing.assert_allclose(c.ridge_step(jacobian, np.ones(4), damping=.002, current=current), -current, atol=2e-16)


@pytest.mark.parametrize('bad', [np.zeros(3), np.zeros((0, 2)), [True, False],
                                 [1+1j, 0], [np.nan, 0], [np.inf, 0], ['1', '2']])
def test_bad_coefficients_rejected(bad):
    with pytest.raises(ValueError):
        c.params_to_raw_packets(bad, 'constant')


@pytest.mark.parametrize('bad', [np.zeros(10), np.zeros((2, 5)), np.ones((5, 2)),
                                 np.full((5, 2), -1.), np.full((5, 2), np.nan),
                                 np.zeros((0, 5, 2)), np.zeros((5, 2), bool)])
def test_bad_raw_inverse_rejected(bad):
    with pytest.raises(ValueError):
        c.raw_packets_to_params(bad, 'full')


def test_off_subspace_inverse_and_saturation_rejected():
    logits = np.zeros(10); logits[0] = .1
    with pytest.raises(ValueError, match='outside'):
        c.full_to_params(logits, 'constant')
    with pytest.raises(ValueError, match='outside'):
        c.raw_packets_to_params(np.tanh(logits).reshape(5, 2), 'constant')
    with pytest.raises(ValueError, match='saturated'):
        c.raw_packet(np.full(10, 999.), 'full')
    with pytest.raises(ValueError, match='one coefficient vector'):
        c.raw_packet(np.zeros((2, 2)), 'constant')


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_bad_horizon_rejected(bad):
    with pytest.raises(ValueError):
        c.chart_matrix('constant', horizon=bad)


@pytest.mark.parametrize('bad', [-1, np.inf, np.nan, True, [1]])
def test_bad_norm_radius_rejected(bad):
    with pytest.raises(ValueError):
        c.project_params([0, 0], 'constant', radius=bad)


@pytest.mark.parametrize('bad', [0, -1, np.inf, np.nan, True])
def test_bad_ridge_damping_rejected(bad):
    with pytest.raises(ValueError):
        c.ridge_step(np.eye(2), np.ones(2), damping=bad)


def test_bad_jacobian_residual_chart_and_current_shapes_rejected():
    for call in (
        lambda: c.basis('other'),
        lambda: c.pullback_jacobian(np.zeros((3, 2)), 'constant'),
        lambda: c.ridge_step(np.zeros(3), np.ones(3), damping=.002),
        lambda: c.ridge_step(np.eye(3), np.ones(2), damping=.002),
        lambda: c.ridge_step(np.eye(3), np.ones(3), damping=.002, current=np.zeros((1, 3))),
    ):
        with pytest.raises(ValueError):
            call()
