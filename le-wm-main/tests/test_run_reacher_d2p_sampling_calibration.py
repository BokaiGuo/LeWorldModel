"""Non-neural orchestration/NumPy fixtures only, not real calibration evidence."""
import ast
import inspect
import json

import numpy as np
import pytest

from experiments.spectral_reachability import run_reacher_d2p_sampling_calibration as runner
from experiments.spectral_reachability import d2p_sampling_baselines as kernels


@pytest.mark.parametrize("method,expected", [("clipped_gaussian_mppi_heuristic", 133), ("bestofn_random", 33)])
def test_real_numpy_kernel_interface_parameter_domain_and_count(method, expected):
    seen = []
    def cost(params):
        assert params.shape[1:] == (2, 10)
        assert np.all(np.abs(params) <= 3.)
        seen.extend(params.copy())
        return np.sum(params**2, axis=(1, 2))
    params, receipt = runner.sample_parameters(method, cost, action_width=10, seed=20261040, kernels=kernels)
    assert params.shape == (2, 10)
    assert receipt["cost_evaluations"] == len(seen) == expected
    assert np.max(np.abs(seen)) > 1.  # Definitely not a [-1,1] raw action box.
    assert receipt["official_reproduction"] is False
    if method == "clipped_gaussian_mppi_heuristic":
        assert receipt["algorithm"] == method
        assert receipt["contract"]["importance_correction"] == "none"


def test_sync_wraps_entire_planner():
    events = []
    times = iter([4., 7.])
    result, seconds = runner.timed_call(lambda: events.append("plan") or "result",
        synchronize=lambda: events.append("sync"), clock=lambda: next(times))
    assert events == ["sync", "plan", "sync"]
    assert result == "result" and seconds == 3.


def test_exclusive_json_and_nonfinite_rejection(tmp_path):
    path = tmp_path / "receipt.json"
    runner._write_new(path, {"fixture": True})
    with pytest.raises(FileExistsError):
        runner._write_new(path, {})
    assert json.loads(path.read_text()) == {"fixture": True}
    with pytest.raises(ValueError):
        runner._write_new(tmp_path / "bad.json", {"cost": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_fixed_calibration_contract():
    assert runner.SEEDS == (20261040, 20261041)
    assert (runner.HORIZON, runner.HISTORY, runner.ACTION_BLOCK) == (2, 3, 5)
    assert runner.TARGET_DELTA == (.65, -.65)
    with pytest.raises(ValueError):
        runner.sample_parameters("unknown", None, action_width=10, seed=1, kernels=kernels)


def test_runtime_imports_deferred_and_no_hdf5_or_download():
    tree = ast.parse(inspect.getsource(runner))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    text = " ".join(ast.unparse(node) for node in imports)
    assert "torch" not in text and "gym" not in text
    source = inspect.getsource(runner.run)
    assert source.index("create_seal(files=") < source.index("require_cuda_allocation(\"cuda\")")
    assert "load_history_packet_pool(" not in source
    assert "h5py" not in source and "download(" not in source
    assert '"P1_admission": False' in source


@pytest.mark.parametrize("method,candidates,calls,predict_samples", [
    ("clipped_gaussian_mppi_heuristic", 133, 9, 266),
    ("bestofn_random", 33, 2, 66),
])
def test_two_step_predict_accounting_fixture(method, candidates, calls, predict_samples):
    # Accounting simulation ONLY: no model, CUDA bridge, or neural computation.
    batches = []
    def cost(params):
        batches.append(len(params))
        return np.sum((params - .2)**2, axis=(1, 2))
    _, receipt = runner.sample_parameters(method, cost, action_width=10,
                                         seed=20261040, kernels=kernels)
    assert receipt["cost_calls"] == len(batches) == calls
    assert receipt["cost_evaluations"] == sum(batches) == candidates
    assert sum(batch * runner.HORIZON for batch in batches) == predict_samples
    if method == "clipped_gaussian_mppi_heuristic":
        assert batches == [1, 32, 1, 32, 1, 32, 1, 32, 1]
    else:
        assert batches == [1, 32]


def test_kernel_receipt_preserved_without_official_mppi_claim():
    params, receipt = runner.sample_parameters("clipped_gaussian_mppi_heuristic",
        lambda x: np.sum(x**2, axis=(1, 2)), action_width=10, seed=20261041, kernels=kernels)
    assert receipt["algorithm"] == "clipped_gaussian_mppi_heuristic"
    assert receipt["contract"]["importance_correction"] == "none"
    assert receipt["contract"]["control_cost"] == "caller_only"
    assert receipt["configuration"]["samples"] == 32
    assert receipt["configuration"]["iterations"] == 4
    assert receipt["official_reproduction"] is False
    assert np.isfinite(params).all()


def test_timer_does_not_return_success_when_planner_fails():
    events = []
    def failed():
        events.append("plan")
        raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError, match="fixture failure"):
        runner.timed_call(failed, synchronize=lambda: events.append("sync"), clock=lambda: 1.)
    assert events == ["sync", "plan"]


def test_cem_configuration_expected_predict_samples_static_only():
    # Check the checked-in call without importing/running P0 or torch.
    tree = ast.parse(inspect.getsource(runner.run))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "_cem"]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert ast.literal_eval(keywords["samples"]) == 32
    assert ast.literal_eval(keywords["steps"]) == 4
    assert ast.literal_eval(keywords["elite"]) == 4
    assert 32 * 4 * runner.HORIZON == 256


def fixture_inputs(tmp_path):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "weights.pt").write_bytes(b"fixture-not-a-model")
    (policy / "config.json").write_text("{}")
    adapter = tmp_path / "adapter.json"
    adapter.write_text("{}")
    return policy, adapter


def test_run_existing_output_fails_before_seal_or_runtime(tmp_path, monkeypatch):
    from experiments.spectral_reachability import d2p_prerun_source_seal as seal
    policy, adapter = fixture_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_bytes(b"preserve")
    def forbidden(**kwargs):
        pytest.fail("existing output must fail before source sealing or inference")
    monkeypatch.setattr(seal, "create_seal", forbidden)
    with pytest.raises(FileExistsError):
        runner.run(policy=policy, adapter_artifact=adapter, output=output)
    assert sentinel.read_bytes() == b"preserve"
    assert list(output.iterdir()) == [sentinel]


def test_prerun_contract_and_seal_failure_preserves_directory_without_runtime(tmp_path, monkeypatch):
    from experiments.spectral_reachability import d2p_prerun_source_seal as seal
    policy, adapter = fixture_inputs(tmp_path)
    output = tmp_path / "new"
    captured = {}
    class SealStopped(RuntimeError):
        pass
    def stopped(**kwargs):
        captured.update(kwargs)
        raise SealStopped("stop before runtime imports")
    monkeypatch.setattr(seal, "create_seal", stopped)
    with pytest.raises(SealStopped):
        runner.run(policy=policy, adapter_artifact=adapter, output=output)
    assert output.is_dir() and list(output.iterdir()) == []
    protocol = json.loads(captured["args_json"])
    assert protocol["sampling_parameter_box"] == [-3., 3.]
    assert protocol["sampling_coordinate"] == "pre_tanh_parameters_not_raw_action_box"
    assert protocol["cem_parameter_domain"] == "P0_unbounded_Gaussian_pre_tanh_parameters"
    assert protocol["methods"] == ["clipped_gaussian_mppi_heuristic", "bestofn_random", "cem"]
    assert protocol["seeds"] == [20261040, 20261041]
    assert protocol["split"] == "known_calibration_only"
    for key in ("P1_admission", "efficacy_claim", "fair_budget_claim"):
        assert protocol[key] is False
    assert "negative results cannot exclude MPPI" in protocol["mppi_claim_boundary"]
    assert len(protocol["inputs"]) == 3
    assert all(len(digest) == 64 for digest in protocol["inputs"].values())
    names = {p.name for p in captured["files"]}
    assert {"d2p_cuda_sampling_cost.py", "d2p_sampling_baselines.py",
            "d2p_model_call_instrumentation.py", "run_reacher_d2p_sampling_calibration.py"} <= names
    assert captured["output_dir"] == output / "prerun_source_seal"
    with pytest.raises(FileExistsError):
        runner.run(policy=policy, adapter_artifact=adapter, output=output)


def test_synchronized_timer_event_order_includes_clock_boundaries():
    events = []
    def clock():
        events.append("clock")
        return float(len(events))
    runner.timed_call(lambda: events.append("plan"),
                      synchronize=lambda: events.append("sync"), clock=clock)
    assert events == ["sync", "clock", "plan", "sync", "clock"]


def test_sampling_orchestration_passes_parameter_box_explicitly():
    from types import SimpleNamespace
    captured = {}
    def fake(cost, **kwargs):
        captured.update(kwargs)
        return kwargs["initial"], {"fixture": True}
    params, receipt = runner.sample_parameters("clipped_gaussian_mppi_heuristic", None,
        action_width=10, seed=20261040, kernels=SimpleNamespace(mppi=fake))
    assert captured["lower"] == -3. and captured["upper"] == 3.
    assert captured["samples"] == 32 and captured["iterations"] == 4
    assert captured["temperature"] == 1. and captured["noise_scale"] == 1.
    np.testing.assert_array_equal(params, np.zeros((2, 10)))
    assert receipt == {"fixture": True}
