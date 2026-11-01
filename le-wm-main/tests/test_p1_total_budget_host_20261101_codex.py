import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p1_total_budget_host_20261101_codex as host


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, callback):
        self.hooks.append(callback)
        return SimpleNamespace(remove=lambda: self.hooks.remove(callback))

    def __call__(self):
        for callback in tuple(self.hooks):
            callback(self, ())
        return 1


def model():
    return SimpleNamespace(**{
        name: Module()
        for names in host.total.HOOKS.values()
        for name in names
    })


def reservation(planner="structured"):
    value = dict.fromkeys(host.total.AXES, 0)
    value.update(
        model_load=1,
        source_encode_calls=1,
        source_encode_samples=2,
        action_encode_calls=1,
        predict_calls=1,
        predict_samples=3,
        candidate_evaluations=1,
        replay_calls=1,
        feedback_calls=1,
        physical_dispatch_calls=1,
        physics_steps=1,
    )
    if planner == "structured":
        value.update(
            backward_calls=1,
            vjp_rows=4,
            jacobian_calls=1,
            svd_calls=1,
            solve_calls=1,
        )
    if planner == "cem":
        value["backward_calls"] = 0
        value["vjp_rows"] = 0
        value["jacobian_calls"] = 0
        value["svd_calls"] = 0
        value["solve_calls"] = 0
    return value


def physical_caps():
    value = dict.fromkeys(host.physical.RESOURCES, 0)
    value.update(public_step=1, physics_step=1)
    return value


def protocol(planner="structured", **changes):
    value = host.build_protocol(
        run_id="test-" + planner,
        planner=planner,
        identity=dict.fromkeys(host.total.IDENTITY, "a" * 64),
        reservation=reservation(planner),
        physical_caps=physical_caps(),
        conditional_axes=changes.pop("conditional_axes", ()),
    )
    value.update(changes)
    return value


def exercise(api, planner="structured"):
    neural = api.load_model(model)
    api.source_encode(2, lambda: (neural.encoder(), neural.projector()))
    api.action_encode(neural.action_encoder)
    api.predict(3, lambda: (neural.predictor(), neural.pred_proj()))
    if planner == "structured":
        api.backward(4, lambda: None)
        for kind in ("jacobian", "svd", "solve"):
            api.algorithm(kind, 1, lambda: None)
    api.algorithm("candidate", 1, lambda: None)
    api.algorithm("replay", 1, lambda: None)
    api.algorithm("feedback", 1, lambda: None)
    api.physical_call("public_step", lambda: None)
    api.physical_call("physics_step", lambda: None)
    return "ok"


@pytest.mark.parametrize("planner", ["structured", "cem"])
def test_complete_host_entry_reserves_and_routes_all_required_axes(tmp_path, planner):
    p = protocol(planner)
    result = host.run_host(tmp_path, p, host.total.digest(p),
                           lambda api: exercise(api, planner))
    receipt = result["receipt"]
    assert result["result"] == "ok"
    assert receipt["status"] == "ready_receipt_unverified_cost"
    assert receipt["coverage_complete"]
    assert receipt["charged"] == p["reservation"]
    assert not receipt["coverage_verified"]
    assert receipt["unverified_bypasses"] == list(host.UNVERIFIED_BYPASSES)


def test_budget_blocks_before_callback_and_preserves_failure_receipt(tmp_path):
    p = protocol("cem")
    p["reservation"]["predict_samples"] = 2
    p["source_sha256"][str(Path(host.__file__).resolve())] = hashlib.sha256(
        Path(host.__file__).read_bytes()).hexdigest()
    p_sha = host.total.digest(p)
    calls = []

    def planner(api):
        neural = api.load_model(model)
        api.predict(3, lambda: calls.append(neural.predictor()))

    with pytest.raises(ValueError, match="budget exceeded"):
        host.run_host(tmp_path, p, p_sha, planner)
    assert calls == []
    assert (tmp_path / "total.jsonl").is_file()
    assert (tmp_path / "physical.jsonl").is_file()


def test_missing_model_hook_fails_closed(tmp_path):
    p = protocol("cem")

    def planner(api):
        api.load_model(model)
        api.source_encode(2, lambda: None)

    with pytest.raises(ValueError, match="missing model hook"):
        host.run_host(tmp_path, p, host.total.digest(p), planner)


@pytest.mark.parametrize("planner,key", [
    ("cem", "predict_calls"),
    ("structured", "backward_calls"),
    ("structured", "jacobian_calls"),
    ("structured", "candidate_evaluations"),
])
def test_readiness_rejects_missing_required_reservation(planner, key):
    p = protocol(planner)
    p["reservation"][key] = 0
    with pytest.raises(ValueError, match="omits required"):
        host.validate_protocol(p, host.total.digest(p))


def test_readiness_rejects_physical_total_mismatch_and_core_conditional_axis():
    p = protocol("cem")
    p["physical_caps"]["public_step"] = 2
    with pytest.raises(ValueError, match="reservations disagree"):
        host.validate_protocol(p, host.total.digest(p))
    p = protocol("cem")
    p["conditional_axes"] = ["predict_calls"]
    with pytest.raises(ValueError, match="conditional axes"):
        host.validate_protocol(p, host.total.digest(p))


def test_source_drift_and_unknown_route_fail_before_execution(tmp_path, monkeypatch):
    p = protocol("cem")
    source = next(iter(p["source_sha256"]))
    p["source_sha256"][source] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        host.Host(tmp_path, p, host.total.digest(p))

    p = protocol("cem")
    api = host.Host(tmp_path, p, host.total.digest(p))
    api.load_model(model)
    with pytest.raises(ValueError, match="unmapped planner"):
        api.algorithm("custom_kernel", 1, lambda: monkeypatch)
    receipt = api.close()
    assert receipt["status"] == "failed_or_incomplete"


def test_unobserved_nonconditional_axis_is_incomplete(tmp_path):
    p = protocol("cem")
    p["reservation"]["render_calls"] = 1
    p["physical_caps"]["explicit_render"] = 1
    result = host.run_host(tmp_path, p, host.total.digest(p),
                           lambda api: exercise(api, "cem"))
    assert result["receipt"]["status"] == "failed_or_incomplete"
    assert not result["receipt"]["coverage_complete"]


def test_explicit_conditional_axis_allows_early_stop(tmp_path):
    caps = physical_caps()
    caps["explicit_render"] = 1
    value = reservation("cem")
    value["render_calls"] = 1
    p = host.build_protocol(
        run_id="test-cem-conditional",
        planner="cem",
        identity=dict.fromkeys(host.total.IDENTITY, "a" * 64),
        reservation=value,
        physical_caps=caps,
        conditional_axes=("render_calls",),
    )
    result = host.run_host(tmp_path, p, host.total.digest(p),
                           lambda api: exercise(api, "cem"))
    assert result["receipt"]["status"] == "ready_receipt_unverified_cost"
