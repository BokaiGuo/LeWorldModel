import numpy as np
import torch

from experiments.spectral_reachability import run_pusht_d2p_p1_physical_calibration as physical


def test_pose_label_and_replay_contract_are_state_and_rgb_sensitive():
    state = np.zeros(7); goal = np.zeros(7); state[2:4] = (3.0, 4.0); state[4] = np.pi / 2
    label = physical._pose_label(state, goal)
    assert label["xy_distance"] == 5.0
    first = {"warmup_rgb_sha256": "a", "goal_rgb_sha256": "b", "successor_rgb_sha256": "c", "final_rgb_sha256": "d", "successor_state": [1.0], "final_state": [2.0]}
    assert physical._replay_matches(first, dict(first))
    changed = dict(first); changed["final_rgb_sha256"] = "different"
    assert not physical._replay_matches(first, changed)


def test_actual_runner_executes25_commands_and_keeps_the_live_decision_frame(monkeypatch):
    from tests.test_pusht_d2p_packet_contract import RecordingEnv, ROOT, audit
    class TimedEnv(RecordingEnv):
        def render(self):
            return np.full((224, 224, 3), len(self.calls), np.uint8)
    env = TimedEnv()
    captured = {}
    def suffix(model, frames, actions, goal_rgb, **kwargs):
        captured["frames"] = [int(f[0, 0, 0]) for f in frames]
        captured["actions"] = actions.copy()
        return np.zeros((1, 2), np.float32), {"projection_sha256": "a"*64}
    monkeypatch.setattr(physical, "_observed_suffix", suffix)
    monkeypatch.setattr(physical, "_predict_prefix", lambda *args, **kwargs: {})
    prefix = np.array([.25, -.125], np.float32)
    physical._execute_branch(env, {"state": ROOT, "goal": audit.DIAGNOSTIC_GOAL}, prefix,
        model=object(), transform=None, device=torch.device("cpu"), damping=.01, sketch_dims=8, projection_seed=1)
    assert len(env.calls) == 25
    np.testing.assert_array_equal(env.calls[15:20], np.tile(prefix, (5, 1)))
    assert captured["frames"] == [10, 15, 20]
    np.testing.assert_array_equal(captured["actions"], [[0., 0.], [0., 0.], prefix])


def test_old_physical_calibration_cannot_open_corrected_outcome(tmp_path):
    import json
    import pytest
    p = tmp_path/"old.json"
    p.write_text(json.dumps({"schema": "pusht_d2p_p1_physical_calibration.v1", "status": "pass", "gates": {"x": True}}))
    with pytest.raises(ValueError): physical._load_calibration(p)


def test_metrics_cannot_substitute_for_each_other():
    def branch(pose, rgb):
        return {"after_prefix_label": {"pose_cost": pose},
                "suffix_recompile": {"observed_successor_goal_cost": rgb}}
    row = {"model_eligible": True, "selected": branch(2., .1), "raw_zero_control": branch(1., .2)}
    gates = physical._prefix_metric_gates([row])
    assert gates["all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero"]
    assert not gates["all_selected_prefixes_actual_rgb_no_more_goal_greedy_than_zero"]
    row["selected"] = branch(.5, .3)
    gates = physical._prefix_metric_gates([row])
    assert not gates["all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero"]
    assert gates["all_selected_prefixes_actual_rgb_no_more_goal_greedy_than_zero"]
    assert not any(physical._prefix_metric_gates([]).values())


def test_prerun_seal_preserves_upstream_and_detects_mutation(tmp_path):
    import pytest
    upstream = tmp_path / "upstream.json"
    upstream.write_text("{}")
    output = tmp_path / "run"
    seal = physical._seal_run(output, upstream, None)
    assert physical._seal_unchanged(seal)
    assert (output / "prerun_seal.json").exists()
    with pytest.raises(FileExistsError):
        physical._seal_run(output, upstream, None)
    upstream.write_text('{"changed": true}')
    assert not physical._seal_unchanged(seal)


def test_v4_preflight_and_v2_physical_receipts_are_rejected(tmp_path):
    import json
    import pytest
    for schema, loader in (("pusht_d2p_p1_reconfiguration_preflight.v4", physical._load_preflight),
                           ("pusht_d2p_p1_physical_calibration.v2", physical._load_calibration)):
        path = tmp_path / (schema + ".json")
        path.write_text(json.dumps({"schema": schema, "status": "pass", "gates": {"x": True}}))
        with pytest.raises(ValueError):
            loader(path)


def test_current_schema_empty_or_missing_gates_cannot_open_outcome(tmp_path):
    import json
    import pytest
    path = tmp_path/"calibration.json"
    report = {"schema": physical.SCHEMA, "status": "pass", "gates": {},
        "protocol": {"cohort": "calibration", "packet_contract": physical.p0.packets.contract_spec(),
                     "root_context": physical.p1.ROOT_CONTEXT, "outcome_roots_opened": False},
        "rows": [{"root_sha256": "a"*64}]}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): physical._load_calibration(path)
    report["gates"] = {k: True for k in physical.REQUIRED_GATES}
    path.write_text(json.dumps(report))
    assert physical._load_calibration(path) == {"a"*64}
    report["gates"].pop("same_root_shared_suffix_projection")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): physical._load_calibration(path)


def test_root_identity_guard_refuses_empty_duplicate_and_nonsha_roots():
    import pytest
    for rows in ([], [{"root_sha256": "a"*64}]*2, [{"root_sha256": "x"*64}], [{"root_sha256": None}]):
        with pytest.raises(ValueError): physical._root_identities(rows)


def test_p0_and_model_pass_require_every_boolean_gate():
    import pytest
    for schema, required in ((physical.p0.SCHEMA, physical.p1.P0_REQUIRED_GATES),
                             (physical.p1.SCHEMA, physical.p1.REQUIRED_GATES)):
        for gates in ({}, {k: 1 for k in required}, {k: True for k in sorted(required)[1:]}):
            with pytest.raises(ValueError):
                physical.p1._require_complete_pass({"schema": schema, "status": "pass", "gates": gates}, schema, required)
        physical.p1._require_complete_pass({"schema": schema, "status": "pass", "gates": {k: True for k in required}}, schema, required)
