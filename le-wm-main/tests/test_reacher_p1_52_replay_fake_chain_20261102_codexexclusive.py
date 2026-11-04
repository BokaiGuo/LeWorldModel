"""No real physics, torch, or neural inference: exercise the complete replay wiring."""
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_d2p_context as ctx
from experiments.spectral_reachability import reacher_p1_52_replay_contract_20261102_codexexclusive as contract
from experiments.spectral_reachability import reacher_p1_52_replay_consumer_20261102_codexexclusive as consumer
from experiments.spectral_reachability import reacher_p1_52_replay_host_20261102_codexexclusive as runner
from experiments.spectral_reachability import reacher_p1_52_replay_roster_20261102_codexexclusive as roster
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved


def image(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


class Array:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def __getitem__(self, key):
        return Array(self.data[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def tolist(self):
        return self.data.tolist()

    @property
    def shape(self):
        return self.data.shape


class FakeEnv:
    def __init__(self, stop_at=None):
        self.qpos = np.zeros(1)
        self.unwrapped = SimpleNamespace(env=SimpleNamespace(
            physics=SimpleNamespace(data=SimpleNamespace(qpos=self.qpos)),
            task=SimpleNamespace(target_qpos=np.zeros(1))))
        self.calls = 0
        self.stop_at = stop_at

    def step(self, action):
        self.calls += 1
        self.qpos[0] += 1
        return None, 0, self.calls == self.stop_at, False, {}


def make_fake_setup():
    def collect(env, *, warmup_packets, **kwargs):
        for action in warmup_packets.reshape(15, 2):
            env.step(action)
        env.qpos[0] = 0
        return {"frames": [image(0), image(1), image(2)], "root_frame": image(3), "goal": image(9),
                "history_action_blocks": [a.reshape(-1) for a in warmup_packets],
                "execution_snapshot": {"distance": 0, "frame": 3}}

    def restore(env, *, seed, snapshot):
        env.qpos[0] = snapshot["distance"]
        return image(snapshot["frame"]), image(9)

    def advance(env, *, context, raw_action_block, **kwargs):
        for action in raw_action_block:
            result = env.step(action)
            if result[2] or result[3]:
                raise RuntimeError("synthetic early stop")
        frame = int(context["root_frame"][0, 0, 0]) + 1
        return {"frames": context["frames"][1:] + [context["root_frame"]], "root_frame": image(frame),
                "goal": context["goal"],
                "history_action_blocks": context["history_action_blocks"][1:] + [raw_action_block.reshape(-1)],
                "execution_snapshot": {"distance": float(env.qpos[0]), "frame": frame}}

    return SimpleNamespace(collect_executed_context=collect, restore_execution_snapshot=restore,
                           advance_executed_context=advance, SOURCE_WARMUP_POLICY="fake")


def make_p0():
    return SimpleNamespace(
        _verify_execution_identity=lambda *args: None,
        _qpos_distance=lambda q, target: float(q[0]),
        _prepare=lambda *args: (None, None, Array(np.zeros((1, 3, 10)))),
        _planning_coordinate=lambda model, latent, role: latent,
        _goal_latent=lambda *args: Array([0, 0]),
        _latent_context=lambda *args: Array([[[1.5, 2.5]] * 3]),
    )


def run_fake(monkeypatch, stop_at=None, index=0):
    rows = json.loads(Path(contract.ROSTER).read_text())["rows"]
    plan = json.loads((contract.ROOT / "results/reacher_response_svd_cp5_PLAN_20261102_v1.json").read_text())
    row = rows[index]
    root = next(r for r in plan["roots"] if r["seed"] == row["seed"])
    setup = make_fake_setup()
    actions = np.asarray(row["actions"], np.float32)
    warmup = np.asarray(root["warmup_packets"], np.float32)
    cp5_value = host.capture([image(1), image(2), image(3)], image(4), image(9),
                             [warmup[1].reshape(-1), warmup[2].reshape(-1), actions[:5].reshape(-1)])
    for paired_index in (index % 26, index % 26 + 26):
        rows[paired_index]["cp5_context_sha256"] = roster.digest(cp5_value)
        rows[paired_index]["cp5_rgb_sha256"] = ctx.rgb_digest(image(4))
        rows[paired_index]["goal_rgb_sha256"] = ctx.rgb_digest(image(9))
    row["original_physical"]["after_first_chunk_qpos_distance"] = 5.0
    row["original_physical"]["final_qpos_distance"] = 15.0
    monkeypatch.setattr(host.bridge, "check_context", lambda context, root: None)
    fake_torch = SimpleNamespace(no_grad=nullcontext, device=lambda name: name,
                                 as_tensor=lambda data, device: Array(data),
                                 cat=lambda seq, dim: Array(np.concatenate([x.data for x in seq], axis=dim)))
    seen_tokens = []
    def action_encoder(value):
        seen_tokens.append(value.data.copy())
        return value
    model = SimpleNamespace(action_encoder=action_encoder,
                            predict=lambda emb, act: Array(np.tile([1, 2], (1, 3, 1))))
    events = []
    env = FakeEnv(stop_at)
    runner.replay_arm(row, root, env, setup, make_p0(), None, model, np.zeros(10), np.ones(10),
                      Array(np.full(10, -2)), Array(np.full(10, 2)),
                      SimpleNamespace(emit=events.append), fake_torch)
    journal = [{"kind": "run_started", "plan_sha256": "plan"}] + events + [
        {"kind": "run_error", "error": "synthetic partial panel"}]
    return rows, plan["roots"], journal, seen_tokens


def test_fake_full_52_panel_and_saved_consumer(monkeypatch):
    full_rows = json.loads(Path(contract.ROSTER).read_text())["rows"]
    full_events = [{"kind": "run_started", "plan_sha256": "plan"}]
    for index in range(52):
        rows, roots, events, tokens = run_fake(monkeypatch, index=index)
        assert events[-2]["kind"] == "arm_complete"
        full_rows[index] = rows[index]
        full_events.extend(events[1:-1])
    full_events.append({"kind": "run_finished", "model_before": "model", "model_after": "model"})
    result = consumer.consume_events(full_events, full_rows, "plan", "model", roots, np.zeros(10), np.ones(10))
    assert result["status"] == "complete"
    assert result["completed"] == 52
    assert all(r["physical_distance_rank"] == 7 for r in result["rows"])


def test_fake_full_arm_and_saved_consumer(monkeypatch):
    rows, roots, events, tokens = run_fake(monkeypatch)
    assert events[-2]["kind"] == "arm_complete"
    assert len(tokens) == 2
    np.testing.assert_array_equal(tokens[0][0, -1], np.asarray(rows[0]["actions"][5:10]).reshape(10))
    np.testing.assert_array_equal(tokens[1][0, -1], np.asarray(rows[0]["actions"][10:15]).reshape(10))
    result = consumer.consume_events(events, rows, "plan", "model", roots, np.zeros(10), np.ones(10))
    assert result["completed"] == 1
    assert result["rows"][0]["prediction_error"] == pytest.approx(2 ** -.5)
    assert result["rows"][0]["physical_distance"] == 15
    assert result["rows"][0]["terminal_rgb_sha256"] == ctx.rgb_digest(image(6))


def test_saved_hash_chain_detects_corruption(monkeypatch, tmp_path):
    rows, roots, events, _ = run_fake(monkeypatch)
    path = tmp_path / "events.jsonl"
    journal = host.durable.Journal(path)
    for event in events:
        journal.emit(event)
    journal.close()
    decoded = saved.verify_journal(path)
    assert consumer.consume_events(decoded, rows, "plan", "model", roots,
                                   np.zeros(10), np.ones(10))["completed"] == 1
    lines = path.read_text().splitlines()
    changed = json.loads(lines[1])
    changed["payload"]["arm"] = "tampered"
    lines[1] = json.dumps(changed)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        saved.verify_journal(path)


@pytest.mark.parametrize("stop_at", [16, 20, 21, 30])
def test_early_stop_is_retained_and_censored(monkeypatch, stop_at):
    rows, roots, events, _ = run_fake(monkeypatch, stop_at)
    assert events[-2]["kind"] == "arm_failed"
    result = consumer.consume_events(events, rows, "plan", "model", roots, np.zeros(10), np.ones(10))
    assert result["rows"][0]["status"] == "failed"
    assert result["rows"][0]["physical_distance_rank"] is None


@pytest.mark.parametrize("tamper", ["action", "token", "cp5", "terminal_rgb", "history", "metric", "chronology"])
def test_consumer_rejects_completed_arm_tampering(monkeypatch, tamper):
    rows, roots, events, _ = run_fake(monkeypatch)
    events = copy.deepcopy(events)
    if tamper == "action":
        next(e for e in events if e["kind"] == "step_intent")["action"][0] += .1
    elif tamper == "token":
        next(e for e in events if e["kind"] == "model_intent" and e["phase"] == "prediction")["suffix_normalized"][0][0] += .1
    elif tamper == "cp5":
        next(e for e in events if e["kind"] == "cp5_verified")["context"]["history"][0][0] += .1
    elif tamper == "terminal_rgb":
        next(e for e in events if e["kind"] == "physical_returned")["context"]["frames"][-1]["sha256"] = "bad"
    elif tamper == "history":
        next(e for e in events if e["kind"] == "physical_returned")["context"]["history"][-1][0] += .1
    elif tamper == "metric":
        next(e for e in events if e["kind"] == "arm_complete")["metrics"]["prediction_error"] = 3
    else:
        a = next(i for i, e in enumerate(events) if e["kind"] == "cp5_verified")
        b = next(i for i, e in enumerate(events) if e["kind"] == "model_returned")
        events[a], events[b] = events[b], events[a]
    with pytest.raises((ValueError, KeyError)):
        consumer.consume_events(events, rows, "plan", "model", roots, np.zeros(10), np.ones(10))
