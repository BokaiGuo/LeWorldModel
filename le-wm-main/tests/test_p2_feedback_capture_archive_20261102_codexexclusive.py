from contextlib import nullcontext
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p2_cp5_native_session_20261102_codexexclusive as native
from experiments.spectral_reachability import p2_cp5_launcher_preflight_20261102_codexexclusive as lm
from experiments.spectral_reachability import p2_feedback_capture_archive_20261102_codexexclusive as bridge
from experiments.spectral_reachability import p2_feedback_saved_consumer_20261102_codexexclusive as consumer
from experiments.spectral_reachability import p2_prefix_feedback_common_budget_20261102_codexexclusive as kernel


def digest(x):
    return hashlib.sha256(x).hexdigest()


class FakePhysical:
    class moving:
        exact = staticmethod(np.array_equal)
        warmup_commands = staticmethod(lambda: np.zeros((10, 2), dtype="<f4"))
    class source:
        vector = staticmethod(np.asarray)
        root_hash = staticmethod(lambda s, g: digest(s.tobytes() + g.tobytes()))
    class paired:
        check_return = staticmethod(lambda *a: False)
    class native:
        class allmode:
            class NativeStop(RuntimeError):
                pass
        metrics = staticmethod(lambda *a: {"native_success": False, "native7_reward": 0., "native": {}})
    @staticmethod
    def place(*args):
        return {"state": np.array([0.])}, np.array([1.])
    bind_goal = staticmethod(lambda *args: None)
    invariant = staticmethod(lambda body: {"goal": "fixed", "snapshot": {"step": body.steps}})
    observe = staticmethod(lambda env, obs, phase, step, *args:
                           np.full((224, 224, 3), step if phase == "plan" else 0, np.uint8))
    @staticmethod
    def dispatch(env, command, phase, step, row, meter, store, goal):
        def run():
            env.unwrapped.steps += 1
            return ({"state": np.array([0.])}, {"goal": np.zeros((224, 224, 3), np.uint8)},
                    False, env.stop == (phase, step), 0., command.copy())
        return meter.call("public_step", run)


class Body:
    def __init__(self):
        self.steps = 0
    def eval_state(self, *args):
        return False, None


class Env:
    def __init__(self, stop=None):
        self.unwrapped = Body()
        self.stop = stop


class Meter:
    def __init__(self):
        self.where = {"branch": 0}
    def call(self, kind, fn, **kwargs):
        return fn()


def fixture(tmp_path, stop=None):
    rootdir = tmp_path / "output"
    rootdir.mkdir()
    budget = lm.Ledger(rootdir / "budget.jsonl", {"public_step": 960})
    sha_root = FakePhysical.source.root_hash(np.array([0.]), np.array([1.]))
    warm = np.zeros((10, 2), "<f4")
    plan = np.zeros((25, 2), "<f4")
    root = {"slot": 0, "goal": [1.], "root_sha256": sha_root, "warm_trace": [],
            "packets": {"off": {"raw_action_history": warm, "rgb_history": np.zeros((3, 224, 224, 3), np.uint8),
                                "goal_rgb": np.zeros((224, 224, 3), np.uint8)}}}
    base = kernel.normalization.normalized_history(plan[5:15])
    rgb = [digest(np.full((224, 224, 3), n, np.uint8).tobytes()) for n in (10, 15)]
    def factory(arm, index):
        store = bridge.Journal(rootdir / f"{index:02d}_{arm}_native.jsonl")
        metered = lm.AggregateMeter(Meter(), budget, f"capture_{index:02d}_{arm}")
        session = native.NativeSession(FakePhysical(), Env(stop), root, {}, {"task_id": "task",
            "commands": plan.astype(float).tolist()}, {"index": 0}, metered, store,
            {"prefix_reference": []}, nullcontext())
        return session, store
    obj = bridge.Archive(rootdir, factory, sha_root, "task", base, rgb, warm, plan[:15], budget)
    return obj, budget


def callbacks():
    def start(arm, action, gradient, meter):
        meter.call("start_transition", lambda: None)
        meter.call("start_transition", lambda: None)
        if gradient:
            meter.call("start_gradient", lambda: None)
        return float(np.square(action-.2).sum()), np.full((10, 2), -.1, np.float32) if gradient else None
    def suffix(arm, context, action, gradient, meter):
        meter.call("suffix_transition", lambda: None)
        if gradient:
            meter.call("suffix_gradient", lambda: None)
        return float(np.square(action[5:]-.1).sum()), np.full((10, 2), -.1, np.float32) if gradient else None
    def encode(rgb, history):
        return {"layout": "states_with_between_state_actions_v1",
                "latent_history": np.zeros((3, 192), np.float32).tolist(), "action_history": history.tolist()}
    return start, suffix, encode


def test_fake_native_capture_through_optimizer_and_consumer(tmp_path):
    obj, budget = fixture(tmp_path)
    start, suffix, encode = callbacks()
    bank = [np.full((10, 2), i/100, np.float32).tobytes() for i in range(16)]
    result = kernel.run_case(bank, obj.root, obj.task, obj.base_history, obj.base_rgb,
        evaluate_start=start, capture=obj.capture, encode=lambda rgb, hist: obj.encode(encode, rgb, hist),
        evaluate_suffix=suffix, emit=obj.emit)
    for role in consumer.SOURCE_ROLES:
        obj.put(f"source_{role}", role.encode())
    sources = {role: obj.ref(f"source_{role}") for role in consumer.SOURCE_ROLES}
    anchor = obj.seal(bank, result, sources)
    assert consumer.consume(obj.path, anchor)["physical_authorization"] is False
    receipt = json.loads((obj.path / "00_composition_receipt.json").read_bytes())
    assert receipt["terminated"] is False and receipt["truncated"] is False
    assert budget.attempted["public_step"] == 960


def test_stop_is_preserved_and_never_seals_success(tmp_path):
    obj, _ = fixture(tmp_path, stop=("plan", 18))
    with pytest.raises(RuntimeError, match="stop"):
        obj.capture(kernel.ARMS[0], 0, np.zeros((5, 2), "<f4"))
    assert not (obj.path / "00_composition_receipt.json").exists()
    assert (obj.path / "00_composition_failure.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        obj.seal([], {}, {})


def test_capture_requires_encode_before_next_candidate(tmp_path):
    obj, _ = fixture(tmp_path)
    obj.capture(kernel.ARMS[0], 0, np.zeros((5, 2), "<f4"))
    with pytest.raises(ValueError, match="pending"):
        obj.capture(kernel.ARMS[1], 0, np.zeros((5, 2), "<f4"))


def test_missing_returned_stop_flag_does_not_create_receipt(tmp_path):
    obj, _ = fixture(tmp_path)
    factory = obj.factory
    def missing_flag(arm, index):
        session, store = factory(arm, index)
        step = session.step
        def broken(command):
            value = step(command)
            del value["truncated"]
            return value
        session.step = broken
        return session, store
    obj.factory = missing_flag
    with pytest.raises(KeyError):
        obj.capture(kernel.ARMS[0], 0, np.zeros((5, 2), "<f4"))
    assert (obj.path / "00_composition_failure.json").exists()
    assert not (obj.path / "00_composition_receipt.json").exists()


def test_changed_encoder_input_fails_without_context(tmp_path):
    obj, _ = fixture(tmp_path)
    output = obj.capture(kernel.ARMS[0], 0, np.zeros((5, 2), "<f4"))
    pixels = output["rgb_history"].copy()
    pixels[0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="encoder input"):
        obj.encode(callbacks()[2], pixels, output["action_history"])
    assert not (obj.path / "00_composition_context.json").exists()


def test_optimizer_abort_is_persisted_after_capture_stop(tmp_path):
    obj, _ = fixture(tmp_path, stop=("plan", 18))
    start, suffix, encode = callbacks()
    bank = [np.zeros((10, 2), np.float32).tobytes()] * 16
    with pytest.raises(RuntimeError, match="stop"):
        kernel.run_case(bank, obj.root, obj.task, obj.base_history, obj.base_rgb,
            evaluate_start=start, capture=obj.capture, encode=lambda rgb, hist: obj.encode(encode, rgb, hist),
            evaluate_suffix=suffix, emit=obj.emit)
    events = [json.loads(line) for line in (obj.path / "model_events.jsonl").read_bytes().splitlines()]
    assert events[-1]["kind"] == "root_aborted"
    assert events[-1]["winners"] is None
