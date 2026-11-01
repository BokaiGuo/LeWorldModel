import pytest

from experiments.spectral_reachability import p1_four_arm_hooked_cuda_20261030_main as audit


class FakeHandle:
    def __init__(self, callbacks, cb):
        self.callbacks, self.cb = callbacks, cb

    def remove(self):
        self.callbacks.remove(self.cb)


class FakeModule:
    def __init__(self):
        self.callbacks = []

    def register_forward_pre_hook(self, cb):
        self.callbacks.append(cb)
        return FakeHandle(self.callbacks, cb)

    def signal(self):
        for cb in self.callbacks:
            cb(self, ())


class FakeModel:
    def __init__(self):
        self.predictor, self.action_encoder = FakeModule(), FakeModule()


def test_hooks_count_and_remove_on_exception():
    model = FakeModel()
    with pytest.raises(RuntimeError):
        with audit.ModuleCalls(model) as hook:
            model.predictor.signal()
            model.action_encoder.signal()
            model.action_encoder.signal()
            assert hook.counts == {"predict": 1, "action_encode": 2}
            raise RuntimeError("failure")
    assert not model.predictor.callbacks and not model.action_encoder.callbacks


def test_ledger_mismatch_rejected():
    account = {"charged_upper_bounds": {"predict_calls": 1, "action_encode_calls": 2}}
    assert audit.validate({"predict": 1, "action_encode": 2}, account)["predict_calls"] == 1
    with pytest.raises(ValueError, match="mismatch"):
        audit.validate({"predict": 0, "action_encode": 2}, account)
