from experiments.spectral_reachability.p1_adam_runtime_counts_20261031 import install_counters


class ActionEncoder:
    def __init__(self):
        self.hook = None

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return self

    def remove(self):
        self.hook = None

    def __call__(self, value):
        if self.hook is not None:
            self.hook(self, (value,))
        return value


class Model:
    def __init__(self):
        self.action_encoder = ActionEncoder()

    def encode(self, value):
        return value + 1

    def predict(self, value):
        return self.action_encoder(value) * 2


def test_hooks_observe_calls_without_changing_outputs():
    model = Model()
    counts = dict(encode=0, predict=0, action_encoder=0, backward=0)
    handle = install_counters(model, counts)
    assert model.encode(2) == 3
    assert model.predict(4) == 8
    assert counts == dict(encode=1, predict=1, action_encoder=1, backward=0)
    handle.remove()
    assert model.action_encoder(5) == 5
    assert counts["action_encoder"] == 1
