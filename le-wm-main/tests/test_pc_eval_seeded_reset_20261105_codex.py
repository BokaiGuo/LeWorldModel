import stable_worldmodel as swm

from experiments.pc_eval_seeded_reset_20261105_codex import install_reset_seed


def test_only_missing_seed_is_replaced(monkeypatch):
    received = []

    def original(self, seed=None, options=None):
        received.append((seed, options))

    monkeypatch.setattr(swm.World, "reset", original)
    install_reset_seed(771001)
    swm.World.reset(object(), seed=None, options={"x": 1})
    swm.World.reset(object(), seed=17)
    assert received == [(771001, {"x": 1}), (17, None)]
