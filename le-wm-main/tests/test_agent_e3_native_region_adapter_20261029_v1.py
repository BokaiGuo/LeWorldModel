from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_e3_native_region_adapter_20261029_v1 as adapter


@pytest.mark.parametrize('start,region', [(0, 'prefix'), (5, 'suffix'), (10, 'suffix'), (15, 'suffix')])
@pytest.mark.parametrize('fail', [False, True])
def test_scoped_native_region_and_restore(start, region, fail):
    meter = SimpleNamespace(region='goal_bind')
    seen = []
    goal = np.zeros(7)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    class Env:
        unwrapped = None
        def __init__(self): self.unwrapped = self
        def step(self, command):
            seen.append(meter.region)
            assert meter.region == region
            if fail:
                raise RuntimeError('native failure')
            self.latest_action = command.copy()
            return {}, 0.0, False, False, {'goal_state': goal, 'goal': rgb}
    def capture(*args):
        return None, {'step': args[-1]}
    def transition(*args):
        return None
    class Store:
        def emit(self, *args, **kwargs): return None
    env = Env()
    ops = SimpleNamespace(capture=capture, native_transition=transition)
    row = {'episode_id': 'test', 'control_returned': 0, 'executed_raw': []}
    args = (env, np.zeros((5, 2), dtype=np.float32))
    kw = dict(start=start, row=row, goal=goal, goal_rgb=rgb, meter=meter, store=Store(), ops=ops)
    if fail:
        with pytest.raises(RuntimeError, match='native failure'):
            adapter.packet(*args, **kw)
    else:
        assert adapter.packet(*args, **kw)['end'] == start + 5
    assert seen == [region] * (1 if fail else 5)
    assert meter.region == 'goal_bind'


def test_invalid_boundary_does_not_change_region():
    meter = SimpleNamespace(region='encode')
    with pytest.raises(ValueError):
        adapter.packet(None, None, start=True, row={}, goal=None, goal_rgb=None, meter=meter, store=None)
    assert meter.region == 'encode'
