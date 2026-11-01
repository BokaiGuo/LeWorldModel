"""Synthetic factories only; no GPU, model, or physical environment."""
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_session_factory_codex_20261029 as module
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as storage


def test_setup_failure_closes_env_and_does_not_retry(tmp_path):
    s = storage.Store(tmp_path/'run'); closed=[]
    class Env:
        def close(self): closed.append(True)
    meter_module = SimpleNamespace(Meter=lambda *a:object(), native_factory=lambda meter:Env())
    reset_module = SimpleNamespace(TracedEnv=lambda env,store,episode:env)
    def fail(*args): raise RuntimeError('setup failure')
    native_module = SimpleNamespace(CaptureStore=lambda store:object(), setup=fail)
    factory = module.Factory(dict(tasks=[{},{}]),dict(roots=[{},{}],source_plan={}),s,None,None,None,
                             meter_module=meter_module,native_module=native_module,reset_module=reset_module)
    with pytest.raises(RuntimeError,match='setup failure'):factory(0,0,'fixed_order')
    assert closed == [True]
    with pytest.raises(ValueError,match='single-attempt'):factory(0,0,'fixed_order')
