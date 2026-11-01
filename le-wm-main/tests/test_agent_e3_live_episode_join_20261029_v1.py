from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_e3_live_episode_join_20261029_v1 as m


@pytest.mark.parametrize('close_fail',[False,True])
def test_setup_failure_still_closes_and_preserves_primary(tmp_path,close_fail):
    class Env:
        calls=0
        def close(self):
            self.calls+=1
            if close_fail:raise RuntimeError('close failed')
    env=Env();store=m.capture.pub.Store(tmp_path/'job');row={'episode_id':'cell00'}
    with pytest.raises(TypeError):
        m.execute(env,{},None,None,None,None,None,store,SimpleNamespace(region='setup'),row)
    assert env.calls==1 and row['status']=='failed'
    if close_fail:assert row['cleanup_errors'][0]['message']=='close failed'
    seal=store.publish('partial')
    assert not m.capture.pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])['complete']
