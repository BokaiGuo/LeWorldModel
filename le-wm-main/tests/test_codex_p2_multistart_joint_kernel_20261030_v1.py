from copy import deepcopy

from test_codex_p2_multistart_joint_host_20261030_v1 import FixtureAdapter,scope
from experiments.spectral_reachability import codex_p2_multistart_joint_kernel_20261030_v1 as k


def test_common_ceilings_independent_anchors_no_filler():
    ctx={'latent_history':[[0.0]*192],'action_history':[]}
    case=dict(base=0,context=deepcopy(ctx),goal=[0.0]*192)
    rows=[k.run_branch(case,arm,FixtureAdapter,lambda _:None,scope) for arm in k.ARMS]
    for row in rows:
        assert len(row['candidates'])==16 and len(row['prefix_contexts'])==4
        assert row['resource_counts']['attempted']['jacobian']==5
        assert row['resource_counts']['attempted']['svd']==5
        assert row['resource_counts']['attempted']['score']==17
        assert row['resource_counts']['attempted']==row['resource_counts']['returned']
    assert rows[0]['resource_counts']['attempted']['packet']==44
    assert rows[1]['resource_counts']['attempted']['packet']==28
    assert k.contract()['caps']['packet']==48
