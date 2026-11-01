from contextlib import nullcontext
from pathlib import Path
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_same_target_host_cuda_20261029_v1 as m


def test_bound_source_hash(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'abc')
    assert m.stream_sha(p)==m.old.sha(b'abc')
    with pytest.raises(ValueError):m.stream_sha(Path('relative'))


def test_four_slot_host_saved_check_non_neural_fixture(tmp_path):
    tasks=[fixture.task(0),fixture.task(1)];targets=[[0.]*192 for _ in range(2)]
    prefixes=[[[0.,0.] for _ in range(5)] for _ in range(2)]
    authority=dict(synthetic=True); policy=tmp_path/'policy'
    previous=dict(policy=str(policy),bindings_sha256={str(policy/'config.json'):'1'*64,str(policy/'weights.pt'):'2'*64})
    binding=dict(expected_model_hash='3'*64)
    loaded=dict(config_sha256='1'*64,weights_sha256='2'*64,strict_key_coverage=True)
    store=m.old.Store(tmp_path/'saved')
    store.event(dict(kind='authority_sealed_before_model',sha256=m.old.sha(m.old.canonical(authority))))
    store.event(dict(kind='model_loaded',loaded=loaded,model_hash='3'*64,cuda=True,tf32=False,autocast=False,math_requested=True))
    staged=m.stage.execute(store,tasks,targets,prefixes,lambda _:fixture.Backend(),
                           lambda i,meter:nullcontext(),lambda:None,store.check)
    staged.update(schema=m.NAME+'.report',prepared='/fixture',prepare_sha256='4'*64,
                  model_hash_before='3'*64,model_hash_after='3'*64,loaded=loaded,host=None,
                  gpu=dict(cap=m.GPU_CAP,peak_allocated=0,peak_reserved=0,tf32_matmul=False,tf32_cudnn=False,
                           autocast=False,requested_math=True,observed_math=None),
                  storage=m.old.storage_bound(),training=False,new_physics=0,new_encoder=0,new_ad=0)
    terminal=m.stage.finish(store,staged)
    report=terminal['report'];completion=m.old.parse_canonical((store.path/'completion.json').read_bytes())
    timing=m.old.parse_canonical((store.path/'timing.json').read_bytes())
    args=(store.path,terminal['report_sha256'],terminal['completion_sha256'],terminal['timing_sha256'],
          report,completion,timing,binding,authority,previous,tasks,targets,prefixes)
    assert m.consume_authenticated(*args)['verification']['verified']
    bad=dict(report,model_hash_after='0'*64)
    badargs=list(args);badargs[4]=bad
    with pytest.raises(ValueError):m.consume_authenticated(*badargs)
