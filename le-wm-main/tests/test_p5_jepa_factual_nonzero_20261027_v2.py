"""No physical/neural execution: fixed schedules, units, durable fixture I/O."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as n


def test_fixed_schedule_no_outcome_selection():
    a=np.array(n.SCHEDULE,np.float64)
    assert a.shape==(3,5,2)
    np.testing.assert_array_equal(a[0],[[.12,.04],[.08,-.06],[-.05,.09],[.11,.03],[-.04,-.07]])
    assert all(not np.array_equal(a[i],a[j]) for i in range(3) for j in range(i))
    assert not np.any(np.all(a==0,axis=-1))
    assert n.SEED==2026102701


def test_native_u_and_pixel_units():
    values=n.action_inputs(n.native_stats(n.ROOT))
    assert values['raw_env_commands'].dtype==np.float64
    assert values['raw_env_commands'].shape==(15,2)
    np.testing.assert_array_equal(values['raw_pixel_displacements'],100*values['raw_env_commands'])
    np.testing.assert_array_equal(values['dataset_pixel_displacements'],values['raw_pixel_displacements'].astype(np.float32))
    np.testing.assert_array_equal(values['raw_env_commands'][0],[.12,.04])


def test_native_dataset_normalization_and_no_extra_divide100():
    s=n.native_stats(n.ROOT);v=n.action_inputs(s)
    expected=((v['dataset_pixel_displacements']/np.float32(100)-s['ACTION_MEAN'])/s['ACTION_STD']).reshape(3,10)
    np.testing.assert_array_equal(v['normalized_packets'],expected)
    assert expected.dtype==np.float32
    wrong=((v['raw_env_commands'].astype(np.float32)/100-s['ACTION_MEAN'])/s['ACTION_STD']).reshape(3,10)
    assert not np.allclose(expected,wrong)


def test_all_model_controls_distinct_and_history_unchanged():
    v=n.action_inputs(n.native_stats(n.ROOT));c=v['controls_packets'];p=v['normalized_packets']
    assert c.shape==(4,3,10) and c.dtype==np.float32
    for arm in c:np.testing.assert_array_equal(arm[:2],p[:2])
    np.testing.assert_array_equal(c[0],p)
    np.testing.assert_array_equal(c[1,-1],np.zeros(10))
    np.testing.assert_array_equal(c[2,-1],-p[-1])
    np.testing.assert_array_equal(c[3,-1].reshape(5,2),p[-1].reshape(5,2)[::-1])
    assert all(not np.array_equal(c[i,-1],c[j,-1]) for i in range(4) for j in range(i))


def test_protocol_interface_and_scope():
    p=n.protocol()
    assert p['schema']=='p5_jepa_factual_nonzero_20261027_v2'
    assert p['context_frames']==[0,5,10] and p['target_frame']==15
    assert p['packet_action_indices']==[[0,5],[5,10],[10,15]]
    assert p['expected']['space_step']==302
    assert not p['neural_execution'] and not p['action_fidelity_claim'] and not p['P5_admission']


def test_prepare_and_helpers_import_no_physics():
    before=set(sys.modules);m=n.prepare(n.ROOT);base=n.helpers(n.ROOT)
    assert m['status']=='prepared_no_physics' and base.NAME==n.PARENT
    assert not any(x.split('.')[0] in ('pymunk','gym','pygame','skimage','torch') for x in set(sys.modules)-before)
    n.verify(n.ROOT,m)


def test_tampered_schedule_fails():
    m=n.prepare(n.ROOT);m['protocol']['schedule'][0][0][0]=0
    with pytest.raises(ValueError):n.verify(n.ROOT,m)


def test_no_outcome_read_or_neural_import_in_source():
    tree=ast.parse(Path(n.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            name=node.module if isinstance(node,ast.ImportFrom) else ' '.join(x.name for x in node.names)
            assert not any(s in name for s in ('torch','pickle','transformers'))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr=='read_text':
                assert ast.unparse(node.func.value) in ('path','args.manifest')
        if isinstance(node,ast.Attribute):assert node.attr!='started'
    assert "'report.json'" in Path(n.__file__).read_text()  # write-only output


def test_exclusive_sha_checked_array(tmp_path):
    base=n.helpers(n.ROOT);p=tmp_path/'x.npy';base.save_array(p,np.zeros(7,np.float32))
    assert n.bounded_array(p,(7,),np.float32,n.sha(p)).shape==(7,)
    with pytest.raises(ValueError):n.bounded_array(p,(7,),np.float64,n.sha(p))
    with pytest.raises(ValueError):n.bounded_array(p,(7,),np.float32,'0'*64)


def test_collector_and_consumer_contract_with_nonphysics_fixture(tmp_path,monkeypatch):
    base=n.helpers(n.ROOT);s=n.native_stats(n.ROOT);v=n.action_inputs(s)
    monkeypatch.setattr(base,'public_state',lambda env:{'fixture_only':True})
    class Calls:
        context=None
        def receipt(self):return {'fixture_only':True}
    class Env:
        def __init__(self,**kwargs):
            self._seed=123;self.agent=SimpleNamespace(position=[10.,20.]);self.i=0
            self.np_random=np.random.default_rng(123);self.random_state=np.random.RandomState(123)
        def seed(self,seed):
            self._seed=seed;self.np_random=np.random.default_rng(seed);self.random_state=np.random.RandomState(seed)
        def arrays(self):
            state=np.array([10,20,30,40,0,0,0],np.float32)
            return dict(visual=np.zeros((224,224,3),np.uint8),proprio=state[[0,1,5,6]]),state
        def reset(self):return self.arrays()
        def step(self,u):
            np.testing.assert_array_equal(u,v['raw_env_commands'][self.i]);self.i+=1
            last=json.loads((tmp_path/'j').read_text().splitlines()[-1])['event']
            assert last['phase']=='intent' and last['operation']=='step'
            obs,state=self.arrays();return obs,0.,False,{'state':state}
        def close(self):return None
    journal=base.Journal(tmp_path/'j')
    records=n.collect_trajectory(base,Env,'a',tmp_path,journal,Calls(),v);journal.close()
    result=n.export_inputs(base,tmp_path/'a',records,v,s)
    assert result['path']=='a/consumer/inputs_manifest.json'
    m=json.loads((tmp_path/result['path']).read_text())
    assert m['frame_order']==[0,5,10,15]
    assert m['frames'][3]['rgb']=='a/frame_15/rgb.npy'
    assert m['frames'][2]['state']=='a/frame_10/state.npy'
    assert m['frames'][1]['proprio']=='a/frame_05/proprio.npy'
    assert m['normalized_packets_path']=='a/consumer/normalized_packets.npy'
    np.testing.assert_array_equal(np.load(tmp_path/m['normalized_packets_path'],allow_pickle=False),v['normalized_packets'])
    assert np.load(tmp_path/'a/consumer/context_proprio.npy',allow_pickle=False).shape==(3,4)
    assert np.load(tmp_path/'a/consumer/states_frames.npy',allow_pickle=False).shape==(4,7)
    assert m['controls_not_executed_physically'] and not m['action_fidelity_claim']
    for k,receipt in m['file_hashes'].items():assert n.sha(tmp_path/'a/consumer'/k)==receipt['sha256']
    command=json.loads((tmp_path/'a/command_06.json').read_text())
    assert command['packet_index']==1 and command['step_within_packet']==0 and command['action_index']==5
    np.testing.assert_array_equal(command['raw_env_argument'],v['raw_env_commands'][5])
    np.testing.assert_array_equal(command['absolute_target'],[10,20]+100*v['raw_env_commands'][5])
    assert len(list((tmp_path/'a').glob('frame_*/rgb.npy')))==16


def test_wrong_manifest_sha_fails_before_runtime(tmp_path):
    m=n.prepare(n.ROOT);f=tmp_path/'manifest';f.write_text(json.dumps(m))
    with pytest.raises(ValueError):n.run(n.ROOT,m,f,'0'*64,tmp_path/'run')
    assert not (tmp_path/'run').exists()
