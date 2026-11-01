import base64
import hashlib
import time
from pathlib import Path
import pytest

from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
from experiments.spectral_reachability import agent_e3_partial_guard_saved_20261029_v1 as audit
from test_agent_e3_launcher_authority_fixture_20261029_v1 import actual_authority,synthetic_loader


@pytest.mark.parametrize('mode',['pair','changed_model','bad_class','before_only'])
def test_partial_guard_saved_with_real_authority_synthetic_snapshots(tmp_path,mode):
    authority,sha,_=actual_authority(tmp_path)
    def perform(binding,store):
        synthetic_loader(binding,store)
        source=str(Path(launch.__file__).resolve())
        snap=dict(tensor_sha256=binding['prepared']['expected_model_hash'],
                  classes=[dict(name='',class_module='fixture',class_qualname='Fake',path=source,
                                sha256=binding['additional_sources'][source])],
                  precision=audit.guard.host.primitives.precision_expected(),allocated=1,reserved=2)
        if mode=='bad_class':snap['classes'][0]['sha256']='0'*64
        def record(stage,value):
            raw=launch.host.raw(value);h=hashlib.sha256(raw).hexdigest()
            store.event('physical','record','guard_snapshot_chunk',store.index,
                        dict(stage=stage,index=0,total=1,sha256=h,data=base64.b64encode(raw).decode()))
            store.event('physical','record',stage,store.index,dict(sha256=h,total=1,bytes=len(raw)))
        record('guard_before',snap)
        if mode!='before_only':
            if mode=='changed_model':snap=dict(snap,tensor_sha256='0'*64)
            record('guard_after',snap)
        raise RuntimeError('fixture failure')
    result=launch._run_lifecycle(tmp_path/'job',time.monotonic(),
        lambda:launch.read_authority(str(authority),sha),perform)
    seal=result['publication'];args=(tmp_path/'job',seal['report_sha256'],seal['completion_sha256'],str(authority),sha)
    if mode=='changed_model':
        with pytest.raises(ValueError,match='unchanged model'):audit.inspect(*args)
    elif mode=='bad_class':
        with pytest.raises(ValueError,match='sealed class source'):audit.inspect(*args)
    else:
        result=audit.inspect(*args)
        assert result['partial']['status']=='partial' and not result['full_saved_strict']
        assert result['guard_pair_verified']==(mode=='pair')
