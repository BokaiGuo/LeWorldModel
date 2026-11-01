from pathlib import Path
import runpy
import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_gn_matched_host_20261029_v1 as m

fixture=runpy.run_path(str(Path(__file__).parent/'test_agent_p3_parent_native_cuda_20261029_v1.py'))


class Fixture(fixture['NumpyNative']):
    def linearize(self,z,ledger):
        self.ledger=ledger;self.step=None;self.phase='current_AD'
        raw=m.gn.raw(z);task=self.task;core=m.core
        def jacobian():
            out=np.asarray(self.trajectory(task['root_context'],raw,False)[-1]['latent_history'][-1],np.float32)
            self.step=None;self.arrays('AD_terminal',value=out)
            j=np.zeros((192,30),np.float32)
            clock=float(task['root_context']['latent_history'][-1][2])
            for k in range(3):
                j[0,k*10:k*10+10:2]=np.float32(1+clock*1e8)
                j[1,k*10+1:k*10+10:2]=np.float32(clock>=5e-9)
                clock=float(np.float32(clock)+np.float32(1e-8))
            for i,row in enumerate(j):
                self.sink.event(dict(kind='direction',scope=ledger.scope,direction=i))
                basis=np.eye(192,dtype=np.float32)[i]
                def raw_return():self.arrays('raw_backward',len(ledger.events)-1,value=row);return row.copy()
                ledger.call('vjp',lambda:self.call('grad',lambda:ledger.call('raw_backward',raw_return),
                    {'output':out,'raw':raw.reshape(-1),'basis':basis}),directions=1)
            self.arrays('jacobian',value=j)
            return out,j
        out,j=ledger.call('jacobian',jacobian)
        z=np.asarray(z,np.float64)
        return dict(logits=z.tolist(),raw=raw.tolist(),nominal=out.tolist(),raw_jacobian=j.tolist(),
                    logit_jacobian=(j.astype(np.float64)*(1-np.tanh(z)**2)[None,:]).tolist())


def test_prepare_and_authorization(tmp_path):
    path=tmp_path/'prepare.json';m.life.single._put(path,m.prepare_value());sha=m.bridge.sha(path)
    p=m.verify_prepare(path,sha)
    assert p['budget_per_root']['predict_samples']==117
    assert p['budget_per_root']['VJP_directions']==768
    with pytest.raises(ValueError,match='authorization'):
        m.run(path,sha,tmp_path/'out','not-authorized')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('root',[0,1])
def test_full_slot_and_saved_reconstruction(tmp_path,root):
    from copy import deepcopy
    task=deepcopy(m.bridge.build_bridge()['roots'][root]['task'])
    task['root_context']['latent_history'][-1][2]=0.0
    task['binding']['context_sha256']=m.core.digest(task['root_context'])
    path=tmp_path/'slot'
    report,sha=m.run_slot(path,task,lambda sink:Fixture(task,sink),lambda:None,'RAM_FIXTURE',root)
    assert report['status']=='complete',report['error']
    assert m.audit_slot(path,sha,task,'RAM_FIXTURE',root)['status']=='complete'
    events=[x['payload'] for x in m.life.transport.read_journal(path/'events.jsonl')]
    start=next(i for i,e in enumerate(events) if e['kind']=='native_call_attempt' and e['method']=='current_ad')
    end=next(i for i in range(start+1,len(events)) if events[i]['kind']=='native_call_return')
    call=deepcopy(events[start:end+1]);call[-1]['result']['raw_jacobian'][0][0]+=1
    with pytest.raises(ValueError):m.slot_impl.audit_ad(call)
    with (path/'events.jsonl').open('ab') as f:f.write(b'{}\n')
    with pytest.raises(ValueError):m.audit_slot(path,sha,task,'RAM_FIXTURE',root)


def test_failure_never_releases_candidate(tmp_path):
    task=m.bridge.build_bridge()['roots'][0]['task'];path=tmp_path/'slot'
    report,sha=m.run_slot(path,task,lambda sink:Fixture(task,sink,fail=True),lambda:None,'RAM_FIXTURE',0)
    assert report['status']=='failed' and report['result'] is None
    with pytest.raises(ValueError):m.audit_slot(path,sha,task,'RAM_FIXTURE',0)
