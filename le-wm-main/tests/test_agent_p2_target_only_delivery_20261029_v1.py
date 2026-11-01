"""RAM fixture/small saved ABI only: no neural, physics or full consumer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_target_only_delivery_20261029_v1 as m


def fixture():
    pm=dict(slots=[]);pr=dict(rows=[]);hm=dict(slots=[]);hr=dict(rows=[],completed=True,errors=[])
    hc=dict(rows=[]);programs=[];arrays={}
    for i in (0,1):
        z=np.arange(576,dtype=np.float32).reshape(1,3,192)/1024+i
        a=np.arange(20,dtype=np.float32).reshape(1,2,10)/1024+i
        goal=np.full((1,192),i+.5,np.float32)
        target=np.full((1,192),i+.25,np.float32)
        roll=np.stack((target,target+1),axis=1)[:,None]
        s=dict(root=i,condition='off',z=z[0].tolist(),a=a[0].tolist(),pixel_sha256=[str(i)]*4,
               blind_sha256='blind'+str(i),task_id='task'+str(i),task_sha256='taskSHA'+str(i))
        pm['slots'].append(s)
        names={k:k+str(i) for k in ('z','a','g','init','goal','target','roll')}
        pr['rows'].append(dict(root=i,status='returned',routes=dict(hi_whole=dict(z=names['z'],a=names['a'])),
                              pixels=[{}, {}, {}, dict(paths=dict(hi=dict(output=names['g'])))]))
        for k,v in (('z',z),('a',a),('g',goal)):arrays['parity',names[k]]=v
        for k,v in (('init',z[:,-1].copy()),('goal',goal.copy()),('target',target),('roll',roll)):arrays['prior',names[k]]=v
        for cond in ('off','on'):
            hm['slots'].append(dict(pixel_sha256=list(s['pixel_sha256']),blind_sha256=s['blind_sha256']))
            hr['rows'].append(dict(root_slot=i,condition=cond,z_init=names['init'],z_goal=names['goal'],
                                  high_target=names['target'],selected_rollout=names['roll']))
            h=hashlib.sha256(target.tobytes()).hexdigest()
            hc['rows'].append(dict(root_slot=i,condition=cond,target_delivered_exact=True,high_target_sha256=h,low_goal_sha256=h))
        ctx=dict(latent_history=z[0].tolist(),action_history=a[0].tolist())
        edges=[dict(index=j,input_context=deepcopy(ctx),task_id=s['task_id'],output_context=dict(
            latent_history=(z[0]+np.float32(j+1)).tolist())) for j in (0,1,2)]
        for arm in ('joint_H3_no_composition','three_program_imagined_recompile'):
            programs.append(dict(root_slot=i,arm=arm,task_id=s['task_id'],task_sha256=s['task_sha256'],
                blind_sha256=s['blind_sha256'],root_context=deepcopy(ctx),goal=goal[0].tolist(),
                edges=deepcopy(edges),compiled_nodes=[0,5,10],raw_steps=15,selection_digest='selection'+str(i),
                commands=['PRIVATE_WITNESS_NOT_DELIVERED']))
    return dict(parity_prepare=pm,parity_report=pr,prior_prepare=hm,prior_report=hr,prior_receipt=hc,
                program_receipt=dict(plans=programs)),arrays


def assemble(d,a):return m.assemble(d,lambda role,name:a[role,name].copy())


def test_fixture_denominator_payload_and_no_witness():
    d,a=fixture();rows=assemble(d,a)
    assert len(rows)==6
    assert [(r['root'],r['arm']) for r in rows]==[(i,arm) for i in (0,1) for arm in ('prior_FIRST','program_first_postcondition','arbitrary')]
    for r in rows:
        if r['arm']=='arbitrary':assert r['status']=='missing' and r['controller_payload'] is None
        else:
            p=r['controller_payload'];assert set(p)=={'z','a','target'}
            v=m.validate_payload(p);assert v['z'].shape==(1,3,192) and v['a'].shape==(1,2,10) and v['target'].shape==(1,192)
    assert 'PRIVATE_WITNESS' not in json.dumps(rows)
    assert rows[0]['origin']['physical_time'] is None
    assert rows[1]['origin']['physical_time']['controller_terminal_equivalence'] is False


@pytest.mark.parametrize('fault',['root','high_order','program_order','missing_program','rgb','blind','init','goal',
    'FIRST','receipt_bytes','receipt_delivered','program_task','program_blind','context_z','context_a',
    'program_goal','edge_index','edge_context','edge_task','compiled_nodes','posthistory','rollshape'])
def test_join_mutations(fault):
    d,a=fixture();h=d['prior_report']['rows'][0];p=d['program_receipt']['plans'][1]
    if fault=='root':d['parity_prepare']['slots'][0]['root']=1
    elif fault=='high_order':h['condition']='on'
    elif fault=='program_order':p['arm']='arbitrary'
    elif fault=='missing_program':d['program_receipt']['plans'].pop()
    elif fault=='rgb':d['prior_prepare']['slots'][0]['pixel_sha256'][0]='other'
    elif fault=='blind':d['prior_prepare']['slots'][0]['blind_sha256']='other'
    elif fault=='init':a['prior',h['z_init']].flat[0]+=1
    elif fault=='goal':a['prior',h['z_goal']].flat[0]+=1
    elif fault=='FIRST':a['prior',h['high_target']].flat[0]+=1
    elif fault=='receipt_bytes':d['prior_receipt']['rows'][0]['high_target_sha256']='0'*64
    elif fault=='receipt_delivered':d['prior_receipt']['rows'][0]['target_delivered_exact']=False
    elif fault=='program_task':p['task_id']='other'
    elif fault=='program_blind':p['blind_sha256']='other'
    elif fault=='context_z':p['root_context']['latent_history'][0][0]+=1
    elif fault=='context_a':p['root_context']['action_history'][0][0]+=1
    elif fault=='program_goal':p['goal'][0]+=1
    elif fault=='edge_index':p['edges'][0]['index']=1
    elif fault=='edge_context':p['edges'][0]['input_context']['latent_history'][0][0]+=1
    elif fault=='edge_task':p['edges'][0]['task_id']='other'
    elif fault=='compiled_nodes':p['compiled_nodes']=[5,10,15]
    elif fault=='posthistory':p['edges'][0]['output_context']['latent_history'].append([0.]*192)
    else:a['prior',h['selected_rollout']]=a['prior',h['selected_rollout']].reshape(1,2,192)
    with pytest.raises(ValueError):assemble(d,a)


@pytest.mark.parametrize('field',['commands','witness','init_action','selected_actions','seed','spec'])
def test_controller_allowlist_rejects_extra(field):
    d,a=fixture();p=assemble(d,a)[0]['controller_payload'];p[field]=[]
    with pytest.raises(ValueError,match='allowlist'):m.validate_payload(p)


@pytest.mark.parametrize('field',['z','a','target'])
def test_no_shape_padding_or_nonfinite(field):
    d,a=fixture();p=assemble(d,a)[0]['controller_payload'];p[field]=[[float('nan')]]
    with pytest.raises(ValueError):m.validate_payload(p)


def test_real_saved_authorities_and_selected_numeric_ABI_only():
    refs,d=m.load_authorities()
    helpers=m.tracker.closure()
    assert all(d['tracker_prepare']['sources'].get(p)==h for p,h in helpers.items())
    def get(role,n):
        folder=Path(refs[role+'_report']['path']).parent
        return m.tracker.parity.load_array(folder,n,d[role+'_report']['arrays'][n])
    rows=m.assemble(d,get)
    assert len(rows)==6 and sum(r['status']=='missing' for r in rows)==2
    assert refs['tracker_receipt']['sha256']=='080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138'
    assert rows[0]['origin']['array']=='array_00215.npy' and rows[3]['origin']['array']=='array_00697.npy'
    assert [r['origin']['pointer'] for r in (rows[1],rows[4])]==[
        '/plans/1/edges/0/output_context/latent_history/2','/plans/3/edges/0/output_context/latent_history/2']


def test_authority_byte_tamper_no_full_consumer(tmp_path,monkeypatch):
    refs=m.authority_paths();first=next(iter(refs));p=tmp_path/'tampered.json';p.write_bytes(b'{}')
    refs[first]['path']=str(p);monkeypatch.setattr(m,'authority_paths',lambda:refs)
    with pytest.raises(ValueError,match='exact JSON SHA'):m.load_authorities()


def test_prepare_verify_fixture_roundtrip_and_no_overwrite(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    d,a=fixture();artifact=dict(schema=m.TAG+'.fixture',rows=assemble(d,a),explicitly_not_HOST=True)
    monkeypatch.setattr(m,'build',lambda:deepcopy(artifact))
    p=tmp_path/'results'/(m.TAG+'_fixture.json');out=m.prepare(p)
    assert m.verify(p,out['sha256'])['full_strict_rebuilt'] is False
    m.main(['verify','--manifest',str(p),'--manifest-sha256',out['sha256']])
    assert json.loads(capsys.readouterr().out)['dispatched'] is False
    with pytest.raises(FileExistsError):m.prepare(p)
    artifact['rows'][0]['target_sha256']='0'*64
    with pytest.raises(ValueError,match='entire delivery'):m.verify(p,out['sha256'])


def test_own_paths_and_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    with pytest.raises(ValueError):m.output_path(tmp_path/'not_results'/ (m.TAG+'_x.json'))
    with pytest.raises(ValueError):m.output_path(tmp_path/'results'/'other.json')
    monkeypatch.setattr(m,'CAP',1);monkeypatch.setattr(m,'build',lambda:dict(fixture=True))
    with pytest.raises(ValueError,match='metadata cap'):m.prepare(tmp_path/'results'/(m.TAG+'_x.json'))


def test_cold_cli_no_torch_no_consumer_execution():
    script='import sys; from experiments.spectral_reachability import '+m.TAG+' as m; m.main(["authorities"]); assert "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert set(json.loads(p.stdout))==set(m.PINS)


def test_build_fixture_source_closure_and_bound_arrays(monkeypatch):
    # build orchestration exercised with RAM metadata/callbacks; never full strict.
    d,a=fixture();refs={k:dict(path='/fixture/'+k,sha256='a'*64) for k in m.PINS}
    d.update(parity_receipt=dict(fixture=True),tracker_receipt=dict(fixture=True),
             tracker_prepare=dict(sources={'fixture_source':'a'*64}))
    for role in ('parity','prior'):
        d[role+'_report']['arrays']={name:dict(file_sha256='a'*64) for (r,name) in a if r==role}
    monkeypatch.setattr(m,'load_authorities',lambda:(refs,d))
    monkeypatch.setattr(m.tracker,'closure',lambda:{'fixture_source':'a'*64})
    monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    monkeypatch.setattr(m.tracker.parity,'load_array',lambda folder,name,desc:a['parity' if str(folder).endswith('parity_report') else 'prior',name].copy())
    # Folder names above must preserve the report role, without fake HOST dispatch.
    refs['parity_report']['path']='/fixture/parity_report/report.json'
    refs['prior_report']['path']='/fixture/prior_report/report.json'
    out=m.build()
    assert out['denominator']==dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0)
    assert out['actual_new_calls']['full_strict']==0 and not out['common_controller']['dispatch']
    assert out['limitations']['generation_budget_comparison'] is None
    assert not out['limitations']['same_generation_budget'] and out['limitations']['prior_physical_time'] is None
    assert out['common_controller']['source_sha256']=='a'*64
    d['tracker_prepare']['sources']['fixture_source']='b'*64
    with pytest.raises(ValueError,match='original tracker closure'):m.build()
