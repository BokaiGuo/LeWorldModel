"""Synthetic RAM + actual writer/reader ABI. No neural/physics/production strict."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as m

spec=importlib.util.spec_from_file_location('old_cp5_fixture',Path(__file__).with_name('test_agent_p3_actual_cp5_task_builder_20261029_v1.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def save(p,value):
    p.write_bytes(m.canonical(value)+b'\n')
    return dict(path=str(p),sha256=m.sha(p))


def png(path,root):
    a=np.full((224,224,3),30+root,np.uint8);buf=io.BytesIO();Image.fromarray(a).save(buf,format='PNG')
    path.write_bytes(buf.getvalue())
    return m.sha(path),hashlib.sha256(a.tobytes()).hexdigest()


def snapshot(state):
    return dict(state=deepcopy(state),state_float64_bits=np.asarray(state,dtype='<f8').tobytes().hex(),bodies={})


@pytest.fixture
def disk(tmp_path):
    p,r,c,e,a=old.fixture()
    # Only fixture schema, not evidence of actual model encoding. Fixed real IDs
    # preserve the resolver's prohibition on introducing replacement roots.
    for root in range(2):
        t=p['tasks'][root];t['binding'].update(model_sha256=m.MODEL_SHA,root_sha256=m.KNOWN_ROOTS[root])
        t['action_mean']=np.asarray([-.0087,.0068],np.float32).astype(float).tolist()
        t['action_std']=np.asarray([.2019,.2002],np.float32).astype(float).tolist()
        warm=np.asarray(e[root]['task']['root_raw_action_history'],np.float32)
        t['root_context']['action_history']=m.cp5.normalize(t,warm).tolist()
        t['binding'].update(context_sha256=m.digest(t['root_context']),normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        for k in ('root_context','action_mean','action_std'):e[root]['task'][k]=deepcopy(t[k])
        image=tmp_path/f'r{root}.png';h,pixel=png(image,root)
        e[root]['native_root_sha256']=None  # actual blind export ABI, never invent native identity
        e[root]['blind_sha256']=e[root]['blind_source_sha256']=t['binding']['blind_sha256']
        e[root]['task_contract'].update(pixel_sha256=['1'*64,'2'*64,pixel,'4'*64],with_target=False)
    p['sources']={}
    for root in range(2):
        ep=tmp_path/f'export{root}.json.gz';ep.write_bytes(gzip.compress(m.canonical(e[root]),mtime=0))
        p['layers'][root].update(task_export=str(ep),task_sha256=m.sha(ep))
        p['tasks'][root]['binding']['task_sha256']=m.sha(ep);p['sources'][str(ep)]=m.sha(ep)
    old.populate(p,r,c,e)
    pref=save(tmp_path/'planner_prepare.json',p)
    r.update(prepared=pref['path'],prepare_sha256=pref['sha256'])
    rref=save(tmp_path/'planner_report.json',r)
    c.update(prepared=pref['path'],prepare_sha256=pref['sha256'],report_sha256=rref['sha256'])
    cref=save(tmp_path/'planner_receipt.json',c)
    native=dict(schema=m.PHYSICAL+'.consume',status='h3_witness4_control20_native_recorded_not_admitted',
        errors=[],all_four_physically_executed=True,rows=[],bindings={},planner_authority=dict(
            prepare_sha256=pref['sha256'],report_sha256=rref['sha256'],receipt_sha256=cref['sha256']))
    for root in range(2):
        for bank in ('endpoints','midpoints'):
            idx=len(native['rows']);state=[float(root+v) for v in range(7)];state5=[x+.5 for x in state]
            ip=tmp_path/f'r{root}.png';pixel=e[root]['task_contract']['pixel_sha256'][2]
            c0=dict(step=0,state=state,snapshot=snapshot(state),png=ip.name,file_sha256=m.sha(ip),pixel_sha256=pixel)
            c5=dict(step=5,state=state5,snapshot=snapshot(state5),png=f'c5_{idx}.png',file_sha256='5'*64,pixel_sha256='2'*64)
            req=next(n['request'] for n in r['nodes'] if n['episode_id']==f's{root}_adaptive_{bank}')
            w=dict(root_slot=root,bank=bank,boundary=5,task_id=p['tasks'][root]['binding']['task_id'],
                prefix=deepcopy(r['rows'][root*4]['prefix']),task=deepcopy(p['tasks'][root]),request=deepcopy(req),actual_cp5=deepcopy(c5),
                result={'unused_future_Q':9999},suffix_commands='not consumed')
            prefix=p['layers'][root]['commands'];goal=[float(v+30) for v in range(7)]
            plan=dict(source_witness=w,prefix_commands=deepcopy(prefix),native_goal=goal,native_goal_pixel_sha256='4'*64)
            native['rows'].append(dict(index=idx,root_slot=root,arm=bank,condition='off',initial_exact=True,
                task_id=p['tasks'][root]['binding']['task_id'],root_sha256=m.KNOWN_ROOTS[root],root_state=state,
                goal=goal,goal_pixel_sha256='4'*64,checkpoints=[c0,c5],plan=plan,unused_future_success=False))
            native['bindings'][str(ip)]=m.sha(ip)
    nreport=dict(schema=m.PHYSICAL+'.report',rows=deepcopy(native['rows']))
    rr=save(tmp_path/'native_report.json',nreport);native['physical_report_sha256']=rr['sha256']
    native['bindings'][rr['path']]=rr['sha256']
    nr=save(tmp_path/'native_receipt.json',native)
    manifest=dict(schema=m.NAME+'.inputs',planner=dict(prepared=pref,report=rref,receipt=cref),native_receipt=nr)
    ir=save(tmp_path/'inputs.json',manifest)
    return dict(tmp=tmp_path,p=p,r=r,c=c,native=native,nreport=nreport,manifest=manifest,ref=ir)


def update(disk,key,value):
    """Rebind fixture metadata after intentional coherent mutations."""
    if key=='native':
        nr=save(disk['tmp']/'native_receipt.json',value);disk['manifest']['native_receipt']=nr
    elif key=='manifest':disk['manifest']=value
    else:raise AssertionError(key)
    disk['ref']=save(disk['tmp']/'inputs.json',disk['manifest'])


def build(disk):return m.build(disk['ref']['path'],disk['ref']['sha256'])


def test_full16_disk_build_write_verify_and_permissions(disk):
    x=build(disk);assert x['ready']==16 and x['blocked']==0 and len(x['cells'])==16
    assert len({c['task']['task_id'] for c in x['cells']})==16
    assert len({c['task']['task_sha256'] for c in x['cells']})==16
    for c in x['cells']:
        body=c['task']['body'];assert m.digest(body)==c['task']['task_sha256']
        assert set(body)==set(x['planner_allowlist'])==set(m.PLANNER_FIELDS)
        assert 'native_goal' not in m.canonical(body).decode() and 'goal_evidence' not in body
        assert 'evaluator' not in body and 'identity' not in body
        assert c['evaluator']['planner_may_read'] is False
        if c['goal_index']==1:
            z=disk['p']['tasks'][c['root']]['root_context']['latent_history'][-1]
            assert body['goal']==z and c['goal_evidence']['original_control_step']==0
        assert c['task']['task_id']!=disk['p']['tasks'][c['root']]['binding']['task_id']
    out=m.write_new(str(disk['tmp']/(m.NAME+'_fixture.json')),x)
    assert out['bytes']<m.OUTPUT_CAP and m.verify(out['path'],out['sha256'])['verified']
    assert all(x[k] is False for k in m.FLAGS) and sum(x['new_calls'].values())==0
    assert x['evidence']['full_upstream_rebuilt_here'] is False


def test_whole_output_mutation_rejected(disk):
    x=build(disk);x['cells'][0]['task']['body']['goal'][0]+=1
    out=m.write_new(str(disk['tmp']/(m.NAME+'_bad.json')),x)
    with pytest.raises(ValueError,match='whole own16'):m.verify(out['path'],out['sha256'])


def test_own_source_binding_preserved(disk):
    x=build(disk);x['bindings'][str(m.own_files()[0])]='0'*64
    out=m.write_new(str(disk['tmp']/(m.NAME+'_seal.json')),x)
    with pytest.raises(ValueError,match='whole own16'):m.verify(out['path'],out['sha256'])


def test_missing_existing_root_png_blocks_only_G1(disk):
    (disk['tmp']/'r0.png').unlink()
    x=build(disk)
    assert (x['ready'],x['blocked'])==(12,4)
    blocked=[c for c in x['cells'] if c['status']=='blocked']
    assert all(c['root']==0 and c['goal_index']==1 and c['task'] is None for c in blocked)


def test_missing_authority_keeps16_no_fake_task(disk):
    disk['manifest']['native_receipt']=None;update(disk,'manifest',disk['manifest'])
    x=build(disk);assert x['blocked']==16 and x['ready']==0
    assert all(c['task'] is None for c in x['cells'])
    assert not x['evidence']['root_PNG_redecoded_here'] and not x['evidence']['inherited_MAIN_complete_receipts']


def test_unsupported_native_schema_blocked(disk):
    n=deepcopy(disk['native']);n['schema']='legacy.fake';update(disk,'native',n)
    x=build(disk);assert x['blocked']==16


def test_unsupported_planner_schema_blocked(disk):
    p=deepcopy(disk['p']);p['schema']='unapproved.other_prepare'
    disk['manifest']['planner']['prepared']=save(disk['tmp']/'planner_prepare.json',p)
    update(disk,'manifest',disk['manifest'])
    assert build(disk)['blocked']==16


def test_decoded_RGB_pixel_hash_is_actually_checked(disk):
    p=disk['tmp']/'r0.png'
    with pytest.raises(ValueError,match='decoded pixel SHA'):
        m.pixels(dict(path=str(p),sha256=m.sha(p)),'0'*64,m.Reader())


def test_missing_native_cp0_no_G1_fabrication(disk):
    n=deepcopy(disk['native']);n['rows'][0]['checkpoints'].pop(0)
    r=deepcopy(disk['nreport']);r['rows']=deepcopy(n['rows']);rr=save(disk['tmp']/'native_report.json',r)
    n['physical_report_sha256']=rr['sha256'];n['bindings'][rr['path']]=rr['sha256'];update(disk,'native',n)
    x=build(disk);assert x['blocked']==8 and x['ready']==8
    assert all(c['task'] is None for c in x['cells'][:8])


@pytest.mark.parametrize('kind',['root','pixel','state','bool','prefix','context','G0goal','model'])
def test_native_prefix_and_goal_mutations_rejected(disk,kind):
    n=deepcopy(disk['native']);r=deepcopy(disk['nreport']);q=n['rows'][0]
    if kind=='root':q['root_sha256']='0'*64
    if kind=='pixel':q['checkpoints'][0]['pixel_sha256']='0'*64
    if kind=='state':q['root_state'][0]+=1
    if kind=='bool':q['index']=False
    if kind=='prefix':q['plan']['source_witness']['prefix']['commands'][0][0]=.5
    if kind=='context':q['plan']['source_witness']['task']['root_context']['latent_history'][0][0]+=1
    if kind=='G0goal':q['plan']['native_goal'][0]+=1
    if kind=='model':q['plan']['source_witness']['task']['goal'][0]+=1
    r['rows']=deepcopy(n['rows']);rr=save(disk['tmp']/'native_report.json',r)
    n['physical_report_sha256']=rr['sha256'];n['bindings'][rr['path']]=rr['sha256'];update(disk,'native',n)
    with pytest.raises(ValueError):build(disk)


def test_unused_future_outcomes_not_task_generation_inputs(disk):
    x=build(disk)
    n=deepcopy(disk['native']);n['rows'][0]['unused_future_success']=True
    n['rows'][0]['plan']['source_witness']['result']={'unused_future_Q':-1000000}
    update(disk,'native',n);y=build(disk)
    assert y['ready']==16
    # Whole authority changed, therefore own identities change, but planner
    # observations/goals/domains and budgets are EXACTLY invariant.
    for a,b in zip(x['cells'],y['cells']):
        aa=deepcopy(a['task']['body']);bb=deepcopy(b['task']['body'])
        aa.pop('identity_sha256');bb.pop('identity_sha256');assert aa==bb
        assert a['budget']==b['budget']


def test_sha_and_decoded_pixel_mutation(disk):
    p=disk['tmp']/'r0.png';p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError,match='SHA'):build(disk)


@pytest.mark.parametrize('K',[1,3])
@pytest.mark.parametrize('domain',['Dxy','Dx'])
def test_chart_mask_normalization_and_history_not_modified(K,domain):
    x=np.zeros((K,5,2));x[:,:,0]=.125;before=x.copy()
    y=m.mask_chart(x.tolist(),K,domain)
    assert y.shape==(5*K,2) and y.dtype==np.float32 and np.array_equal(x,before)
    assert y[:,0].tobytes()==np.tanh(x.astype(np.float32))[:,:,0].ravel().tobytes()
    if domain=='Dx':assert y[:,1].tobytes()==b'\0'*(5*K*4)
    g=np.ones((K,5,2));gg=m.mask_gradient(g.tolist(),K,domain)
    assert np.all(g==1) and np.all(gg[:,:,0]==1)
    if domain=='Dx':assert np.all(gg[:,:,1]==0)
    mean=np.asarray([-.0087,.0068],np.float32);std=np.asarray([.2019,.2002],np.float32)
    if domain=='Dx':assert np.all(((y-mean)/std)[:,1]!=0)  # raw zero != normalized zero


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),'0'])
def test_numeric_core_strict_literals(value):
    x=np.zeros((1,5,2)).tolist();x[0][0][0]=value
    with pytest.raises(ValueError):m.mask_chart(x,1,'Dxy')


def test_chart_no_silent_projection_or_masking():
    x=np.zeros((1,5,2));x[0,0,1]=.1
    with pytest.raises(ValueError,match='already be masked'):m.mask_chart(x.tolist(),1,'Dx')
    x[:]=10
    with pytest.raises(ValueError,match='packet ball'):m.mask_chart(x.tolist(),1,'Dxy')
    with pytest.raises(ValueError):m.mask_chart(np.zeros((1,5,2)).tolist(),True,'Dx')


def test_all16_budget_exact_and_no_FLOPs_claim():
    rows=m.roster();b=m.complete_path_budget();assert len(rows)==16
    for key in ('predict_API','non_AD_predict','AD_primal_predict','predict_samples','predict_frame_samples',
                'scalar_reverse','encoder_new','projector_new','all_public','complete_path_physics'):
        assert sum(c['budget'][key] for c in rows)==b[key]
    assert b['predict_API']==6528 and b['scalar_reverse']==1536 and b['FLOPs'] is None


def test_real_disk_fsync_open_file_and_exclusive(tmp_path,monkeypatch):
    real=m.os.fsync;seen=[]
    def sync(fd):
        seen.append('dir' if stat.S_ISDIR(m.os.fstat(fd).st_mode) else 'file');real(fd)
    monkeypatch.setattr(m.os,'fsync',sync)
    p=tmp_path/(m.NAME+'_codec.json');out=m.write_new(str(p),{'fixture':True})
    assert seen==['file','dir'] and out['sha256']==m.sha(p)
    with pytest.raises(FileExistsError):m.write_new(str(p),{'fixture':True})


def test_cli_complete_build_verify(disk,capsys):
    output=str(disk['tmp']/(m.NAME+'_cli.json'));ref=disk['ref']
    m.main(['prepare','--inputs',ref['path'],'--inputs-sha',ref['sha256'],'--output',output])
    m.main(['verify','--prepared',output,'--prepare-sha',m.sha(output)])
    assert '"verified": true' in capsys.readouterr().out


def test_actual_saved_prefix_ABI_rebound_to_temporary_disk(tmp_path):
    """Actual saved numeric/pixel schema, NOT production build or new receipt.

    Reconstruct a minimal synthetic shared-report envelope using the already
    saved cp5 join. Retain all8 branches, all2 roots and both physical banks.
    No 132MiB producer report/fullstrict/neural replay is executed here.
    """
    results=m.ROOT/'results'
    paths=[results/'agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json',
           results/'agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json',
           results/'agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json']
    if not all(p.is_file() for p in paths):pytest.skip('local saved prefix ABI assets absent')
    p,j,n=[json.loads(path.read_bytes()) for path in paths]
    r=dict(schema=m.cp5.PLANNER+'.report',rows=[],nodes=[])
    c=dict(schema=m.cp5.PLANNER+'.consume',status='saved_shared_panel_verified_NOT_admitted',complete=True,audited=[])
    for root in range(2):
        layer=p['layers'][root];oldpath=Path(layer['task_export']);newpath=tmp_path/f'export{root}.json.gz'
        newpath.write_bytes(oldpath.read_bytes());layer['task_export']=str(newpath)
        p['sources'][str(newpath)]=m.sha(newpath)
        prov=j['provenance'][root];base=p['tasks'][root]
        for b in prov['branch_witnesses']:
            eid=b['episode_id'];mode=eid.split('_',1)[1]
            identity=dict(episode_id=eid,source_slot=root,mode=mode,boundary=5,
                          raw10=deepcopy(prov['raw10']),observation=deepcopy(b['observation']))
            task=m.cp5.local_task(base,j['tasks'][root]['root_context'],identity,m.cp5.LIVE)
            req=dict(schema=m.cp5.PLANNER+'.node',base_task_sha256=m.digest(base),task=task,**identity)
            r['nodes'].append(dict(episode_id=eid,source_slot=root,mode=mode,boundary=5,status='sealed',request=req))
            prefix=dict(commands=deepcopy(prov['prefix5']),raw_float32_sha256=hashlib.sha256(
                np.asarray(prov['prefix5'],np.float32).tobytes()).hexdigest(),source_selection_sha256=layer['selection_sha256'])
            r['rows'].append(dict(episode_id=eid,source_slot=root,mode=mode,prefix_exact=True,prefix=prefix))
            c['audited'].append(dict(episode_id=eid,boundary=5,node_verified=True))
    pref=save(tmp_path/'p.json',p);r.update(prepared=pref['path'],prepare_sha256=pref['sha256'])
    rref=save(tmp_path/'r.json',r);c.update(prepared=pref['path'],prepare_sha256=pref['sha256'],report_sha256=rref['sha256'])
    cref=save(tmp_path/'c.json',c)
    oldreport=Path(next(path for path,h in n['bindings'].items() if h==n['physical_report_sha256']))
    for row in n['rows']:
        row['checkpoints']=[q for q in row['checkpoints'] if q['step'] in (0,5)]
        # Drop futures from the fixture. The real accepted source is untouched.
        keep=('root_slot','bank','boundary','task_id','prefix','task','request','actual_cp5')
        row['plan']['source_witness']={k:row['plan']['source_witness'][k] for k in keep}
    bindings={}
    for row in n['rows']:
        cp=row['checkpoints'][0];ip=tmp_path/cp['png'];ip.write_bytes((oldreport.parent/cp['png']).read_bytes())
        bindings[str(ip)]=m.sha(ip)
    report=dict(schema=m.PHYSICAL+'.report',rows=deepcopy(n['rows']))
    rr=save(tmp_path/'native_report.json',report);bindings[rr['path']]=rr['sha256']
    n.update(bindings=bindings,physical_report_sha256=rr['sha256'],planner_authority=dict(
        prepare_sha256=pref['sha256'],report_sha256=rref['sha256'],receipt_sha256=cref['sha256']))
    nr=save(tmp_path/'native_receipt.json',n)
    manifest=dict(schema=m.NAME+'.inputs',planner=dict(prepared=pref,report=rref,receipt=cref),native_receipt=nr)
    ir=save(tmp_path/'inputs.json',manifest)
    value=m.build(ir['path'],ir['sha256'])
    assert value['ready']==16 and value['blocked']==0
    output=m.write_new(str(tmp_path/(m.NAME+'_actual_ABI_fixture.json')),value)
    assert output['bytes']<=m.OUTPUT_CAP
    assert m.verify(output['path'],output['sha256'])['verified']
    print('actual-prefix-ABI temporary full16 serialized bytes:',output['bytes'])
