"""RAM-only schema/NPY/module mocks. No actual consume/prepare/model/physics."""
from copy import deepcopy
from functools import lru_cache
from io import BytesIO
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import moving_history_observed_suffix_rgb_cuda_20261027_v1 as m
import test_moving_history_observed_suffix_physical_20261027_v1 as phys
from test_p1_constructive_moving_history_encode_cuda_20261027_v1 import protocol
from test_p1_constructive_selected_rgb_cuda_20261027_v1 import RAMStore


@lru_cache(maxsize=1)
def _fixture():
    p,d=phys.fixture();tasks=[];branches={};selected={};results={};prefixes={}
    p.update(manifest='/RAM/manifest.json',manifest_sha256='a'*64,sourcepack_sha256=m.parent.SOURCEPACK_SHA,
        original_physical_report_sha256=m.parent.PHYSICAL_REPORT_SHA)
    for slot,ref in enumerate(d['references']):
        packet=dict(rgb_history=ref['history'],raw_action_history=ref['raw_history'],goal_rgb=ref['goal_rgb'])
        pr=protocol();s=pr['slots'][slot];s.update(blind_sha256=ref['blind_sha256'],pixel_sha256=[m.moving.digest(a.tobytes()) for a in [*packet['rgb_history'],packet['goal_rgb']]])
        export=m.moving.task_export(pr,'d'*64,s,packet,[np.full(192,i,np.float32) for i in range(4)])
        tasks.append(m.moving.task_interfaces(export))
    for index,(plan,pair) in enumerate(zip(p['plans'],d['paired'],strict=True)):
        slot=plan['root_slot'];arm=plan['arm'];task=tasks[slot];t=task['gn_task'];k=(slot,arm)
        old=dict(schema=m.compiler.planner.NAME+'.model_plan',source_slot=slot,arm=arm,task_id=task['task_id'],
            task_export=m.parent.EXPORTS[slot],task_sha256=m.parent.EXPORT_SHAS[slot],blind_sha256=d['references'][slot]['blind_sha256'],
            commands=deepcopy(pair['original_commands']),with_target=False,selection_uses_physics=False,P1_admission=False)
        old['selection_sha256']=m.compiler.core.digest(old)
        item={f:old[f] for f in ('source_slot','arm','task_id','task_export','task_sha256','blind_sha256')}
        item.update(root_slot=slot,role='prefix5',planner_selection_sha256=old['selection_sha256'],
            selection_digest=pair['original_selection_digest'],checkpoint=deepcopy(pair['prefix'][5]))
        z=np.full(192,2.5,np.float32);buf=BytesIO();np.save(buf,z,allow_pickle=False)
        entry=dict(path=f'/RAM/prefix{index}.npy',file=f'prefix{index}.npy',sha256=m.enc.digest(buf.getvalue()),shape=[192],dtype='float32')
        b=m.compiler.branch_input(slot,arm,task,old,item,z,entry);branches[k]=b
        prefix=dict(entry=deepcopy(entry),latent=z.astype(float).tolist(),checkpoint=deepcopy(item['checkpoint']),
            planner_selection_sha256=b['planner_selection_sha256'],original_rgb_report_sha256=m.compiler.RGB_REPORT_SHA)
        prefixes[k]=prefix
        raw=np.asarray(plan['commands'][5:],np.float32);params=np.arctanh(raw.reshape(10)).astype(float)
        evaluations=[]
        for stage,commands,newz in zip(m.compiler.STAGES[2:],(b['old_suffix_commands'],raw.tolist(),raw.tolist()),(2.625,2.75,2.75)):
            before=b['observed_context'];incoming=m.compiler.planner.incoming(b,before,commands)
            context=dict(before,latent_history=before['latent_history'][1:]+[[newz]*192],action_history=incoming['action_history'][-2:])
            evaluations.append(dict(stage=stage,commands=commands,status='completed',context=context,cost=float(np.mean((np.asarray(context['latent_history'][-1])-t['goal'])**2))))
        result=dict(index=index,source_slot=slot,arm=arm,status='suffix_compiled',branch_id=b['branch_id'],equal_new_replay=True,dispatch='new_suffix_unconditionally',
            evaluations=evaluations,prefix_commands=b['prefix_commands'],observed_context_sha256=m.compiler.core.digest(b['observed_context']),
            solve=dict(params=params.tolist(),radius=b['packet_logit_cap']),new_suffix_commands=raw.astype(float).tolist())
        body=dict(schema=m.compiler.NAME+'.model_plan',source_slot=slot,arm=arm,branch_id=b['branch_id'],
            parent_task_id=b['parent_task_id'],blind_sha256=b['blind_sha256'],planner_selection_sha256=b['planner_selection_sha256'],
            observed_context_sha256=m.compiler.core.digest(b['observed_context']),prefix_commands=b['prefix_commands'],suffix_commands=result['new_suffix_commands'],
            commands=b['prefix_commands']+result['new_suffix_commands'],selection='new_suffix_unconditionally_old_diagnostic_only',P1_admission=False,P4_admission=False)
        result['plan']=dict(body,selection_sha256=m.compiler.core.digest(body));results[k]=result
        selected[k]=dict(result['plan'],whole_run_report_sha256=m.COMPILER_REPORT_SHA)
        plan.update(compiler_report='/RAM/consume.json',compiler_report_SHA='b'*64,compiler_schema=m.compiler.NAME+'.consume',
            compiler_status='observed_suffix_commands_verified_not_admitted',commands_pointer=f'/plans/{index}/commands',
            source_slot_pointer=f'/plans/{index}/source_slot',arm_pointer=f'/plans/{index}/arm',blind_sha_pointer=f'/plans/{index}/blind_sha256')
        plan.pop('selection_digest');plan['selection_digest']=m.physical.source.digest(plan)
    report,store,_=phys.execute(p=p,data=d)
    assert report['status']=='moving_history_observed_suffix_physical_recorded'
    return report,store.events,p,branches,selected,results,tasks,prefixes


def fixture():return deepcopy(_fixture())


def test_four_terminal_only_fixed_order_and_no_mutation():
    data=fixture();before=deepcopy(data);images=m.assemble(*data)
    assert data==before and len(images)==4
    assert [(i['source_slot'],i['arm'],i['role']) for i in images]==[(s,a,'terminal10') for s in (0,1) for a in m.ARMS]
    assert all(i['checkpoint']['step']==10 and i['predicted']==[2.75]*192 for i in images)


@pytest.mark.parametrize('damage',('oldschema','partial','missing10','missing5','prefix','prefixgoal','scope','count','raw',
    'goal','guard','paircheck','admission','prefixNPY','prefiximage','observed','prediction','oldreport','digest','boolslot','planseal'))
def test_failclosed_without_padding_or_old_report_substitution(damage):
    r,e,p,b,s,res,t,pr=fixture();row=r['rows'][0];k=(0,m.ARMS[0])
    if damage=='oldschema':r['schema']=m.parent.physical.NAME+'.report'
    elif damage=='partial':row['full_horizon']=False
    elif damage.startswith('missing'):row['checkpoints']=[x for x in row['checkpoints'] if x['step']!=int(damage[7:])]
    elif damage=='prefix':row['prefix_exact']=False
    elif damage=='prefixgoal':row['prefix_goal_exact']=False
    elif damage=='scope':next(x for x in e if x['kind']=='branch_started')['arm']='other'
    elif damage=='count':r['counts']['physics_step']['completed']-=1
    elif damage=='raw':next(x for x in e if x['kind']=='returned_transition' and x['phase']=='plan')['command']=[0.,0.]
    elif damage=='goal':t[0]['gn_task']['goal_source_sha256']='0'*64
    elif damage=='guard':next(x for x in e if x['kind']=='guard_receipt')['accepted']=False
    elif damage=='paircheck':next(x for x in e if x['kind']=='paired_prefix_exact')['accepted']=False
    elif damage=='admission':next(x for x in e if x['kind']=='suffix_admitted_after_exact_prefix')['index']=10**9
    elif damage=='prefixNPY':pr[k]['latent'][0]+=1.
    elif damage=='prefiximage':pr[k]['checkpoint']['file_sha256']='0'*64
    elif damage=='observed':s[k]['observed_context_sha256']='0'*64
    elif damage=='prediction':res[k]['evaluations'][1]['context']['latent_history'][-1][0]+=1.
    elif damage=='oldreport':s[k]['whole_run_report_sha256']=m.parent.PLANNER_REPORT_SHA
    elif damage=='digest':row['selection_digest']='0'*64
    elif damage=='boolslot':row['root_slot']=False
    else:s[k]['selection_sha256']='0'*64
    with pytest.raises(ValueError):m.assemble(r,e,p,b,s,res,t,pr)


def build_fixture(monkeypatch):
    r,e,p,b,s,res,t,prefix=fixture();h='a'*64
    cp=dict(branches=list(b.values()),bindings={})
    cr=dict(prepared='/RAM/compiler.prepare',prepare_sha256=h,rows=list(res.values()),artifact_sha256={})
    consume=dict(schema=m.compiler.NAME+'.consume',status='observed_suffix_commands_verified_not_admitted',errors=[],plans=list(s.values()),new_physics=0,P1_admission=False,P4_admission=False)
    oldrows=[]
    for k,item in prefix.items():
        slot,arm=k;entry=item['entry']
        oldrows.append(dict(source_slot=slot,arm=arm,role='prefix5',status='encoded',latent={f:entry[f] for f in ('file','sha256','shape','dtype')},
            provenance=dict(checkpoint=item['checkpoint'],planner_selection_sha256=item['planner_selection_sha256'])))
    manifest=dict(schema=m.physical.NAME+'.selection',plans=deepcopy(p['plans']),arms=m.ARMS,
        original_physical_report_sha256=m.parent.PHYSICAL_REPORT_SHA,sourcepack_sha256=m.parent.SOURCEPACK_SHA,
        compiler_authority=dict(source=m.compiler.__file__,source_sha256=m.COMPILER_SHA,report=str(m.COMPILER_REPORT),report_sha256=m.COMPILER_REPORT_SHA,
            consume='/RAM/consume.json',consume_sha256='b'*64))
    saved={str(m.REPORT):r,str(m.REPORT.parent/'execution_binding.json'):dict(prepared='/RAM/physical.prepare',prepare_sha256=m.PHYSICAL_PREPARE_SHA),
        '/RAM/physical.prepare':p,'/RAM/manifest.json':manifest,'/RAM/consume.json':consume,str(m.COMPILER_REPORT):cr,
        str(m.compiler.RGB_REPORT):dict(schema=m.parent.NAME+'.report',status='moving_selected_rgb_encoded_not_admitted',errors=[],model_unchanged=True,
            rows=oldrows,artifact_sha256={})}
    # Preserve distinct saved identities: pointer data and authority must not alias.
    saved['/RAM/consume.json']=deepcopy(consume)
    expected_sha={str(m.compiler.__file__):m.COMPILER_SHA,str(m.physical.__file__):m.PHYSICAL_SHA,str(m.parent.__file__):m.PARENT_SHA,
        str(m.moving.__file__):m.parent.MOVING_SHA,str(m.rgb.__file__):m.parent.RGB_SHA,str(m.rgb.old.__file__):m.rgb.OLD_SHA,
        str(m.REPORT):m.REPORT_SHA,str(m.COMPILER_REPORT):m.COMPILER_REPORT_SHA,str(m.compiler.RGB_REPORT):m.compiler.RGB_REPORT_SHA,
        '/RAM/physical.prepare':m.PHYSICAL_PREPARE_SHA,'/RAM/consume.json':'b'*64,
        str(m.REPORT.parent/'journal.jsonl'):r['journal_sha256'],**m.enc.MODELS,**dict(zip(m.parent.EXPORTS,m.parent.EXPORT_SHAS))}
    for row in r['rows']:
        c=row['checkpoints'][-1];expected_sha[str(m.REPORT.parent/c['png'])]=c['file_sha256']
    # Bind prefixes under their original RGB output, with real entry shape/dtype metadata.
    for k,item in prefix.items():
        entry=item['entry'];entry['path']=str(m.compiler.RGB_REPORT.parent/entry['file'])
        b[k]['prefix_npy']=deepcopy(entry);expected_sha[entry['path']]=entry['sha256']
    monkeypatch.setattr(m,'sha',lambda path:expected_sha.get(str(path),h))
    def read(path,expected=None):
        assert 'sourcepack' not in str(path) and 'certificate' not in str(path)
        return deepcopy(saved[str(path)])
    monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m.compiler,'consume',lambda *a:deepcopy(consume))
    monkeypatch.setattr(m.compiler,'verify',lambda *a:deepcopy(cp))
    def load_npy(path,entry):
        item=next(x for x in prefix.values() if x['entry']['path']==str(path))
        return np.asarray(item['latent'],np.float32),b'RAM'
    monkeypatch.setattr(m.compiler,'load_npy',load_npy)
    monkeypatch.setattr(m.moving,'load_task',lambda path,h:deepcopy(t[m.parent.EXPORTS.index(path)]))
    monkeypatch.setattr(m.physical.base,'journal',lambda *a:deepcopy(e))
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    monkeypatch.setattr(m.moving,'source_files',lambda:[])
    monkeypatch.setattr(m.physical,'load_original',lambda *a:pytest.fail('no privileged physical loader'))
    return saved,p,cp


def test_build_new_compiler_and_four_terminal_receipts(monkeypatch):
    build_fixture(monkeypatch);p=m.build()
    assert p['schema']==m.NAME+'.prepare' and len(p['images'])==4
    assert p['contract']['encoder']==p['contract']['projector']==4 and p['contract']['new_prefix_encodes']==0
    assert p['contract']['output_cap_bytes']==8*1024**2 and p['contract']['disk_floor_bytes']==200*1024**2


@pytest.mark.parametrize('damage',('binding','authority','commands','source','compilerstatus'))
def test_build_rejects_wrong_new_authority_before_encoding(damage,monkeypatch):
    saved,p,cp=build_fixture(monkeypatch)
    if damage=='binding':saved[str(m.REPORT.parent/'execution_binding.json')]['prepare_sha256']='0'*64
    elif damage=='authority':saved['/RAM/manifest.json']['compiler_authority']['report_sha256']=m.parent.PLANNER_REPORT_SHA
    elif damage=='commands':saved['/RAM/consume.json']['plans'][0]['commands'][0][0]=0.
    elif damage=='source':p['sourcepack_sha256']='0'*64
    else:monkeypatch.setattr(m.compiler,'consume',lambda *a:dict(schema='old',status='old',errors=[]))
    with pytest.raises(ValueError):m.build()


def test_only_four_terminal_modules_and_native_NPY(monkeypatch):
    images=m.assemble(*fixture())
    for i in images:i['png_path']='/RAM/terminal.png'
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=RAMStore();ledger=m.enc.Ledger(1,store.event);rows=[dict(status='not_started') for _ in images]
    def encode(a):
        for resource,shape in (('encoder',(1,3,224,224)),('projector',(1,192))):
            ledger.dispatch(resource,NS(device=NS(type='cuda'),shape=shape),lambda:None)
        return np.full(192,4,np.float32)
    m.encode_images(dict(images=images),encode,ledger,store,rows)
    assert ledger.counts()=={k:dict(attempted=4,completed=4) for k in ('encoder','projector')}
    assert len(store.files)==4 and all(np.load(BytesIO(raw),allow_pickle=False).dtype==np.float32 for raw in store.files.values())
    d=rows[0]['metrics'];assert d['actual_Q']==1. and d['predicted_Q']==.0625 and d['prefix_Q']==.25
    assert d['actual_Q_change_from_prefix']==.75 and d['actual_minus_predicted_Q']==.9375


def test_partial_keeps_four_denominator_no_retry(monkeypatch):
    images=m.assemble(*fixture())
    for i in images:i['png_path']='/RAM/terminal.png'
    monkeypatch.setattr(m.rgb.old,'png',lambda *a:np.zeros((224,224,3),np.uint8))
    store=RAMStore();ledger=m.enc.Ledger(1,store.event);rows=[dict(status='not_started') for _ in images]
    def fail(a):
        ledger.dispatch('encoder',NS(device=NS(type='cuda'),shape=(1,3,224,224)),lambda:None)
        raise RuntimeError('mock projector')
    with pytest.raises(RuntimeError):m.encode_images(dict(images=images),fail,ledger,store,rows)
    assert len(rows)==4 and rows[0]['status']=='attempted' and all(x['status']=='not_started' for x in rows[1:]) and not store.files


def test_run_verify_before_output_or_neural(monkeypatch):
    def fail(*a):raise ValueError('saved input')
    monkeypatch.setattr(m,'verify',fail);monkeypatch.setattr(m,'Store',lambda *a:pytest.fail('no output before verify'))
    with pytest.raises(ValueError,match='saved input'):m.run('x','x','x')


def test_source_pins_and_parent_stays_2MiB():
    assert m.sha(m.compiler.__file__)==m.COMPILER_SHA and m.sha(m.physical.__file__)==m.PHYSICAL_SHA
    assert m.sha(m.parent.__file__)==m.PARENT_SHA and m.rgb.CAP==2*1024**2 and m.CAP==8*1024**2


@pytest.mark.parametrize('fault',('floor','cap'))
def test_store8MiB_and_floor_gate_before_write(monkeypatch,fault):
    class Path:
        def is_symlink(self):return False
        def open(self,*a):pytest.fail('must reject before write')
    store=object.__new__(m.Store)
    used=m.CAP-m.FINAL if fault=='cap' else 0
    store.output=NS(iterdir=lambda:[NS(is_file=lambda:True,stat=lambda:NS(st_size=used))])
    monkeypatch.setattr(m.rgb.old,'local',lambda *a:Path())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:NS(free=m.FLOOR if fault=='floor' else m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='cap/floor'):store.write('RAM',b'x')


def test_prepare_disk_reserve_and_verify_reconstruction(monkeypatch):
    class Path:
        parent='RAM'
        def resolve(self):return self
        def exists(self):return False
        def open(self,*a):pytest.fail('prepare disk gate')
    monkeypatch.setattr(m,'build',lambda:dict(images=[]))
    monkeypatch.setattr(m,'Path',lambda *a:Path())
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:NS(free=m.FLOOR))
    with pytest.raises(ValueError,match='200MiB'):m.prepare('RAM')
    monkeypatch.setattr(m,'read',lambda *a:dict(images=[]))
    assert m.verify('RAM','a'*64)==dict(images=[])
    monkeypatch.setattr(m,'build',lambda:dict(images=[],changed=True))
    with pytest.raises(ValueError):m.verify('RAM','a'*64)
