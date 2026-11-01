"""Saved-shaped synthetic receipts/NPYs only, not production fullstrict.

Only this NEW module is reduced to one branch for end-to-end fixture IO.
External complete receipts below are explicit fixtures, not real load evidence.
"""
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_short_start_comparison_20261029_v1 as m


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'inner';(root/'results').mkdir(parents=True)
    for folder,name in (('tests','test_'+m.NAME+'.py'),('innerdocs',m.NAME+'.md')):
        (root/folder).mkdir();(root/folder/name).write_bytes(b'fixture-only bound delivery metadata')
    src=root/'experiments/spectral_reachability';src.mkdir(parents=True)
    pins={}
    for stem in (m.H1,m.H2,m.SHORT):
        p=src/(stem+'.py');p.write_bytes(('fixture source '+stem).encode());pins[stem]=m.sha(p)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'N',1);monkeypatch.setattr(m,'schedule',lambda:[]);monkeypatch.setattr(m,'SOURCE_PINS',pins)
    fixed=deepcopy(m.FIXED);monkeypatch.setattr(m,'FIXED',fixed)
    proto=deepcopy(m.short.protocol());proto.update(branches=1,pairs=[],predictor=dict(api=4,batch_samples=4,frame_samples=9),native_IO_bytes=7373160,bank_tensor_bytes=409800,label_tensor_bytes=1966080)
    monkeypatch.setattr(m,'expected_short_protocol',lambda:deepcopy(proto))
    # Paths deliberately distinguish plan0 from warmup10 (real producer seam).
    def image(phase,step):return dict(path='/fixture/'+phase+str(step)+'.png',file_sha256=('a' if phase=='warmup' else 'b')*64,
        pixel_sha256=('c' if phase=='warmup' else 'd')*64,phase=phase,step=step,condition='off',overlay_index=0)
    images=[image('warmup',s) for s in (0,5,10)];labels=[image('plan',s) for s in (5,10)]
    ident=dict(index=0,root_slot=0,condition='off',task_id='fixture-task',selection_digest='fixture-plan')
    commands=np.zeros((20,2),np.float32).tolist();packets=np.zeros((4,10),np.float32).tolist();stats={'fixture_stats':True}
    q2=dict(ident,context_global_frames=[0,5,10],successor_global_frames=[15,20],images=images,labels=labels,commands=commands,packets=packets)
    q1=dict(ident,context_global_frames=[5,10,15],successor_global_frame=20,images=[images[1],image('plan',0),labels[0]],label=labels[1],commands=commands[5:],packets=packets[1:])
    model=dict(fixture_state='NOT an actual checkpoint');strict=dict(fixture_schema=True)
    def make(kind):
        stem=m.H1 if kind=='H1' else m.H2 if kind=='H2' else m.SHORT
        d=root/'results'/(stem+'_cuda_MAIN01');d.mkdir();artifacts={};calls=[];recindex=0
        def save(a):
            nonlocal recindex
            p=d/f'tensor_{recindex:05d}.npy';recindex+=1;b=io.BytesIO();np.save(b,a,allow_pickle=False);p.write_bytes(b.getvalue())
            rec=dict(file=p.name,file_bytes=p.stat().st_size,sha256=m.sha(p),shape=list(a.shape),dtype='float32',nbytes=a.nbytes)
            artifacts[p.name]={k:rec[k] for k in ('sha256','file_bytes')};return rec
        f=lambda shape,value=0:np.full(shape,value,np.float32)
        def tree(v):return m.short.codec.tree(v,save,lambda a:isinstance(a,np.ndarray))
        def call(key,args,out,phase='prediction'):
            idx=len(calls);shape=args[0].shape
            units=dict(api=1,batch_samples=shape[0],frame_samples=shape[0] if key=='official_encoder' else shape[0]*shape[1])
            c=dict(index=idx,status='returned',inputs=tree((args,{})),outputs=tree(out),units=units)
            if kind=='short':c['T']=shape[1]
            else:c.update(branch=0,phase=phase,module=key)
            calls.append(c);return c
        if kind!='short':
            context=np.stack([f((256,384),j+1) for j in range(3)])
            enc=call('official_encoder',(f((3,3,224,224)),),dict(x_norm_patchtokens=context))
            call('proprio_encoder',(f((1,3,4)),),f((1,3,1,16)))
            pred=lambda value:call('predictor',(f((1,3,1,16,16,384)),f((1,3,10)),f((1,3,256,16))),
                (f((1,3,256,384),value),None,f((1,3,256,16))))['outputs']['$tuple'][0]['$array']
            if kind=='H2':p15=pred(3)
            p20=pred(4)
            target=f((1,256,384),7) if kind=='H1' else np.stack([f((256,384),4),f((256,384),5)])
            lab=call('official_encoder',(f((len(target),3,224,224)),),dict(x_norm_patchtokens=target),phase='label')
            row=dict(ident,context_patch=enc['outputs']['$dict']['x_norm_patchtokens']['$array'],prediction=p20,
                label_patch=lab['outputs']['$dict']['x_norm_patchtokens']['$array'],score=dict(successor_MSE=9. if kind=='H1' else 1.))
            if kind=='H2':row.update(prediction15=p15);row['score']['prediction15_MSE']=1.
            protocol=dict(caps={k:dict(api=0,batch_samples=0,frame_samples=0) for k in m.short.codec.KEYS},pairs=[])
            for c in calls:
                for axis,v in c['units'].items():protocol['caps'][c['module']][axis]+=v
            counts=dict(attempted=deepcopy(protocol['caps']),returned=deepcopy(protocol['caps']))
            prepared=dict(schema=stem+'.prepare',protocol=protocol,bridge=dict(requests=[q1 if kind=='H1' else q2],stats=stats),bindings={str(src/(stem+'.py')):pins[stem]})
        else:
            record=dict(ident,frames=[0],parent_frames=[0,5,10],image0=images[0],source_normalization=dict(commands=commands,packets=packets,stats=stats),
                refs=dict(encoder_patch=dict(directory=str(old2.d),array=old2.r['rows'][0]['context_patch'])),inherited_authority=dict(report_sha256=old2.sha))
            stages=[]
            for j,t in enumerate((1,2,3,3)):
                c=call('predictor',(f((1,t,1,16,16,384)),f((1,t,10)),f((1,t,256,16))),
                    (f((1,t,256,384),j+2),None,f((1,t,256,16))))
                stages.append(dict(target=(j+1)*5,frames=list(m.short.FRAMES[j]),call=j,prediction=c['outputs']))
            copied=[np.stack([f((256,384),j+1) for j in range(3)]),np.stack([f((256,384),4),f((256,384),5)])]
            row=dict(index=0,identity={k:ident[k] for k in ('root_slot','condition','task_id','selection_digest')},stages=stages,labels=[save(x) for x in copied],score=dict(MSE=[0.]*4))
            ref2=dict(directory=str(old2.d),array=old2.r['rows'][0]['label_patch'],report_json_pointer='/rows/0/label_patch')
            protocol=deepcopy(proto);counts={k:dict(api=4,batch_samples=4,frame_samples=9) for k in ('attempted','returned','validated')}
            prepared=dict(schema=stem+'.prepare',authority=dict(protocol=protocol,rows=[dict(index=0,record=record,labels=[record['refs']['encoder_patch'],ref2])]),
                bindings={str(src/(stem+'.py')):pins[stem]})
        pp=root/'results'/(stem+'_prepare_MAIN01.json');pp.write_bytes(m.raw(prepared));prepared_sha=m.sha(pp)
        e=dict(sequence=0,previous_sha256=None,kind='fixture_saved_terminal');e['event_sha256']=m.short.digest(e)
        journal=d/'journal.jsonl';journal.write_bytes(m.raw(e));artifacts[journal.name]=dict(sha256=m.sha(journal),file_bytes=journal.stat().st_size)
        status={'H1':'fixed164_actual15_conditioned_factual_H1_recorded','H2':'fixed164_root10_open_loop_H2_recorded','short':'all164_short_start_causal_roll_recorded'}[kind]
        report=dict(schema=stem+'.report',status=status,protocol=protocol,rows=[row],pairs=[],calls=calls,counts=counts,
            model_before=model,model_after_predictions=model,model_after=model,strict_loads=strict,errors=[],source_errors=[],elapsed_seconds=1.,memory={'fixture':True},
            artifacts=artifacts,journal_chain_tail=e['event_sha256'],execution_binding=dict(prepared=str(pp),prepare_sha256=prepared_sha),P1_admission=False,P5_admission=False)
        total=sum(x['nbytes'] for c in calls for part in ('inputs','outputs') for x in m.descriptors(c[part]))
        if kind=='short':report.update(tensor_bytes=9339240,inherited_cost_source=dict(report_sha256=old2.sha))
        else:report['actual_tensor_io_bytes']=total
        rp=d/'report.json';rp.write_bytes(m.raw(report));report_sha=m.sha(rp)
        receipt=dict(schema=stem+'.consume',status={'H1':'all164_factual_H1_and80_pairs_verified','H2':'all164_open_loop_H2_and80_pairs_verified','short':'all164_short_start_causal_roll_saved_verified'}[kind],
            report_sha256=report_sha,pairs=0,P1_admission=False,P5_admission=False)
        if kind=='short':receipt.update(branches=1,predictor=counts['validated'])
        else:receipt.update(requests=1,counts=counts)
        cp=root/'results'/(stem+'_consume_MAIN01.json');cp.write_bytes(m.raw(receipt))
        if kind!='short':fixed[kind].update(report_sha=report_sha,prepared_sha=prepared_sha,receipt_sha=m.sha(cp))
        return NS(d=d,r=report,p=prepared,pp=pp,cp=cp,receipt=receipt,sha=report_sha,receipt_sha=m.sha(cp))
    old1=make('H1');old2=make('H2');new=make('short')
    return NS(root=root,h1=old1,h2=old2,new=new,args=dict(short_output=str(new.d),short_report_sha=new.sha,short_receipt=str(new.cp),short_receipt_sha=new.receipt_sha))


def test_saved_end_to_end_same_common_label_rescore_seam_and_resource_axes(case):
    a=m.analyze(**case.args)
    assert a['status']=='all164_and80_common_H2_label_saved_rescored' and not a['P1_admission']
    row=a['rows'][0]
    assert row['common_MSE']==dict(H1_20=1.,H2_15=1.,H2_20=1.,short_5=0.,short_10=0.,short_15=0.,short_20=0.)
    assert row['H1_label_batch_seam']==dict(bitwise_equal=False,maxabs=2.,MSE=4.,old_H1_own_label_MSE=9.,old_H1_common_H2_label_MSE=1.)
    assert row['availability']['H1']=={'5':None,'10':None,'15':None,'20':1.}
    assert row['history_source_seams']['frame10_same_source_record'] is False
    assert row['history_source_seams']['frame10_same_pixel_sha'] is False
    assert a['resources']['H1']['observed_returned']['predictor']==dict(api=1,batch_samples=1,frame_samples=3)
    assert a['resources']['H2']['observed_returned']['predictor']==dict(api=2,batch_samples=2,frame_samples=6)
    assert a['resources']['short']['observed_returned']['predictor']==dict(api=4,batch_samples=4,frame_samples=9)
    assert a['resources']['short']['observed_returned']['official_encoder']==dict(api=0,batch_samples=0,frame_samples=0)
    assert a['resources']['short']['native_tensor_IO_bytes']==7373160 and a['resources']['short']['FLOPs'] is None
    assert a['resources']['short']['reported_AD_calls']==0 and a['resources']['H1']['reported_AD_calls'] is None
    p=case.root/'results'/(m.NAME+'_fixture.json');m.write_new(p,a)
    assert m.read(p,m.sha(p),{})==m.analyze(**a['arguments'])
    assert m.verify_analysis(p,m.sha(p))['status']=='all164_and80_common_label_comparison_recomputed'
    a['rows'][0]['common_MSE']['short_20']=123.
    p.write_bytes(m.raw(a))
    with pytest.raises(ValueError):m.verify_analysis(p,m.sha(p))


@pytest.mark.parametrize('fault',('partial','receipt','absent','denominator','new_source','counts','native_alias','normalizer','action','image15','label_pointer','batch_copy','scalar','old_scalar','model'))
def test_refuses_missing_evidence_or_mismatched_controls(case,fault):
    args=case.args.copy()
    if fault=='absent':args['short_output']=str(case.root/'not_yet_produced')
    elif fault=='receipt':args['short_receipt_sha']='0'*64
    elif fault in ('partial','denominator','counts','native_alias','scalar','model'):
        r=deepcopy(case.new.r)
        if fault=='partial':r['status']='partial_or_failed'
        elif fault=='denominator':r['rows']=[]
        elif fault=='counts':r['counts']['returned']['api']=3
        elif fault=='native_alias':r['calls'][1]['inputs']=r['calls'][0]['inputs']
        elif fault=='scalar':r['rows'][0]['score']['MSE'][3]=1.
        else:r['model_after']={}
        rp=case.new.d/'report.json';rp.write_bytes(m.raw(r));args['short_report_sha']=m.sha(rp)
        c=deepcopy(case.new.receipt);c['report_sha256']=m.sha(rp);case.new.cp.write_bytes(m.raw(c));args['short_receipt_sha']=m.sha(case.new.cp)
    elif fault=='batch_copy':
        rec=case.new.r['rows'][0]['labels'][1];p=case.new.d/rec['file'];p.write_bytes(p.read_bytes()+b'\n')
    elif fault=='old_scalar':
        r=deepcopy(case.h1.r);r['rows'][0]['score']['successor_MSE']=0.
        p=case.h1.d/'report.json';p.write_bytes(m.raw(r));m.FIXED['H1']['report_sha']=m.sha(p)
        c=deepcopy(case.h1.receipt);c['report_sha256']=m.sha(p);case.h1.cp.write_bytes(m.raw(c));m.FIXED['H1']['receipt_sha']=m.sha(case.h1.cp)
    else:
        p=deepcopy(case.new.p);record=p['authority']['rows'][0]['record']
        if fault=='new_source':p['bindings'].clear()
        elif fault=='normalizer':record['source_normalization']['stats']={'wrong':True}
        elif fault=='action':record['source_normalization']['commands'][0][0]=.5
        elif fault=='image15':
            # Alter OLD H1 plan5 identity, leaving target20 untouched.
            p=deepcopy(case.h1.p);p['bridge']['requests'][0]['images'][2]['pixel_sha256']='0'*64
            case.h1.pp.write_bytes(m.raw(p));newsha=m.sha(case.h1.pp);m.FIXED['H1']['prepared_sha']=newsha
            r=deepcopy(case.h1.r);r['execution_binding']['prepare_sha256']=newsha
            rp=case.h1.d/'report.json';rp.write_bytes(m.raw(r));m.FIXED['H1']['report_sha']=m.sha(rp)
            c=deepcopy(case.h1.receipt);c['report_sha256']=m.sha(rp);case.h1.cp.write_bytes(m.raw(c));m.FIXED['H1']['receipt_sha']=m.sha(case.h1.cp)
            with pytest.raises(ValueError):m.analyze(**args)
            return
        else:p['authority']['rows'][0]['labels'][1]['report_json_pointer']='/wrong'
        case.new.pp.write_bytes(m.raw(p));r=deepcopy(case.new.r);r['execution_binding']['prepare_sha256']=m.sha(case.new.pp)
        rp=case.new.d/'report.json';rp.write_bytes(m.raw(r));args['short_report_sha']=m.sha(rp)
        c=deepcopy(case.new.receipt);c['report_sha256']=m.sha(rp);case.new.cp.write_bytes(m.raw(c));args['short_receipt_sha']=m.sha(case.new.cp)
    with pytest.raises((ValueError,FileNotFoundError)):m.analyze(**args)


def test_independent_pair_zero_positive_negative_preserved():
    f=lambda v:np.full((1,256,384),v,np.float32)
    data=lambda v,o:dict(predictions={p:f(v) for p in m.PHASES},observations={t:f(o) for t in (5,10,15,20)},H1_original_label=f(o))
    spec=m.schedule()[0]
    zero=m.pair(spec,data(0,0),data(0,0))
    assert all(x['cosine'] is None and x['observed_zero'] and x['predicted_zero'] for x in zero['phases'].values())
    pos=m.pair(spec,data(0,0),data(1,1));neg=m.pair(spec,data(0,0),data(1,-1))
    assert all(x['cosine_sign']=='positive' for x in pos['phases'].values())
    assert all(x['cosine_sign']=='negative' for x in neg['phases'].values())
    assert all(x['central_observed_L2']>0 for x in pos['phases'].values())


def test_fixed_production_schedule_all80_and_source_pins():
    assert m.N==164 and len(m.schedule())==80 and len({tuple(s.values()) for s in m.schedule()})==80
    assert m.PHASES==dict(H1_20=20,H2_15=15,H2_20=20,short_5=5,short_10=10,short_15=15,short_20=20)
    assert m.sha(m.short.__file__)==m.SOURCE_PINS[m.SHORT]
    assert m.protocol()['upstream_fullstrict_rebuilt'] is False and m.protocol()['new_neural_calls']==0


def test_all164_80_nulls_preserved_in_fixed_stratified_summaries():
    rows=[dict(index=i,root_slot=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41][0],condition=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41][1],
        common_MSE={p:0. for p in m.PHASES},contrasts={a+'__to__'+b:0. for a,b in m.COMPARISONS}) for i in range(164)]
    pairs=[dict(spec,phases={p:dict(observed_zero=True,predicted_zero=True,cosine_sign='null') for p in m.PHASES},
        signs={a+'__to__'+b:['null','null'] for a,b in m.COMPARISONS}) for spec in m.schedule()]
    s=m.summaries(rows,pairs)
    assert s[0]['branches']==164 and s[0]['pairs']==80
    assert all(x['branches']==41 and x['pairs']==20 for x in s[1:])
    assert all(x['observed_zero']==x['predicted_zero']==80 and x['signs']==dict(positive=0,negative=0,zero=0,null=80) for x in s[0]['responses'].values())
    assert all(x['null__to__null']==80 and sum(x.values())==80 for x in s[0]['sign_transitions'].values())


def test_no_model_physics_imports_and_CLI_only_saved():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,check=True)
    r=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],cwd=m.ROOT,text=True,capture_output=True,check=True)
    assert '{analyze,verify}' in r.stdout


def test_bound_writer_rejects_overwrite_and_wrong_scope(case):
    p=case.root/'results'/(m.NAME+'_small.json');m.write_new(p,dict(status='fixture'))
    with pytest.raises(ValueError):m.write_new(p,dict(status='fixture'))
    with pytest.raises(ValueError):m.write_new(case.root/'results'/'MAIN.json',dict(status='fixture'))
    with pytest.raises(ValueError):m.raw(dict(value=float('nan')))
