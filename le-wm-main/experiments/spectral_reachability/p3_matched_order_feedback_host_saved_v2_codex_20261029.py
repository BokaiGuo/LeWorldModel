"""Independent saved four-slot HOST consumer; no model or physical execution."""
import io
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_numeric_bridge_codex_20261029 as bridge
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_saved_codex_20261029 as encoder
from experiments.spectral_reachability import p3_matched_order_feedback_context_saved_codex_20261029 as contexts
from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as meter

need=host.need
core=host.life.seam.core
design=host.life.seam.design


def _native_lifecycle(events, episode):
    counts={k:0 for k in meter.CAPS}; stack=[]; closed=False; stopped=False
    for e in events:
        r=e['resource']; d=e['detail']
        if r in ('native_control_return','native_warmup_return'):
            need(type(d.get('terminated')) is bool and type(d.get('truncated')) is bool,'native stop types')
            stopped |= d['terminated'] or d['truncated']
        if r=='native_initial_native_evaluation':
            need(type(d.get('success')) is bool,'initial success type');stopped |= d['success']
        if not r.startswith('native_meter/'):continue
        name=r.removeprefix('native_meter/');kind=e['kind']
        need(name in meter.CAPS and kind in ('intent','return') and
             set(d)=={'episode_id','region','units'} and d['episode_id']==episode and
             type(d['units']) is int and d['units']==1 and d['region'] in meter.REGIONS[name], 'native lifecycle ABI')
        identity=(e['token'],name,d['region'])
        if kind=='intent':
            need(not closed and not (stopped and name=='public_step'),'no native dispatch after stop or close')
            counts[name]+=1;need(counts[name]<=meter.CAPS[name],'native independent cap');stack.append(identity)
        else:
            need(stack and stack.pop()==identity,'native nested return')
            if name=='close':closed=True
    need(not stack and not stopped and closed and counts['constructor']==1 and counts['close']==1 and
         counts['reset']==2 and counts['public_step']==30 and counts['goal_bind']==1,'complete native lifecycle')
    return counts


def _artifacts(directory,events,report,partial=False):
    root=Path(directory);files=list(root.iterdir())
    need(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<=host.storage.CAP,
         'flat bounded artifact inventory')
    refs={};charged={k:0 for k in host.storage.LIMITS}
    def walk(v):
        if type(v) is dict:
            if set(v) in ({'file','sha256','shape','dtype','origin','role'}, {'file','sha256','pixel_sha256','shape','dtype'}):
                name=v['file'];need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat artifact name')
                need(name not in refs or refs[name]==v,'immutable descriptor');refs[name]=v
            else:
                for x in v.values():walk(x)
        elif type(v) is list:
            for x in v:walk(x)
    for e in events:
        charged[e['origin']+'_events']+=len(host.storage.base.raw(e))+1;walk(e['detail'])
    for name,d in refs.items():
        if 'pixel_sha256' in d:
            need(name.endswith('.png') and d['shape']==[224,224,3] and d['dtype']=='uint8','PNG descriptor')
            raw=host.storage.base.read_bytes(root/name,host.storage.publication.PNG_CAP)
            need(host.storage.base.fsha(raw)==d['sha256'] and host.storage.publication.png_data(raw)==d['pixel_sha256'],'PNG hash/pixels')
            charged['physical_png']+=len(raw);continue
        origin,role,shape,dtype=d['origin'],d['role'],d['shape'],d['dtype']
        need(origin in ('encoding','numerical') and type(shape) is list and all(type(x) is int and x>=0 for x in shape)
             and dtype in ('<f4','|u1') and name.endswith('.npy'),'array descriptor')
        if origin=='encoding':need((role,tuple(shape),dtype) in host.storage.base.ENC_SHAPES,'encoding shape')
        else:need(dtype=='<f4' and np.prod(shape,dtype=object)<=768,'numeric shape')
        n=int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize
        raw=host.storage.base.read_bytes(root/name,n+4096);need(host.storage.base.fsha(raw)==d['sha256'],'array SHA')
        stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream);need(version in ((1,0),(2,0)),'NPY version')
        read=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        actual,fortran,dt=read(stream)
        need(list(actual)==shape and not fortran and dt.str==dtype and len(raw)-stream.tell()==n,'NPY header')
        stream.seek(0);a=np.load(stream,allow_pickle=False);need(stream.tell()==len(raw) and np.isfinite(a).all(),'NPY exact finite')
        charged[origin+'_arrays']+=len(raw)
    terminals={'report.json','completion.json'}|({'events.jsonl'} if (root/'events.jsonl').exists() else set())
    orphan={f.name for f in files}-(set(refs)|terminals)
    need(partial or not orphan,'all and only referenced output')
    need(all(name.startswith('a') and name.endswith('.npy') and name[1:-4].isdigit() or name.endswith('.png') for name in orphan),'partial orphan names')
    reported=report['category_bytes']
    need(set(charged)==set(reported) and all(type(reported[k]) is int and charged[k]<=reported[k]<=host.storage.LIMITS[k] for k in charged),'category accounting')
    orphan_bytes=sum((root/n).stat().st_size for n in orphan)
    need(sum(reported.values())-sum(charged.values())==orphan_bytes,'orphan byte accounting')
    return dict(referenced=len(refs),orphans=[dict(file=n,size=(root/n).stat().st_size,sha256=host.source.host.sha(root/n)) for n in sorted(orphan)])


def _events_with_positions(events,slot):
    selected=[e for e in events if e['origin']=='numerical' and e['resource']=='matched_numeric_chunk' and e['detail'].get('slot')==slot]
    values=bridge.reconstruct(events,slot);positions=[];i=0
    for value in values:
        n=selected[i]['detail']['total'];positions.append((selected[i]['index'],selected[i+n-1]['index']));i+=n
    need(i==len(selected),'complete numeric chunk positions')
    need(all(v['kind']=='arm_event' and v['slot']==slot for v in values),'fixed arm wrapper')
    return values,positions


def _native_calls(wrapped,result):
    events=[x['payload'] for x in wrapped]
    host.prior.resource_census(result['resource_receipt'],events)
    calls=[];ids=[]
    for e in wrapped:
        cid=e['call']
        if cid is None:continue
        if not ids or ids[-1]!=cid:
            need(type(cid) is int and cid==len(ids),'native call sequence');ids.append(cid);calls.append([])
        calls[-1].append(e['payload'])
    need(calls and all(host.prior.audit_native(c) for c in calls),'actual saved native IO audit')
    return calls


def _block_math(root,result,observations,calls):
    context=root['root_context'];remaining=[1,3];call_index=0
    for stage,block in enumerate(result['blocks']):
        task=design.parent.parent_task(root,context)
        need(core.digest(context)==block['context_sha256'],'block actual context')
        for h,response in zip((1,3),block['responses'],strict=True):
            call=calls[call_index];call_index+=1
            need(call[0]['method']=='signal' and call[0]['task']==task and call[0]['request']['horizon']==h and
                 call[-1]['result']==response,'paid signal matches saved block')
        diagnostic=design.signal.spectra(task,block['responses'])
        need(diagnostic==block['diagnostic'],'independent signal spectra')
        node,signals,_=design.signal.runtime_signal(task,block['responses'],diagnostic)
        decision=core.choose(task,node,signals,'adaptive')
        need(decision==block['decision'],'independent decision')
        h=design.choose(result['policy'],decision,remaining,stage);remaining.remove(h)
        need(h==block['horizon'],'fixed remaining horizon')
        ledger=core.Ledger(design.expected());ledger.scope=dict(phase='saved_recompute')
        bank=design.frozen.generate(task,block['responses'][0 if h==1 else 1],'endpoints',ledger)
        need(bank==block['bank'] and len(block['candidates'])==4,'independent original bank')
        for j,c in enumerate(block['candidates']):
            need(c['index']==j and c['actions']==bank['actions'][j] and c['whole']==c['segmented'],'candidate identity')
            for segmented,path in ((False,c['whole']),(True,c['segmented'])):
                call=calls[call_index];call_index+=1
                need(call[0]['method']=='replay' and call[0]['task']==task and
                     call[0]['request']==dict(parent_context=context,actions=c['actions'],segmented=segmented) and
                     call[-1]['result']==path,'paid candidate replay')
            terminal=core.checked_path(task,context,c['actions'],c['whole'])
            need(c['cost']==core.model_cost(task,terminal),'independent candidate cost')
        winner=min(block['candidates'],key=lambda c:(c['cost'],c['index']))
        need(block['winner_index']==winner['index'] and block['actions']==winner['actions'],'model winner')
        call=calls[call_index];call_index+=1
        need(call[0]['method']=='replay' and call[0]['task']==task and
             call[0]['request']==dict(parent_context=context,actions=winner['actions'],segmented=False) and
             call[-1]['result']==winner['whole'],'paid independent winner replay')
        previous=np.asarray(observations['warm'][-5:],dtype=np.float32) if stage==0 else previous
        for packet in [x for x in result['packets'] if x['stage']==stage]:
            got=np.asarray(packet['raw'],dtype=np.float32);z=observations[packet['end']]['latent']
            context=host.life.seam.join(context,previous,got,got,z,
                np.asarray(root['action_mean'],np.float32),np.asarray(root['action_std'],np.float32));previous=got
    need(call_index==len(calls) and not remaining,'all paid calls consumed')


def _slot(root,source_root,row,wrapped,positions,events,directory):
    slot=row['slot'];ident=f'matched_slot_{slot:02}'
    values=[x['payload'] for x in wrapped];final=[x['result'] for x in values if x['kind']=='slot_final']
    need(len(final)==1 and core.digest(final[0])==row['result_sha256'],'exact saved slot result')
    result=final[0];need(result['status']=='callback_feedback_complete_not_certified','complete callback only')
    start=next(e['index'] for e in events if e['resource']=='matched_slot_attempt' and e['detail']['slot']==slot)
    stop=next(e['index'] for e in events if e['resource']=='matched_slot_return' and e['detail']['slot']==slot)
    need(start<stop and all(start<a<=b<stop for a,b in positions),'numeric events inside slot')
    physical=[e for e in events if start<e['index']<stop and e['origin']=='physical']
    _native_lifecycle(physical,ident)
    observations=[e for e in physical if e['resource']=='native_fresh_observation' and e['detail'].get('episode_id')==ident]
    warm=[e for e in observations if e['detail']['receipt']['phase']=='warmup']
    plan=[e for e in observations if e['detail']['receipt']['phase']=='plan']
    need([e['detail']['receipt']['step'] for e in warm]==list(range(11)) and
         [e['detail']['receipt']['step'] for e in plan]==list(range(21)),'all native warm10 and plan20 observations')
    traces={(x['receipt']['step'],x['receipt']['overlay_index']):x['receipt'] for x in source_root['warm_trace']}
    for e in warm:
        obs=e['detail']['receipt'];ref=traces[(obs['step'],0)]
        need(obs['pixel_sha256']==ref['pixel_sha256'] and obs['snapshot']==ref['snapshot'] and obs['state']==ref['state'],'actual warm source')
    root_guard=[e for e in physical if e['resource']=='native_matched_root_accepted']
    need(len(root_guard)==1 and root_guard[0]['detail']['root_sha256']==root['binding']['root_sha256'],'live source root guard')
    commands=[e for e in physical if e['resource']=='native_authorized_dispatch']
    returns=[e for e in physical if e['resource']=='native_control_return']
    need(len(commands)==len(returns)==20,'complete public native action denominator')
    for j,(a,b) in enumerate(zip(commands,returns,strict=True)):
        command=np.asarray(result['packets'][j//5]['raw'][j%5],dtype='<f4')
        d=a['detail'];r=b['detail'];obs=plan[j+1]
        need(d['episode_id']==r['episode_id']==ident and d['step']==r['step']==j+1 and
             d['raw_bits']==r['raw_bits']==command.tobytes().hex() and
             a['index']<obs['index']<b['index'] and not r['terminated'] and not r['truncated'],
             'actual ordered public dispatch return and current observation')
    encoded=[e for e in events if start<e['index']<stop and e['origin']=='encoding' and e['resource']=='encoding_complete']
    need([e['detail']['boundary'] for e in encoded]==[5,10,15,20],'all actual encoding boundaries')
    for boundary in (5,10,15,20):
        ops=[e for e in events if start<e['index']<stop and e['origin']=='encoding' and e['detail'].get('boundary')==boundary]
        need([(e['resource'],e['kind']) for e in ops]==[('encoder','intent'),('encoder','return'),
             ('projector','intent'),('projector','return'),('encoding_complete','record')], 'complete encoder hook lifecycle')
    obs_latents=[];lookup={'warm':source_root['packets']['off']['raw_action_history']}
    numeric_positions=[(x['payload'],pos) for x,pos in zip(wrapped,positions,strict=True)]
    for e in encoded:
        detail=e['detail'];boundary=detail['boundary'];frame=detail['frame'];obs=plan[boundary]['detail']['receipt']
        need(Path(obs['path']).name==frame['file'] and obs['file_sha256']==frame['sha256'] and
             obs['pixel_sha256']==frame['pixel_sha256'] and returns[boundary-1]['index']<e['index'], 'same fresh frame and encoding')
        encoder.inspect(directory,{k:frame[k] for k in ('file','sha256','pixel_sha256')},detail['arrays'],boundary)
        descriptor=detail['arrays']['latent'];raw=host.storage.base.read_bytes(Path(directory)/descriptor['file'],4096+192*4)
        z=np.load(io.BytesIO(raw),allow_pickle=False);item=dict(boundary=boundary,latent=z,rgb_sha256=frame['sha256'])
        obs_latents.append(item);lookup[boundary]=item
        ctx=[(v,p) for v,p in numeric_positions if v['kind']=='actual_context' and v['end']==boundary]
        pre=[(v,p) for v,p in numeric_positions if v['kind']=='packet_return' and v['end']==boundary]
        need(len(ctx)==len(pre)==1 and pre[0][1][1]<e['index']<ctx[0][1][0], 'return then encoding then observed context')
    contexts.audit(root,lookup['warm'],result,values,obs_latents)
    calls=_native_calls(wrapped,result);_block_math(root,result,lookup,calls)
    return dict(slot=slot,physical_steps=20,encoding_boundaries=4,native_calls=len(calls),
                observed_context_rebuilt=True,known_root=True,scientific_pass=False)


def consume(output,report_sha,completion_sha):
    directory=Path(output).absolute();need(directory.resolve()==directory and directory.is_dir() and not directory.is_symlink(),'canonical output')
    report=host.source.host.design.read_pinned(directory/'report.json',report_sha)
    completion=host.source.host.design.read_pinned(directory/'completion.json',completion_sha)
    need(report['schema']==host.NAME+'.report' and completion['schema']==host.NAME+'.completion' and
         completion['report_sha256']==report_sha and completion['boundary']=='after_report_fsync' and
         completion['scientific_pass'] is False and report['scientific_pass'] is False,'terminal binding')
    prepared=host.verify(report['prepared'],report['prepare_sha256'])
    original=host.source.host.verify_prepare(host.source.PREP,host.source.PREP_SHA)
    data=host.source.physical.source_material()
    if report['journal_sha256'] is None:
        need(not (directory/'events.jsonl').exists() and report['event_count']==0 and report['journal_tail'] is None,'empty failed journal')
        journal=dict(events=[],pending={},transport_complete=False)
    else:journal=host.storage.inspect_journal(directory/'events.jsonl',report['journal_sha256'])
    events=journal['events']
    need(len(events)==report['event_count'] and (events[-1]['sha256'] if events else None)==report['journal_tail'],'terminal journal')
    pending=[dict(origin=k[0],token=k[1],resource=v) for k,v in sorted(journal['pending'].items())]
    need(pending==report['pending'],'pending exact')
    inventory=_artifacts(directory,events,report,partial=report['status']=='partial_or_failed')
    rows=report['rows'];need(len(rows)==4 and all((r['slot'],r['root'],r['policy'])==ident for r,ident in zip(rows,host.life.seam.ROSTER,strict=True)), 'fixed roster')
    returned=[e['detail'] for e in events if e['resource']=='matched_slot_return']
    need(all(r==rows[r['slot']] for r in returned) and len({r['slot'] for r in returned})==len(returned),'preserved actual terminal rows')
    need(report['status'] in ('four_complete_candidate','partial_or_failed'),'known terminal status')
    complete=report['status']=='four_complete_candidate'
    need(completion['complete'] is complete,'completion and report status')
    if not complete:
        return dict(schema=host.NAME+'.consume',status='partial_preserved_not_certified',rows=rows,
                    report_sha256=report_sha,completion_sha256=completion_sha,known_roots=2,
                    inventory=inventory,full_saved_strict=False,scientific_pass=False)
    need(len(returned)==4 and not report['errors'] and not pending and journal['transport_complete'] and
         report['sources_verified'] is True and report['hooks_restored'] is True and report['model_before']==report['model_after'] and
         report['classes_before']==report['classes_after'] and report['precision']==[prepared['precision']]*2 and
         report['model_before']['original_state_dict']==original['expected_model_hash'] and
         all(r['status']=='callback_feedback_complete_not_certified' and r['cleanup']=='returned' and r['error'] is None for r in rows) and
         type(report['coordination']) is str and len(report['coordination'].strip())>=8 and
         type(report['elapsed_seconds']) in (int,float) and 0<=report['elapsed_seconds']<=host.storage.WALL and
         report['known_roots']==2 and report['fresh_roots']==0 and report['training'] is False and
         report['equal_FLOPs'] is False, 'complete model and source guard')
    gpu=report['gpu'];need(type(gpu) is dict and gpu.get('device')=='cuda:0' and gpu.get('allocation_sum')==4.0 and gpu.get('visible_nvidia_nodes'), 'host CUDA allocation')
    peak=report['peak_bytes'];need(type(peak) is list and len(peak)==2 and all(type(x) is int and x>=0 for x in peak) and
         peak[0]<=prepared['cuda_allocated_bytes'] and peak[1]<=prepared['cuda_reserved_bytes'], 'GPU peak bound')
    audits=[]
    for row in rows:
        wrapped,positions=_events_with_positions(events,row['slot'])
        audits.append(_slot(original['tasks'][row['root']],data['roots'][row['root']],row,wrapped,positions,events,directory))
    return dict(schema=host.NAME+'.consume',status='four_saved_feedback_engineering_verified_known_roots',
                report_sha256=report_sha,completion_sha256=completion_sha,slots=audits,known_roots=2,fresh_roots=0,
                full_saved_strict=True,scientific_pass=False,equal_FLOPs=False,
                limitation='source-trusted saved execution and recorded hooks; not attestation, fresh roots or equal total compute')
