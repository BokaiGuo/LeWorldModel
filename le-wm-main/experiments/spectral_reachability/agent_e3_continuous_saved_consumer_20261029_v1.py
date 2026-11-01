"""Saved E3 physical trace reconstruction; candidate, not full strict."""
import math
import argparse
from copy import deepcopy
import io
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub

need = pub.base.need


def f64(value, shape):
    a=np.asarray(value,dtype=np.float64)
    need(a.shape==shape and np.isfinite(a).all(), 'finite native vector')
    return a


def native(goal,state):
    d=f64(state,(7,))-f64(goal,(7,))
    angle=float(abs(d[4]));angle=min(angle,2*np.pi-angle)
    return bool(float(np.linalg.norm(d[:4]))<20 and angle<np.pi/9),float(np.linalg.norm(d))


def validate_snapshot(snapshot,state):
    """Independently check native7 against both saved nine-field body seals."""
    need(type(snapshot) is dict and set(snapshot)=={'state','state_float64_bits','bodies'}
         and type(snapshot['bodies']) is dict and set(snapshot['bodies'])=={'agent','block'},'snapshot schema')
    s=f64(state,(7,));need(f64(snapshot['state'],(7,)).tobytes()==s.tobytes()
        and snapshot['state_float64_bits']==s.astype('<f8').tobytes().hex(),'snapshot native7 exact bits')
    decoded={}
    for name in ('agent','block'):
        x=snapshot['bodies'][name];need(type(x) is dict and set(x)=={'position','velocity','unwrapped_angle',
            'angular_velocity','force','torque','is_sleeping','float64_bits'} and type(x['is_sleeping']) is bool,
            'native body snapshot ABI')
        position=f64(x['position'],(2,));velocity=f64(x['velocity'],(2,));force=f64(x['force'],(2,))
        for k in ('unwrapped_angle','angular_velocity','torque'):
            need(type(x[k]) in (int,float) and math.isfinite(x[k]),'finite body scalar')
        value=np.asarray([*position,*velocity,x['unwrapped_angle'],x['angular_velocity'],*force,x['torque']],dtype='<f8')
        need(x['float64_bits']==value.tobytes().hex(),'body dynamics bits');decoded[name]=value
    a,b=decoded['agent'],decoded['block']
    need(a[:2].tobytes()==s[:2].tobytes() and b[:2].tobytes()==s[2:4].tobytes()
         and float(b[4]%(2*np.pi))==float(s[4]) and a[2:4].tobytes()==s[5:7].tobytes(),
         'snapshot and native observation consistency')


def validate_physical(events, output, row, setup, first5):
    """Validate observed actions, frames, root/goal and native stop semantics.

    All records must already be transport-authenticated. No physics replay is
    performed. Numerical nodes and source/model guards are separate checks.
    """
    need(type(events) is list and type(row) is dict and type(row.get('episode_id')) is str,
         'trace and episode')
    episode=row['episode_id']; physical=[e for e in events if e['origin']=='physical']
    def select(name):return [e for e in physical if e['resource']=='native_'+name]
    def one(name):
        v=select(name);need(len(v)==1,'unique '+name);return v[0]
    def bits(v,shape):
        a=np.asarray(v,dtype=np.float32);need(a.shape==shape and np.isfinite(a).all(),'action finite shape');return a.tobytes()
    observations=select('fresh_observation'); frames=select('frame'); guards=select('guard_passed')
    need(len(observations)==len(frames)==len(guards)==row['PNG'],'one saved frame per observation/guard')
    obs={};previous_guard=-1
    for fr,ob,gd in zip(frames,observations,guards,strict=True):
        d=ob['detail'];r=d['receipt']; phase=r['phase']; step=r['step']; frame=fr['detail']['frame']
        need(d['episode_id']==episode and gd['detail']['episode_id']==episode and
             previous_guard<fr['index']<ob['index']<gd['index'] and gd['detail']['phase']==phase and gd['detail']['step']==step
             and r['observed_event_index']==ob['index'] and type(step) is int and
             phase in ('warmup','plan') and (phase,step) not in obs, 'capture chronology and identity')
        need(Path(r['path']) == Path(output)/frame['file'] and r['file_sha256']==frame['sha256']
             and r['pixel_sha256']==frame['pixel_sha256'], 'exact physical frame/receipt')
        state=f64(r['state'],(7,))
        validate_snapshot(r['snapshot'],state)
        with Image.open(Path(output)/frame['file']) as image:
            pixels=np.asarray(image)
        need(pixels.shape==(224,224,3) and pixels.dtype==np.uint8,'actual RGB ABI')
        agent_pixels=int(np.count_nonzero(np.all(pixels==[65,105,225],axis=-1)))
        need(agent_pixels>0 and (state[:4]>=0).all() and (state[:4]<=512).all()
             and type(gd['detail']['canonical_agent_pixels']) is int
             and gd['detail']['canonical_agent_pixels']==agent_pixels,'independently recomputed image/canvas guard')
        obs[(phase,step)]=(ob,frame);previous_guard=gd['index']
    warm=select('warmup_return');controls=select('control_return');intents=select('authorized_dispatch')
    need(len(warm)==row['warmup_returned'] and len(controls)==row['control_returned']
         and len(intents)==len(controls)==len(row['executed_raw']) and len(warm)<=10 and len(controls)<=20,
         'returned counts and saved dispatch denominator')
    rawwarm=np.asarray(setup['ref']['raw_history'],dtype=np.float32)
    need(rawwarm.shape==(10,2), 'fixed warm actions')
    warm_stopped=False
    for i,e in enumerate(warm,1):
        d=e['detail'];need(d['episode_id']==episode and d['step']==i and
            bits(d['command'],(2,))==rawwarm[i-1].tobytes() and ('warmup',i) in obs
            and obs['warmup',i][0]['index']<e['index'] and type(d['terminated']) is bool
            and type(d['truncated']) is bool, 'warm action and observation')
        if d['terminated'] or d['truncated']:
            need(i==len(warm) and not controls,'warm native stop');warm_stopped=True
    need(('warmup',0) in obs, 'initial warm observation')
    for j,step in enumerate((0,5,10)):
        if ('warmup',step) not in obs:continue
        record,frame=obs['warmup',step];r=record['detail']['receipt']
        need(f64(r['state'],(7,)).tobytes()==f64(setup['ref']['states'][j],(7,)).tobytes()
             and r['snapshot']==setup['ref']['dynamics'][j], 'fixed warm state and snapshot')
        with Image.open(Path(output)/frame['file']) as image:
            pixels=np.asarray(image)
        need(pixels.dtype==np.uint8 and pixels.tobytes()==np.asarray(setup['ref']['history'][j]).tobytes(),
             'fixed warm RGB bits')
    actions=[np.asarray(setup['prefix5'],np.float32),np.asarray(first5,np.float32)]
    missing_node=False
    for boundary in (10,15):
        found=[e for e in select('e3_node_return') if e['detail']['boundary']==boundary]
        need(len(found)<=1,'one node return per boundary')
        if found:
            need(not missing_node, 'node boundaries in order')
            b=bytes.fromhex(found[0]['detail']['raw_bits']);need(len(b)==40,'node action bytes')
            actions.append(np.frombuffer(b,dtype='<f4').reshape(5,2))
        else:missing_node=True
    goal=f64(setup['new_goal'],(7,));stopped=False
    if controls:
        root=one('actual_root_guard')['detail']
        need(root['episode_id']==episode and root['original_root_sha256']==setup['original_root_sha']
             and f64(root['new_goal'],(7,)).tobytes()==goal.tobytes()
             and root['new_goal_pixel_sha256']==setup['expected_goal_pixel']
             and ('plan',0) in obs,'source-bound native goal and original root')
    for i,(sent,ret) in enumerate(zip(intents,controls,strict=True),1):
        x=sent['detail'];d=ret['detail'];need(i<=len(actions)*5 and not stopped and
            x['episode_id']==d['episode_id']==episode and x['step']==d['step']==i and
            x['packet_boundary']==((i-1)//5)*5 and sent['index']<ret['index'] and
            x['raw_bits']==d['raw_bits']==actions[(i-1)//5][(i-1)%5].tobytes().hex() and
            bits(row['executed_raw'][i-1],(2,)).hex()==d['raw_bits'], 'literal causal action')
        need(('plan',i) in obs and sent['index']<obs['plan',i][0]['index']<ret['index'], 'fresh actual state')
        record=obs['plan',i][0]['detail']['receipt'];success,distance=native(goal,record['state'])
        need(type(d['terminated']) is bool and type(d['truncated']) is bool and
             d['terminated']==success and type(d['reward']) in (int,float) and
             math.isfinite(d['reward']) and d['reward']==-distance, 'original native terminal/reward')
        need(f64(record['goal'],(7,)).tobytes()==goal.tobytes()
             and record['goal_rgb_sha256']==setup['expected_goal_pixel'],'constant bound goal')
        stopped=d['terminated'] or d['truncated']
        if stopped:need(i==len(controls),'no actions after stop')
    if controls:
        final=obs['plan',len(controls)][0]['detail']['receipt']
        need(f64(row['final_state'],(7,)).tobytes()==f64(final['state'],(7,)).tobytes()
             and row['native_success']==native(goal,final['state'])[0], 'terminal native row')
    if len(controls)>=5 and not (stopped and len(controls)==5):
        cp=one('actual_cp5_guard')['detail'];expected=setup['expected_cp5']
        record=obs['plan',5][0]['detail']['receipt']
        need(cp['episode_id']==episode and cp['observation']==record and
             cp['prefix_raw_bits']==np.asarray(setup['prefix5'],dtype=np.float32).tobytes().hex()
             and record['pixel_sha256']==expected['pixel_sha256'] and record['snapshot']==expected['snapshot']
             and f64(record['state'],(7,)).tobytes()==f64(expected['state'],(7,)).tobytes(),
             'actual source-bound cp5')
    status=row.get('status');initial=select('initial_native_evaluation');initial_success=None
    if len(controls)>=5 and not (stopped and len(controls)==5):
        need(len(initial)==1, 'actual initial evaluation')
        d=initial[0]['detail'];s=obs['plan',5][0]['detail']['receipt']['state'];success,distance=native(goal,s)
        need(d['episode_id']==episode and d['boundary']==5 and type(d['success']) is bool
             and d['success']==success and type(d['distance']) in (int,float) and d['distance']==distance
             and row['initial_success']==success and row['initial_native_distance']==distance,
             'independent initial native evaluation')
        initial_success=success
        if len(controls)>5:need(not success,'no dispatch after initial success')
    else:need(not initial,'no premature initial evaluation')
    if warm_stopped:
        need(status=='native_stopped_warmup' and len(warm)<11 and not initial,'actual warm stop status')
    elif stopped:
        need(status==('native_stopped_prefix' if len(controls)<=5 else 'native_stopped')
             and row['terminated']==controls[-1]['detail']['terminated']
             and row['truncated']==controls[-1]['detail']['truncated'],'actual control stop status')
    else:
        need(len(warm)==10 and len(controls)>=5 and initial_success is not None, 'actual initial evaluation')
        if initial_success:
            need(status=='native_initial_success_control5' and len(controls)==5
                 and row['initial_success_control']==5,'actual initial success stop')
        else:need(status=='completed20' and len(controls)==20 and row['initial_success_control'] is None,
                  'complete fixed control denominator')
    return dict(warm_steps=len(warm),control_steps=len(controls),frames=len(frames),
                stopped=warm_stopped or stopped,physical_replay_executed=False,scientific_admission=False)


def validate_feedback(events, output, report_sha, completion_sha, body, prefix5, first5, episode, count):
    """Reconstruct only nodes actually reached; never pad native stops."""
    from experiments.spectral_reachability import agent_e3_published_nodes_saved_20261029_v1 as numeric
    from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as encoder
    from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v2 as join
    p=Path(output);b=deepcopy(body);numeric.m.bridge.host.numerical.validate_body(b)
    prior=np.asarray(prefix5,dtype=np.float32);action=np.asarray(first5,dtype=np.float32)
    need(prior.shape==action.shape==(5,2) and np.isfinite(prior).all() and np.isfinite(action).all(), 'literal feedback inputs')
    expected=[x for x in (10,15) if count>x];bodies=[];raws=[];last=-1
    def one(resource,boundary):
        out=[e for e in events if e['origin']=='physical' and e['resource']=='native_'+resource
             and e['detail'].get('boundary')==boundary]
        need(len(out)==1,'unique feedback '+resource);return out[0]
    for boundary in expected:
        start=boundary-5;pi=one('e3_packet_intent',start);pr=one('e3_packet_return',start)
        ei=one('e3_encode_intent',boundary);er=one('e3_encode_return',boundary)
        ni=one('e3_node_intent',boundary);nr=one('e3_node_return',boundary)
        need(last<pi['index']<pr['index']<ei['index']<er['index']<ni['index']<nr['index']
             and pi['detail']['raw_bits']==pr['detail']['returned_bits']==action.tobytes().hex()
             and pr['detail']['end']==boundary and pr['detail']['stopped'] is False,
             'feedback packet and event chronology')
        complete=[e for e in events if e['origin']=='encoding' and e['resource']=='encoding_complete'
                  and e['kind']=='record' and e['detail'].get('boundary')==boundary]
        mods=[e for e in events if e['origin']=='encoding' and e['kind'] in ('intent','return','failure')
              and e['detail'].get('boundary')==boundary]
        need(len(complete)==1 and bool(mods) and ei['index']<mods[0]['index'],'actual encoder chronology')
        need([(e['resource'],e['kind']) for e in mods]==[('encoder','intent'),('encoder','return'),
             ('projector','intent'),('projector','return')] and mods[-1]['index']<complete[0]['index']<er['index'],
             'encoder projector lifecycle')
        frame=complete[0]['detail']['frame']; observed=ei['detail']['observation']
        need(observed['step']==boundary and observed['path']==str(p/frame['file'])
             and observed['file_sha256']==frame['sha256'] and observed['pixel_sha256']==frame['pixel_sha256'],
             'actual boundary frame')
        seen=[e for e in events if e['origin']=='physical' and e['resource']=='native_fresh_observation'
              and e['detail'].get('episode_id')==episode and e['detail'].get('receipt')==observed]
        need(len(seen)==1 and pi['index']<seen[0]['index']<pr['index'], 'same environment fresh frame')
        png=pub.base.read_bytes(p/frame['file'],pub.PNG_CAP);need(pub.base.fsha(png)==frame['sha256'],'frame stable')
        arrays={}
        for role,d in complete[0]['detail']['arrays'].items():
            need(d['origin']=='encoding' and d['role']==role,'encoder role')
            blob=pub.base.read_bytes(p/d['file'],2*1024**2);need(pub.base.fsha(blob)==d['sha256'],'encoder array SHA')
            stream=io.BytesIO(blob);a=np.load(stream,allow_pickle=False)
            need(stream.tell()==len(blob) and list(a.shape)==d['shape'] and a.dtype.str==d['dtype'],'array ABI')
            arrays[role]=a
        encoder.verify_arrays(png,arrays);latent=arrays['latent']
        need(np.asarray(er['detail']['latent'],np.float32).tobytes()==latent.tobytes()
             and er['detail']['observation_sha']==frame['sha256'],'actual encoded latent')
        context=deepcopy(b['root_context'])
        for key in ('latent_history','action_history'):context[key]=np.asarray(context[key],np.float32)
        value=join.joined_context(context,prior,action,action,latent,np.asarray(b['action_mean'],np.float32),
            np.asarray(b['action_std'],np.float32),expected_rgb_sha=frame['sha256'],rgb_bytes=png,
            encoding_input_sha=frame['sha256'],episode_id=episode,physical_episode_id=episode,
            encoder_episode_id=episode,start=start,end=boundary,physical_end=boundary,
            encoding_boundary=boundary,stopped=False)
        b['root_context']=dict(value,latent_history=value['latent_history'].tolist(),action_history=value['action_history'].tolist())
        b['feedback_boundary']=boundary;digest=numeric.m.bridge.host.digest(b)
        need(ni['detail']['body_sha256']==nr['detail']['body_sha256']==digest,'actual joined task')
        raw=bytes.fromhex(nr['detail']['raw_bits']);need(len(raw)==40,'node raw5')
        bodies.append(deepcopy(b));raws.append(raw);prior=action.copy()
        action=np.frombuffer(raw,dtype='<f4').copy().reshape(5,2);need(np.isfinite(action).all() and (abs(action)<=1).all(),'node action')
        last=nr['index']
    for name in ('e3_encode_intent','e3_encode_return','e3_node_intent','e3_node_return'):
        actual=[e['detail']['boundary'] for e in events if e['origin']=='physical' and e['resource']=='native_'+name]
        need(actual==expected,'no missing/extra feedback boundaries')
    if not bodies:
        need(not any(e['origin'] in ('numerical','encoding') for e in events),'no uncharged later node/encoder')
        return dict(nodes=[])
    result=numeric.consume(bodies,p,report_sha,completion_sha)
    for node,raw in zip(result['nodes'],raws,strict=True):
        need(np.asarray(node['result']['first5'],np.float32).tobytes()==raw,'replayed literal action')
    return result


def consume(output, report_sha, completion_sha, authority_path, authority_sha):
    """Verify pinned transport and available saved evidence; never issue full strict.

    Full native snapshot semantics and failure-path reconstruction remain
    unimplemented; a returned result is deliberately not production approval.
    """
    from experiments.spectral_reachability import agent_e3_continuous_host_launcher_20261029_v1 as launch
    from experiments.spectral_reachability import agent_e3_native_meter_20261029_v1 as meter
    from experiments.spectral_reachability import agent_e3_guard_saved_20261029_v1 as guards
    from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal
    transport=pub.consume(output,report_sha,completion_sha)
    p=Path(output); rb=pub.base.read_bytes(p/'report.json',pub.FINAL_CAP)
    need(pub.base.fsha(rb)==report_sha,'stable report')
    j=pub.base.read_bytes(p/'events.jsonl',pub.base.CAP)
    need(pub.base.fsha(j)==pub.base.decode(rb)['journal_sha256'],'stable journal')
    events=[pub.base.decode(x) for x in j.splitlines()]
    if not transport['complete']:
        starts=[e for e in events if e['resource']=='launcher_started' and e['origin']=='physical']
        failures=[e for e in events if e['resource']=='launcher_failure' and e['origin']=='physical']
        need(len(starts)==1 and starts[0]['index']==0 and len(failures)==1
             and not any(e['resource']=='launcher_guarded_return' for e in events),
             'partial launcher failure chronology')
        detail=failures[0]['detail'];need(set(detail)=={'type','message','traceback'}
            and all(type(x) is str for x in detail.values()),'preserved bounded failure')
        closes=[e for e in events if e['resource']=='native_cleanup_return' and e['origin']=='physical']
        need(len(closes)<=1 and (not closes or closes[0]['index']<failures[0]['index']), 'partial cleanup chronology')
        markers=[e for e in events if e['origin']=='physical' and e['resource']=='launcher_authority_verified']
        need(len(markers)<=1 and (not markers or markers[0]['index']<failures[0]['index']), 'partial authority chronology')
        partial_meter=None;loader_complete=False;authority_verified=False
        if markers:
            binding=launch.read_authority(authority_path,authority_sha)
            need(markers[0]['detail']==dict(cell=binding['cell'],manifest_path=str(Path(authority_path)),
                                           manifest_sha256=authority_sha),'partial actual authority binding')
            authority_verified=True;episode='cell%02d'%binding['cell']
            partial_meter=meter.consume(output,report_sha,completion_sha,episode)
            cleanup_intents=[e for e in events if e['origin']=='physical' and e['resource']=='native_cleanup_intent']
            cleanup_failures=[e for e in events if e['origin']=='physical' and e['resource']=='native_cleanup_failure']
            need(len(cleanup_intents)<=1 and len(cleanup_failures)<=1 and not (closes and cleanup_failures),
                 'partial cleanup unique outcome')
            outcomes=closes+cleanup_failures
            if outcomes:
                need(len(cleanup_intents)==1 and cleanup_intents[0]['index']<outcomes[0]['index']<failures[0]['index']
                     and cleanup_intents[0]['detail'].get('episode_id')==episode
                     and outcomes[0]['detail'].get('episode_id')==episode,'partial cleanup identity and chronology')
            if closes:
                close=partial_meter['counts']['close']
                need(close==dict(attempted=1,completed=1,failed=0),'partial cleanup return requires metered close')
            loader_records=[e for e in events if e['origin']=='physical' and e['resource'] in ('loader_event','loader_receipt')]
            if any(e['resource']=='loader_receipt' for e in loader_records):
                launch.loader_bridge.verify_events(events,binding['prepared'],binding['prepare_sha'],
                                                  launch.host.primitives.native.loader)
                loader_complete=True
            else:
                need(len(loader_records)<=4 and all(e['resource']=='loader_event' and e['kind']=='record'
                     for e in loader_records),'partial loader event prefix')
                sequence=['load_intent','consumed_buffer_frozen','load_intent','consumed_buffer_frozen']
                need(all(e['detail']['event']['kind']==sequence[i] for i,e in enumerate(loader_records)),
                     'partial loader event chronology')
        return dict(schema='agent_e3_continuous_saved_consumer_20261029_v1.inspect',
                    status='partial',error=detail,cleanup_return_observed=bool(closes),
                    reset_trace=launch.reset_identity.inspect_partial(events),
                    authority_verified_here=authority_verified,loader_receipt_verified=loader_complete,
                    meter=partial_meter,full_saved_strict=False,scientific_admission=False,
                    missing=['failure-path source and native semantic reconstruction'])
    binding=launch.read_authority(authority_path,authority_sha)
    def exactly(name):
        x=[e for e in events if e['origin']=='physical' and e['resource']==name]
        need(len(x)==1 and x[0]['kind']=='record','unique '+name);return x[0]
    start=exactly('launcher_started');authority=exactly('launcher_authority_verified')
    end=exactly('launcher_guarded_return')
    need(start['index']==0 and start['detail']==dict(schema=launch.NAME,scientific_admission=False)
         and authority['detail']==dict(cell=binding['cell'],manifest_path=str(Path(authority_path)),
                                       manifest_sha256=authority_sha)
         and start['index']<authority['index']<end['index']==len(events)-1
         and not any(e['resource']=='launcher_failure' for e in events), 'launcher authority and final guard chronology')
    row=terminal.read(events);need(row['episode_id']=='cell%02d'%binding['cell']
        and end['detail']==dict(cell=binding['cell'],status=row['status']),'source-bound terminal status')
    launch.loader_bridge.verify_events(events,binding['prepared'],binding['prepare_sha'],
                                       launch.host.primitives.native.loader)
    setup=launch.episode.old.setup_authority.resolve(binding['resolver']['path'],binding['resolver']['sha256'],
             binding['cell'], launch.episode.old.authority.load(*binding['args'])['body'])
    source=dict(binding['prepared']['bindings_sha256'])
    launch.host.source.merge(source,binding['additional_sources'])
    launch.host.source.merge(source,setup.pop('source_bindings'))
    module_census=launch.module_census.saved(events,source)
    markers={stage:[e for e in events if e['origin']=='physical' and e['resource']=='module_census_'+stage]
             for stage in ('before','after')}
    cleanup=[e for e in events if e['origin']=='physical' and e['resource']=='native_cleanup_return']
    need(len(cleanup)==1 and authority['index']<markers['before'][0]['index']<cleanup[0]['index']
         <markers['after'][0]['index']<end['index'],'module census encloses native lifecycle')
    ep=launch.episode;old=ep.old
    modules=(old,old.authority,old.guard,old.live,old.setup_authority,old.live.setup,old.live.region,
             old.live.capture,old.live.dispatcher,old.live.encoding,old.live.nodes,old.native,meter,ep,launch.reset_identity)
    required=[str(Path(x.__file__).resolve()) for x in modules]
    guard=guards.consume(output,report_sha,completion_sha,source,required)
    native_meter=meter.consume(output,report_sha,completion_sha,row['episode_id'])
    need(native_meter['ledger_complete'], 'native meter and cleanup ledger')
    actual=old.authority.load(*binding['args'])
    reset=launch.reset_identity.verify_events(events,row['episode_id'],setup)
    geometry=launch.reset_identity.verify_geometry(events,source)
    physical=validate_physical(events,p,row,setup,actual['first5'])
    warm0=[e for e in events if e['origin']=='physical' and e['resource']=='native_fresh_observation'
           and e['detail']['receipt']['phase']=='warmup' and e['detail']['receipt']['step']==0]
    need(len(warm0)==1 and launch.reset_identity.vector_bytes(warm0[0]['detail']['receipt']['state'])
         ==reset['reset_state'],'actual reset return matches first warm frame')
    if row['status']=='completed20':
        need(physical['control_steps']==20 and physical['warm_steps']==10 and not physical['stopped'],
             'completed20 exact denominator')
    feedback_result=validate_feedback(events,p,report_sha,completion_sha,actual['body'],setup['prefix5'],
                                      actual['first5'],row['episode_id'],physical['control_steps'])
    return dict(schema='agent_e3_continuous_saved_consumer_20261029_v1.inspect',cell=binding['cell'],
                status=row['status'],physical=physical,guard=guard,meter=native_meter,reset=reset,
                geometry=geometry,module_census=module_census,feedback=feedback_result,
                full_saved_strict=False,scientific_admission=False,
                missing=['complete native snapshot semantic validation','failure-path saved reconstruction'])


def main():
    from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('output','report-sha256','completion-sha256','authority','authority-sha256'):
        p.add_argument('--'+option,required=True)
    a=p.parse_args()
    print(host.raw(consume(a.output,a.report_sha256,a.completion_sha256,a.authority,a.authority_sha256)).decode())


if __name__=='__main__':main()
