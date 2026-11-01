"""Rebuild completed20 physical-frame/encoder/later-node joins from saved IO.

This does not certify native physics, source authority, partial episodes, or
the experiment. The caller must separately verify authority and native ABI.
"""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_e3_published_nodes_saved_20261029_v1 as numeric
from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as encoder
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v2 as join

u=numeric.pub.base
need=u.need


def consume(output,report_sha,completion_sha,body,prefix5,first5,episode_id):
    """Completed paths only; no claim of native or neural parity."""
    p=Path(output);need(numeric.pub.consume(p,report_sha,completion_sha)['complete'],'complete transport')
    rb=u.read_bytes(p/'report.json',numeric.pub.FINAL_CAP);need(u.fsha(rb)==report_sha,'stable report')
    journal=u.read_bytes(p/'events.jsonl',u.CAP)
    need(u.fsha(journal)==u.decode(rb)['journal_sha256'],'stable full journal')
    events=[u.decode(line) for line in journal.splitlines()]
    def one(resource, boundary):
        found=[e for e in events if e['origin']=='physical' and e['resource']=='native_'+resource
               and e['detail'].get('boundary')==boundary]
        need(len(found)==1,'unique '+resource+' boundary');return found[0]
    def rawbits(value):
        a=np.asarray(value,dtype=np.float32);need(a.shape==(5,2) and np.isfinite(a).all() and (abs(a)<=1).all(),'raw5')
        return a
    b=deepcopy(body);numeric.m.bridge.host.numerical.validate_body(b)
    prior=rawbits(prefix5);action=rawbits(first5);bodies=[];node_raw=[];last=-1
    for start in (5,10,15):
        intent=one('e3_packet_intent',start);ret=one('e3_packet_return',start)
        need(last<intent['index']<ret['index'] and intent['detail']['raw_bits']==action.tobytes().hex()
             and ret['detail']['end']==start+5 and ret['detail']['stopped'] is False
             and ret['detail']['returned_bits']==action.tobytes().hex(),'actual packet action and chronology')
        for j in range(5):
            step=start+j+1
            sent=[e for e in events if e['origin']=='physical' and e['resource']=='native_authorized_dispatch'
                  and e['detail'].get('step')==step]
            got=[e for e in events if e['origin']=='physical' and e['resource']=='native_control_return'
                 and e['detail'].get('step')==step]
            need(len(sent)==len(got)==1 and intent['index']<sent[0]['index']<got[0]['index']<ret['index']
                 and sent[0]['detail']['raw_bits']==got[0]['detail']['raw_bits']==action[j].tobytes().hex(),
                 'per-step literal dispatched/returned action')
        last=ret['index']
        if start==15:break
        boundary=start+5;ei=one('e3_encode_intent',boundary);er=one('e3_encode_return',boundary)
        ni=one('e3_node_intent',boundary);nr=one('e3_node_return',boundary)
        complete=[e for e in events if e['origin']=='encoding' and e['kind']=='record'
                  and e['resource']=='encoding_complete' and e['detail'].get('boundary')==boundary]
        need(len(complete)==1 and last<ei['index']<complete[0]['index']<er['index']<ni['index']<nr['index'],
             'actual observation/encoding/node chronology')
        module_events=[e for e in events if e['origin']=='encoding' and e['kind'] in ('intent','return','failure')
                       and e['detail'].get('boundary')==boundary]
        need([(e['resource'],e['kind']) for e in module_events]==[
            ('encoder','intent'),('encoder','return'),('projector','intent'),('projector','return')]
            and ei['index']<module_events[0]['index']<module_events[-1]['index']<complete[0]['index'],
            'exact encoder/projector actual lifecycle')
        observed=ei['detail']['observation'];frame=complete[0]['detail']['frame']
        need(observed['step']==boundary and observed['path']==str(p/frame['file'])
             and observed['file_sha256']==frame['sha256'] and observed['pixel_sha256']==frame['pixel_sha256'],
             'same physical RGB frame')
        seen=[e for e in events if e['origin']=='physical' and e['resource']=='native_fresh_observation'
              and e['detail'].get('episode_id')==episode_id and e['detail'].get('receipt')==observed]
        need(len(seen)==1 and intent['index']<seen[0]['index']<ret['index'],'actual fresh frame event')
        png=u.read_bytes(p/frame['file'],numeric.pub.PNG_CAP);need(u.fsha(png)==frame['sha256'],'frame stable')
        arrays={}
        for role,d in complete[0]['detail']['arrays'].items():
            need(d['origin']=='encoding' and d['role']==role,'owned exact encoder tensor role')
            blob=u.read_bytes(p/d['file'],2*1024**2);need(u.fsha(blob)==d['sha256'],'encoder tensor stable SHA')
            stream=io.BytesIO(blob);a=np.load(stream,allow_pickle=False)
            need(stream.tell()==len(blob) and list(a.shape)==d['shape'] and a.dtype.str==d['dtype'], 'saved tensor ABI')
            arrays[role]=a
        encoder.verify_arrays(png,arrays)
        latent=arrays['latent'];need(np.asarray(er['detail']['latent'],np.float32).tobytes()==latent.tobytes()
            and er['detail']['observation_sha']==frame['sha256'],'encoded latent joined to physical frame')
        context=deepcopy(b['root_context'])
        for key in ('latent_history','action_history'):context[key]=np.asarray(context[key],np.float32)
        value=join.joined_context(context,prior,action,action,latent,np.asarray(b['action_mean'],np.float32),
            np.asarray(b['action_std'],np.float32),expected_rgb_sha=frame['sha256'],rgb_bytes=png,
            encoding_input_sha=frame['sha256'],episode_id=episode_id,physical_episode_id=episode_id,
            encoder_episode_id=episode_id,start=start,end=boundary,physical_end=boundary,encoding_boundary=boundary,stopped=False)
        b['root_context']=dict(value,latent_history=value['latent_history'].tolist(),action_history=value['action_history'].tolist())
        b['feedback_boundary']=boundary;digest=numeric.m.bridge.host.digest(b)
        need(ni['detail']['body_sha256']==nr['detail']['body_sha256']==digest,'independent actual node body')
        next_raw=bytes.fromhex(nr['detail']['raw_bits']);need(len(next_raw)==40,'node returned raw5')
        bodies.append(deepcopy(b));node_raw.append(next_raw);prior=action.copy()
        action=np.frombuffer(next_raw,dtype='<f4').copy().reshape(5,2);rawbits(action);last=nr['index']
    nodes=numeric.consume(bodies,p,report_sha,completion_sha)
    for n,bits in zip(nodes['nodes'],node_raw,strict=True):
        need(rawbits(n['result']['first5']).tobytes()==bits,'saved replay selects actual next raw')
    return dict(completed20_feedback_io_joined=True,full_native_strict=False,
                model_parity_recomputed=False,scientific_admission=False)
