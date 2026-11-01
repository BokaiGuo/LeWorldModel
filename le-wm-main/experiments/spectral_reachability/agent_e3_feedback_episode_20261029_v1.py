"""Same-session feedback orchestration primitive; callbacks must be authenticated externally.

This module is NOT a native adapter, CUDA launcher, or saved strict consumer.
"""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v2 as join
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node


def episode(body, cp5_first5, prior_raw5, session, plan, emit):
    """One live session: dispatch5, encode actual 10, plan10, dispatch10, etc.

    session.packet(start, raw5) returns {raw, stopped, end, rgb_bytes}; session.encode
    returns {latent, input_sha, episode_id, boundary}. They are NOT authoritative
    merely because they implement this interface. Caller owns setup, cleanup,
    persistence, source/model guards and authenticity.
    """
    b=deepcopy(body);node.host.numerical.validate_body(b)
    eid=session.episode_id
    node.need(type(eid) is str and bool(eid),'episode identity')
    prior=join.v1.f32(prior_raw5,(5,2)).copy()
    raw=join.v1.f32(cp5_first5,(5,2)).copy()
    rows=[]
    for start in (5,10,15):
        emit(dict(kind='dispatch_intent',boundary=start,raw=raw.tolist()))
        result=session.packet(start,raw.copy())
        node.need(type(result) is dict and set(result)=={'raw','stopped','end','rgb_bytes'},'physical packet ABI')
        got=join.v1.f32(result['raw'],(5,2))
        node.need(raw.tobytes()==got.tobytes(),'physical exact full ACK')
        node.need(type(result['stopped']) is bool and type(result['end']) is int and result['end']==start+5,'return boundary')
        emit(dict(kind='packet_return',boundary=result['end'],stopped=result['stopped']))
        rows.append(dict(start=start,end=result['end'],stopped=result['stopped']))
        if result['stopped'] or start==15:break
        rgb=result['rgb_bytes'];rgb_sha=join.v1.sha(rgb)
        emit(dict(kind='encode_intent',boundary=start+5,rgb_sha256=rgb_sha))
        enc=session.encode(start+5,rgb)
        node.need(type(enc) is dict and set(enc)=={'latent','input_sha','episode_id','boundary'},'encoder ABI')
        actual_context=deepcopy(b['root_context'])
        for key in ('latent_history','action_history'):
            actual_context[key]=np.asarray(actual_context[key],np.float32)
        context=join.joined_context(actual_context,prior,raw,got,enc['latent'],
            np.asarray(b['action_mean'],np.float32),np.asarray(b['action_std'],np.float32),
            expected_rgb_sha=rgb_sha,rgb_bytes=rgb,encoding_input_sha=enc['input_sha'],
            episode_id=eid,physical_episode_id=eid,encoder_episode_id=enc['episode_id'],
            start=start,end=start+5,physical_end=result['end'],encoding_boundary=enc['boundary'],stopped=False)
        emit(dict(kind='encoded',boundary=start+5,rgb_sha256=rgb_sha))
        b['root_context']=dict(context,latent_history=context['latent_history'].tolist(),action_history=context['action_history'].tolist())
        b['feedback_boundary']=start+5
        planned=plan(deepcopy(b));node.need(type(planned) is dict and 'first5' in planned,'plan ABI')
        prior=got.copy();raw=np.asarray(planned['first5'],dtype=np.float32)
        join.v1.f32(raw,(5,2));node.need(np.isfinite(raw).all() and (abs(raw)<=1).all(),'new action box')
        if b['domain']['name']=='Dx':node.need(np.all(raw[:,1].view(np.uint32)==0),'Dx positive zero')
        emit(dict(kind='plan_return',boundary=start+5,raw=raw.tolist()))
    return dict(rows=rows,closed_loop_complete=len(rows)==3 and rows[-1]['end']==20,
                source_authenticated=False,production_receipt=False)
