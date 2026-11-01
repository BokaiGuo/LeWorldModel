"""Single-env E3 dispatcher core; callbacks are not production authority."""
from copy import deepcopy
import traceback
import numpy as np
from experiments.spectral_reachability import agent_e3_actual_feedback_join_20261029_v2 as join
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node
from experiments.spectral_reachability import agent_e3_feedback_single_capture_20261029_v1 as packet


def dispatch(env,body,setup_result,first5,encode,plan,store,meter,row,*,packet_fn=packet.packet):
    """Drive one already setup env until control20 or native stop.

    Caller MUST authorize sealed cp5 plan and source/model/encoder IO; this
    orchestrator does not authenticate arbitrary callbacks or produce a receipt.
    """
    node.host.numerical.validate_body(body)
    b=deepcopy(body);goal=setup_result['goal'];rgb_goal=setup_result['goal_rgb']
    prior=join.v1.f32(setup_result['prior_raw5'],(5,2)).copy()
    action=join.v1.f32(first5,(5,2)).copy()
    row.setdefault('slots',[dict(boundary=x,status='not_reached') for x in (5,10,15)])
    node.need([x['boundary'] for x in row['slots']]==[5,10,15],'fixed node roster')
    try:
        for idx,start in enumerate((5,10,15)):
            row['slots'][idx]['status']='dispatch_attempted'
            store.emit('e3_packet_intent',boundary=start,raw_bits=action.tobytes().hex())
            out=packet_fn(env,action,start=start,row=row,goal=goal,goal_rgb=rgb_goal,meter=meter,store=store)
            returned=out['returned_raw'];node.need(isinstance(returned,np.ndarray) and returned.dtype==np.float32
                and returned.ndim==2 and returned.shape[1:]==(2,) and len(returned)<=5
                and returned.tobytes()==action[:len(returned)].tobytes(),'literal returned prefix')
            node.need(type(out['end']) is int and out['end']==start+len(returned)
                and type(out['stopped']) is bool and type(out['complete']) is bool
                and out['complete']==(len(returned)==5),'physical packet return ABI')
            row['slots'][idx].update(status='stopped' if out['stopped'] else 'returned',end=out['end'])
            store.emit('e3_packet_return',boundary=start,end=out['end'],stopped=out['stopped'],
                       returned_bits=returned.tobytes().hex())
            if out['stopped']:
                row['status']='native_stopped';return row
            node.need(out['complete'],'incomplete without native stop')
            if start==15:row['status']='completed20';return row
            observation=out['observation'];node.need(type(observation) is dict and observation.get('step')==start+5,'actual own boundary image')
            store.emit('e3_encode_intent',boundary=start+5,observation=deepcopy(observation))
            encoded=encode(env,deepcopy(observation),start+5)
            node.need(type(encoded) is dict and set(encoded)=={'latent','rgb_bytes','input_sha','episode_id','boundary'},'encoding response ABI')
            rgb=encoded['rgb_bytes'];physical_sha=observation['file_sha256']
            current=deepcopy(b['root_context'])
            for key in ('latent_history','action_history'):current[key]=np.asarray(current[key],np.float32)
            context=join.joined_context(current,prior,action,returned,encoded['latent'],
                np.asarray(b['action_mean'],np.float32),np.asarray(b['action_std'],np.float32),
                expected_rgb_sha=physical_sha,rgb_bytes=rgb,encoding_input_sha=encoded['input_sha'],
                episode_id=row['episode_id'],physical_episode_id=row['episode_id'],encoder_episode_id=encoded['episode_id'],
                start=start,end=start+5,physical_end=out['end'],encoding_boundary=encoded['boundary'],stopped=False)
            store.emit('e3_encode_return',boundary=start+5,observation_sha=physical_sha,
                       latent=np.asarray(encoded['latent'],np.float32).tolist())
            b['root_context']=dict(context,latent_history=context['latent_history'].tolist(),action_history=context['action_history'].tolist())
            b['feedback_boundary']=start+5
            store.emit('e3_node_intent',boundary=start+5,body_sha256=node.host.digest(b))
            result=plan(deepcopy(b));node.need(type(result) is dict and 'first5' in result,'node output')
            candidate=np.asarray(result['first5']);node.need(candidate.shape==(5,2) and np.isfinite(candidate).all()
                and np.array_equal(candidate,candidate.astype(np.float32)),'literal f32 plan action')
            action=candidate.astype(np.float32);node.need((abs(action)<=1).all(),'raw box')
            if b['domain']['name']=='Dx':node.need(np.all(action[:,1].view(np.uint32)==0),'Dx positive zero')
            store.emit('e3_node_return',boundary=start+5,body_sha256=node.host.digest(b),raw_bits=action.tobytes().hex())
            prior=returned.copy()
    except BaseException as exc:
        row['status']='failed';row['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        try:store.emit('e3_dispatch_failure',error=deepcopy(row['error']))
        except BaseException as secondary:exc.add_note('failure event: '+repr(secondary))
        raise
