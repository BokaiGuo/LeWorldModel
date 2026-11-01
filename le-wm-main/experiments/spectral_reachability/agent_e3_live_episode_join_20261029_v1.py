"""Wire original E3 setup, single environment, HOST encoding and later nodes.

This is a lifecycle component, not an authorized launcher or full consumer.
The caller must authenticate all inputs and the model/source/precision before
calling, enforce whole-job budgets, persist the terminal row and publish.
"""
from copy import deepcopy
import traceback
from experiments.spectral_reachability import agent_e3_feedback_setup_20261029_v2 as setup
from experiments.spectral_reachability import agent_e3_native_region_adapter_20261029_v1 as region
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture
from experiments.spectral_reachability import agent_e3_continuous_dispatch_20261029_v1 as dispatcher
from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as encoding
from experiments.spectral_reachability import agent_e3_multinode_saved_20261029_v1 as nodes
from experiments.spectral_reachability import agent_e3_terminal_row_20261029_v1 as terminal


def execute(env, setup_kwargs, body, first5, model, torch, device, unified_store, meter, row):
    """Use the same environment for source setup and both feedback boundaries.

    No artificial stop, state hydration, alternate task, or callback-defined
    model is introduced. Errors propagate after recording and closing env.
    """
    bridge=capture.CaptureStore(unified_store)
    args=dict(setup_kwargs)
    args.update(row=row,meter=meter,store=bridge,packet_ops=region)
    def encode(actual_env, observation, boundary):
        dispatcher.node.need(actual_env is env,'same environment for encoding')
        frame=bridge.descriptor(observation)
        previous=meter.region;meter.region='encode'
        try:
            check=lambda: unified_store.check('encoding_arrays',0)
            latent,_=encoding.encode(model,torch,device,unified_store,frame,boundary,check)
            rgb=encoding.u.read_bytes(unified_store.output/frame['file'],capture.pub.PNG_CAP)
            return dict(latent=latent,rgb_bytes=rgb,input_sha=frame['sha256'],
                        episode_id=row['episode_id'],boundary=boundary)
        finally:meter.region=previous
    def plan(task):
        previous=meter.region;meter.region='compile'
        try:
            adapter=nodes.NodeBridge(unified_store,task['feedback_boundary'])
            record=dict(forwards=[],AD=[],result=None)
            backend=nodes.bridge.host.HostBackend(model,device,torch,adapter,task,record)
            with backend.observe():record['result']=nodes.node.solve(task,backend,adapter.event)
            adapter.finish(record)
            return record['result']
        finally:meter.region=previous
    primary=None
    try:
        result=setup.setup(env,**args)
        if result is not None:
            dispatcher.dispatch(env,body,result,first5,encode,plan,bridge,meter,row,packet_fn=region.packet)
        return row
    except BaseException as exc:
        primary=exc
        detail=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        row['status']='failed';row['error']=detail
        try:bridge.emit('episode_failure',error=deepcopy(detail))
        except BaseException as secondary:exc.add_note('failure event: '+repr(secondary))
        raise
    finally:
        cleanup_errors=[]
        try:
            bridge.emit('cleanup_intent',episode_id=row['episode_id'])
        except BaseException as exc:cleanup_errors.append(exc)
        closed=False
        try:
            env.close()
            closed=True
        except BaseException as exc:cleanup_errors.append(exc)
        try:
            bridge.emit('cleanup_return' if closed else 'cleanup_failure',episode_id=row['episode_id'])
        except BaseException as exc:cleanup_errors.append(exc)
        if cleanup_errors:
            row['cleanup_errors']=[dict(type=type(exc).__name__,message=str(exc)) for exc in cleanup_errors]
            row['status']='failed'
        try:
            terminal.write(unified_store,row)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            if primary is None:raise cleanup_errors[0]
            for exc in cleanup_errors:primary.add_note('cleanup or terminal persistence failure: '+repr(exc))
