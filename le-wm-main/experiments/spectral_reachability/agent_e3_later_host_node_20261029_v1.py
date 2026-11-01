"""HOST backend execution plus independent numerical replay for E3 later nodes.

This is not a source-authorizing launcher and does not authenticate actual RGB.
"""
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_e3_feedback_node_20261029_v1 as node

host=node.host


def execute(body,model,device,torch,store,state_seal):
    """Run native HOST backend; store MUST be compatible with frozen HOST ABI."""
    b=node.validate_body(body);host.need(getattr(device,'type',None)=='cuda','HOST CUDA only')
    before=deepcopy(state_seal());row=dict(forwards=[],AD=[],result=None)
    backend=host.HostBackend(model,device,torch,store,b,row)
    try:
        with backend.observe():row['result']=node.solve(b,backend,store.event)
        host.same(state_seal(),before,'planner cannot mutate live environment')
    except BaseException as exc:
        row['failure']=dict(type=type(exc).__name__,message=str(exc))
        try:store.put('later_node_partial.json',row,final=True)
        except BaseException as secondary:exc.add_note('partial row persistence: '+repr(secondary))
        raise
    store.put('later_node_row.json',row,final=True)
    return row


def consume(body,row,output,events):
    """Replay saved native arrays/AD and event sequence; no upstream authority."""
    b=node.validate_body(body);host.need(type(row) is dict and set(row)=={'forwards','AD','result'},'complete own row')
    reconstructed=[];backend=host.SavedBackend(b,row,output,reconstructed.append)
    result=node.solve(b,backend,reconstructed.append);backend.complete()
    host.same(result,row['result'],'full saved candidates, raw gradients, replays')
    host.same(events,reconstructed,'all native/AD chronologies')
    counts=host.axes(events);host.same(counts['attempted'],host.numerical.axes27(b['horizon_packets']),'attempted axes')
    host.same(counts['completed'],counts['attempted'],'completed axes')
    return dict(result=deepcopy(result),axes=counts,full_native_strict=False,
                authenticated_actual_feedback=False,scientific_admission=False)
