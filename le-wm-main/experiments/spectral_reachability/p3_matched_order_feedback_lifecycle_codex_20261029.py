"""Four-slot lifecycle over source-bound callback factories, not a host launcher.

Factories must provide authenticated instances; this module does not certify
their provenance.  Failed/terminated slots are retained and never replaced.
"""
from copy import deepcopy
from experiments.spectral_reachability import p3_matched_order_feedback_seam_codex_20261029 as seam
from experiments.spectral_reachability import p3_matched_order_feedback_numeric_bridge_codex_20261029 as bridge
from experiments.spectral_reachability import p3_matched_order_feedback_native_codex_20261029 as native
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as prior

NAME = 'p3_matched_order_feedback_lifecycle_codex_20261029'
need = seam.need


def execute(roots, store, create, backend_factory, *, engine=seam.execute_slot, permit_ram=False):
    """Execute four fixed identities; no model/physics follows an actual stop.

    create(slot, root, policy) returns (session, prior_raw, close). It must
    close an environment itself if setup raises before returning. close() is
    always attempted once when create returned, including after engine errors.
    """
    need(type(roots) is list and len(roots) == 2, 'fixed two roots')
    rows = [dict(slot=s,root=r,policy=p,status='not_started',result_sha256=None,
                 cleanup='not_required',error=None) for s,r,p in seam.ROSTER]
    store.event('physical','record','matched_denominator',store.index,
                dict(rows=deepcopy(rows),fixed_denominator=4))
    for row in rows:
        slot,root,policy = row['slot'],row['root'],row['policy']
        store.event('physical','record','matched_slot_attempt',store.index,
                    dict(slot=slot,root=root,policy=policy))
        sink = prior.ArmSink(bridge.Bridge(store,slot),slot)
        close = None
        try:
            session,raw,close = create(slot,root,policy)
            need(callable(close), 'one explicit cleanup callback')
            row['cleanup'] = 'pending'
            result = engine(roots[root],policy,raw,session,backend_factory,sink,permit_ram=permit_ram)
            need(type(result) is dict and result['policy'] == policy, 'same-slot engine result')
            row['status'] = result['status']
            row['result_sha256'] = seam.core.digest(result)
        except native.NativeInitialStop as exc:
            row.update(status='native_initial_stop',error=str(exc))
        except Exception as exc:
            row.update(status='engineering_failed',error=type(exc).__name__+': '+str(exc))
        finally:
            if close is not None:
                try:
                    close(); row['cleanup'] = 'returned'
                except Exception as exc:
                    row['cleanup'] = 'failed'
                    row['error'] = (row['error']+'; ' if row['error'] else '')+'cleanup: '+type(exc).__name__+': '+str(exc)
                    row['status'] = 'engineering_failed'
        store.event('physical','record','matched_slot_return',store.index,deepcopy(row))
    return dict(schema=NAME+'.roster',rows=rows,fixed_denominator=4,
                source_authenticated=False,physical_certified=False,scientific_pass=False)
