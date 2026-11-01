"""Integer monotonicity check, conditional on frozen compact format invariants.

Not an output-fit certificate: routing and all writer enforcement still need
an independent end-to-end proof. No model or data is loaded.
"""
from experiments.spectral_reachability import agent_p3_c3_k2_online_transport_20261029_v2 as online


def bound():
    b = online.count_envelope(); t = b['new_totals']
    n, j, v = t['predict_API'], t['jacobian_API'], t['VJP_directions']
    requests, slots, replay = b['request_denominator'], b['new_slots'], t['replay_calls']
    # Keep old 7464 scalar slack rather than infer a smaller unsupported bound.
    ledger = 4*n+3*v+2*j+7464
    arrays = 10*n+3*v+5*j
    calls = 2*(j+replay+requests)
    auxiliary = slots*4096
    resources = 2*ledger+slots
    events = resources+arrays+v+calls+auxiliary
    tensors = 19*n+5*v+13*j
    parts = dict(native_tensor_bytes=4*(8192*n+474*v+8192*j),
        literal_f64_bytes=8*(5952*j+596*n),
        tensor_headers_and_indices=128*(tensors+2*j+2*n),resource_rows=resources*192,
        array_rows=arrays*512,direction_rows=v*64,native_call_rows=calls*128,
        auxiliary_rows=auxiliary*256,journal=events*104,event_object_indices=events*48,
        container_spines_and_secondary_views=events*256,parent_structures=slots*17*524288,
        candidates=requests*65536,slot_structures=slots*1048576,
        sealed_static_metadata=online.codec.STATIC)
    old = online.codec.proof()['parts']
    online.need(parts.keys() == old.keys() and all(parts[k] <= old[k] for k in old),
                'componentwise conditional format bound')
    return dict(parts=parts, old_parts=old, conditional_bytes=sum(parts.values()),
                old_bytes=sum(old.values()), slots=slots, requests=requests,
                output_fit_proved=False, requires='K2 routing, metadata, source and failure proof')
