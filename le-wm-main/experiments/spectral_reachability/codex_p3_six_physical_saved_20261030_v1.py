"""Saved ACK semantic consumer; does not certify source or physical execution."""
from experiments.spectral_reachability import codex_p3_six_physical_contract_20261030_v1 as spec


def verify_slot(plan, acks, row):
    if type(acks) is not list or type(row) is not dict:
        raise ValueError("saved slot")
    events = []
    for i, record in enumerate(acks):
        if record["ordinal"] != i or record["sent_dtype"] != "float32" or record["actual_dtype"] != "float32" or \
           record["sent_shape"] != [2] or record["actual_shape"] != [2]:
            raise ValueError("saved ACK metadata")
        events.append(dict(step=i, sent_float32_hex=record["sent_hex"], actual_float32_hex=record["actual_hex"],
                           terminated=record["terminated"], truncated=record["truncated"]))
    result = spec.verify_ack(plan, events, row)
    return dict(**result, source_authenticated=False, runtime_reset_verified=False,
                budget_census_verified=False)
