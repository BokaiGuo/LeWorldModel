"""K2 transport development tests; no neural or physics execution."""
from copy import deepcopy
import runpy
from pathlib import Path

import pytest
from experiments.spectral_reachability import agent_p3_c3_k2_online_transport_20261029_v2 as m
from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as host


def test_full_six_slot_online_tempdisk(tmp_path):
    f = runpy.run_path(str(host.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))
    h = runpy.run_path(str(host.ROOT/'tests/test_agent_p3_c3_compact_host_cuda_20261029_v1.py'))
    p = h['ram_prepare'](f); before = h['before'](p)
    _, store, catalog, _ = host.make_seal(p, before, fixture=True)
    online = m.OnlineStore(tmp_path/'bundle', store, catalog, p['tasks'])
    try:
        rows = m.execute(p, lambda t,s:f['Native'](t,s), online, permit_ram=True)
        assert len(rows) == 6
        assert all(r['result'] is not None for r in rows)
        assert len(online.transport.journal) > 0
        events = [store.event(i) for i in range(len(store.events))]
        summaries = m.validate_rows(p, rows, events,
            lambda slot, ref: store.get(bytes.fromhex(ref['object_sha256'])),
            online.capsule, permit_fixture=True)
        assert len(summaries) == 6
    finally:
        online.transport.close()
