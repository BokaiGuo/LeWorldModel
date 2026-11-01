"""Source-bound cp15 guard for a future ten-branch settling physical driver.

This module does not implement an environment driver or saved consumer.  The
caller must durably record the returned proof before dispatching control 16.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

from experiments.spectral_reachability import agent_p3_same_time_capability_probe_physical_20261029_v2 as old

NAME = 'agent_p3_settling_cp15_guard_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
OLD_SOURCE_SHA = 'f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344'
AUTH = {
    'prepare': ('agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json', '8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f'),
    'batch': ('agent_p3_same_time_capability_probe_physical_20261029_v2_MAIN01/batch.json', 'b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b'),
    'receipt': ('agent_p3_same_time_capability_probe_physical_20261029_v2_consume_MAIN01.json', '4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1'),
}
CAPS = dict(constructor=1,close=1,reset=2,set_state=4,goal_render=2,goal_bind=1,
            goal_setter=3,public_step=30,physics_step=304,explicit_render=96)
TAIL = [[0.0, 0.0] for _ in range(5)]
require, same, digest = old.require, old.same, old.digest


def source_references():
    return {k:dict(path=str(ROOT/'results'/path), sha256=h) for k,(path,h) in AUTH.items()}


def whole_reference(prepared, batch, receipt, refs, *, fixture=False):
    """Join saved authorities without rerunning old upstream fullstrict."""
    require(refs == source_references() or fixture, 'actual frozen three-authority references')
    require(prepared['schema'] == old.NAME+'.prepare' and len(prepared['jobs']) == 10
        and (prepared['fixture_only'] is False or fixture), 'old complete prepare')
    require(batch['schema'] == old.NAME+'.batch' and batch['prepared'] == refs['prepare']['path']
        and batch['prepare_sha256'] == refs['prepare']['sha256'] and len(batch['jobs']) == 10, 'old batch identity')
    require(receipt['schema'] == old.NAME+'.consume' and receipt['status'] == 'all10_saved_native_jobs_reconstructed_NOT_admitted'
        and receipt['fixed_denominator'] == 10 and len(receipt['jobs']) == 10
        and receipt['prepared'] == refs['prepare']['path'] and receipt['prepare_sha256'] == refs['prepare']['sha256'], 'old strict receipt identity')
    result = []
    for i, (job, seal, rec) in enumerate(zip(prepared['jobs'], batch['jobs'], receipt['jobs'], strict=True)):
        require(job['index'] == i and (job['root_slot'],job['source_role'],job['probe']) ==
            (old.ORDER[i][1],old.ORDER[i][2],old.ORDER[i][3]), 'fixed ten branch identities')
        require(rec['schema'] == old.NAME+'.job_consume' and rec['output'] == seal['output']
            and rec['report_sha256'] == seal['report_sha256'] and rec['completion_sha256'] == seal['completion_sha256'], 'per-job saved receipts')
        row = rec['row']; require(same(row['job'],job) and row['status'] == 'horizon_completed'
            and row['guards'] == [51,66] and not row['missing_checkpoints'] and len(row['observations']) == 81
            and len(row['operations']) == 374 and row['endpoint15']['step'] == 15, 'full original branch for cp15 replay')
        require(all(c['attempted'] == c['completed'] == old.COUNTS[k] for k,c in row['counts'].items()), 'full old native counts')
        result.append(dict(job=deepcopy(job), old_row=deepcopy(row), old_seal=deepcopy(seal)))
    return result


def load_references():
    refs = source_references()
    require(old.sha(str(Path(old.__file__).resolve())) == OLD_SOURCE_SHA, 'frozen source identity')
    for v in refs.values(): require(old.sha(v['path']) == v['sha256'], 'whole source SHA')
    p,b,r = (old.read(refs[k]['path']) for k in ('prepare','batch','receipt'))
    rows = whole_reference(p,b,r,refs)
    for v in refs.values(): require(old.sha(v['path']) == v['sha256'], 'stable authority bytes')
    return refs, rows


def raw_history_bits(commands):
    require(len(commands) >= 10, 'full chronological ten raw commands')
    return ''.join(old.bits(commands[-10:],10))


def guard81(job, original, observations, operations):
    """Full original 81 observations and all previously completed native operations.

    `original` comes from whole_reference, NOT an unverified synthetic row. Any
    unrecorded RNG/contact cache remains unknown; this is replay, not restore.
    """
    row = original['old_row']; require(same(job,original['job']) and same(row['job'],job), 'same source job')
    require(row['status'] == 'horizon_completed' and row['guards'] == [51,66]
            and len(row['observations']) == len(observations) == 81, 'full cp15 source and actual observations')
    source_ops = [x for x in row['operations'] if x['resource'] != 'close']
    require(len(source_ops) == 373 and len(operations) == 373, 'all native operations before close')
    require(all(same(a,b) for a,b in zip(operations,source_ops,strict=True)), 'whole native receipts, reset/goal/action/return/snapshot and timing order')
    require(all(x['status'] == 'returned' for x in operations), 'all prior operations returned')
    mapping = [old.projected(a,b,job) for a,b in zip(observations,row['observations'],strict=True)]
    pub = [x for x in operations if x['resource'] == 'public_step']
    commands = job['warm10']+job['raw15']
    require(len(commands) == len(pub) == 25 and [x['raw_bits'] for x in pub] == old.bits(commands,25), 'all literal chronological raw actions')
    require(all(x['result']['actual_bits'] == x['raw_bits'] and x['result']['actual_dtype'] == '<f4'
        and not x['result']['terminated'] and not x['result']['truncated'] for x in pub), 'actual returns and no prior stop')
    final = observations[-3:]
    require([(x['receipt']['phase'],x['receipt']['step'],x['receipt']['overlay_index']) for x in final] ==
        [('plan',15,0),('plan',15,1),('plan',15,2)], 'three cp15 RGB observations')
    require(all(same(x['receipt']['state'],pub[-1]['result']['state']) and
        same(x['receipt']['snapshot'],pub[-1]['result']['snapshot']) for x in final), 'actual cp15 state and full snapshot join')
    state = final[0]['receipt']['state']
    require(struct.pack('<7d',*state).hex() == final[0]['receipt']['snapshot']['state_float64_bits'], 'state float64 bits')
    require(same(final[0]['receipt'],row['endpoint15']), 'original full cp15 off RGB receipt')
    source_history = raw_history_bits(job['warm10']+job['raw15'])
    require(source_history == ''.join(x['raw_bits'] for x in pub[-10:]), 'exact last ten executed raw history bytes')
    return dict(schema=NAME+'.guard', source_job_sha256=digest(job), source_row_sha256=digest(row),
        actual_observations_sha256=digest(observations), source_observations_sha256=digest(row['observations']),
        actual_operations_sha256=digest(operations), source_operations_sha256=digest(source_ops),
        cp15_observations_sha256=digest(final), raw_history_float32_bits=source_history,
        raw_history_sha256=hashlib.sha256(bytes.fromhex(source_history)).hexdigest(),
        transport_mappings=mapping, unknown_semantics=deepcopy(job['unknown_semantics']),
        old_receipt_sha256=original.get('receipt_sha256'), source_verified_here=False)


def before_step16(job, original, observations, operations, durable_emit, callback):
    """Invoke only a RAM fixture callback unless MAIN integrates a reviewed driver."""
    proof = guard81(job,original,observations,operations)
    durable_emit(dict(kind='cp15_guard',accepted=True,proof=proof))
    # Caller-supplied sink must return only after journal write and fsync.
    return callback(deepcopy(TAIL[0]))


def budget():
    png = old.png_budget()['png_upper_bytes']
    total = 96*png + sum(old.QUOTAS.values()) + old.RESERVE
    require(total <= old.CAP, 'PNG plus original quotas and reserve fit 64MiB')
    return dict(per_job=deepcopy(CAPS), jobs=10, total={k:10*v for k,v in CAPS.items()},
        operations=sum(CAPS.values()), operation_events=2*sum(CAPS.values()),
        observation_events=96, guards=3, lifecycle=3,
        events=2*sum(CAPS.values())+96+3+3, png_upper_bytes=png,
        conservative_bytes=total, cap_bytes=old.CAP, reserve_bytes=old.RESERVE,
        floor_bytes=old.FLOOR, wall_seconds_per_job=old.WALL,
        metadata_structural_bound_proven=False, runtime_ready=False)
