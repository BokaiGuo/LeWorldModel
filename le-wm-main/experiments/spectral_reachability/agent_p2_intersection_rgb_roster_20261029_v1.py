"""RAM-only intersection six-slot RGB roster; not a CUDA runner or authority."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

NAME = 'agent_p2_intersection_rgb_roster_20261029_v1'
ORDER = ((0,'off','joint'), (0,'off','composition'), (0,'off','grasp'),
         (1,'off','grasp'), (1,'off','composition'), (1,'off','joint'))
CP = (0, 5, 10, 15)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vec(v, shape):
    a = np.asarray(v, dtype=np.float32)
    require(a.shape == shape and np.isfinite(a).all(), 'finite float32 shape')
    return a


def assemble(prepared, receipt, output):
    """Project already authenticated objects. Caller must establish external authority."""
    output = Path(output)
    require(output.is_absolute(), 'absolute physical output')
    require(prepared['schema'] == 'agent_p2_intersection_physical_20261029_v2.prepare', 'prepare schema')
    require(receipt['schema'] == 'agent_p2_intersection_physical_20261029_v2.consume'
            and receipt['fixed_denominator'] == 6 and receipt['known_roots'] == 2
            and receipt['fresh_roots'] == 0 and receipt['inherited'] == 0, 'receipt identity')
    require(len(prepared['plans']) == len(receipt['plans']) == len(receipt['rows']) == 6
            and prepared['plans'] == receipt['plans']
            and prepared['planner_authority'] == receipt['planner_authority'], 'full plan authority')
    rows = []
    for i, (plan, native) in enumerate(zip(prepared['plans'], receipt['rows'], strict=True)):
        root, condition, arm = ORDER[i]
        require((plan['index'], plan['root_slot'], plan['condition'], plan['arm']) == (i, root, condition, arm)
                and (native['index'], native['root_slot'], native['condition'], native['arm']) == (i, root, condition, arm)
                and native['plan'] == plan and native['task_id'] == plan['task_id']
                and native['selection_digest'] == plan['selection_digest'], 'slot and action authority')
        raw = vec(plan['raw15'], (15, 2))
        require(raw.tobytes().hex() == plan['raw_float32_bits']
                and hashlib.sha256(raw.tobytes()).hexdigest() == plan['raw_float32_sha256'], 'raw15 bits')
        require(vec(plan['commands'], (15, 2)).tobytes() == raw.tobytes(), 'native commands')
        goal = vec(plan['goal'], (192,))
        require(len(plan['contexts']) == 3, 'three model contexts')
        steps = [c['step'] for c in native['checkpoints']]
        require(steps == [c for c in CP if c in steps] and native['missing_checkpoints'] == [c for c in CP if c not in steps], 'checkpoint roster')
        for j, cp in enumerate(CP):
            context = plan['root_context'] if cp == 0 else plan['contexts'][j-1]
            pred = vec(context['latent_history'][-1], (192,))
            frame = next((x for x in native['checkpoints'] if x['step'] == cp), None)
            response = next((x for x in native['responses'] if x['step'] == cp), None) if cp else None
            if frame is not None:
                require(native['initial_exact'] is True and frame['state'] == frame['snapshot']['state']
                        and Path(frame['png']).name == frame['png'], 'checkpoint state and PNG')
                if cp:
                    require(response is not None and response['state'] == frame['state'], 'same-step native return')
            else:
                require(response is None, 'missing checkpoint with native label')
            rows.append(dict(index=len(rows), slot=i, root=root, condition=condition, arm=arm,
                checkpoint=cp, task_id=plan['task_id'], selection_digest=plan['selection_digest'],
                raw15_bits=plan['raw_float32_bits'], prefix_bits=raw[:cp].tobytes().hex(),
                prediction_role='observed_root_context_not_forecast' if cp == 0 else 'same_plan_forecast',
                predicted192=pred.tolist(), goal192=goal.tolist(), native_goal=deepcopy(native['goal']),
                frame=None if frame is None else dict(frame, path=str(output/frame['png'])),
                native_response=deepcopy(response), missing_reason=None if frame is not None else 'physical_checkpoint_missing',
                physical_status=native['status']))
    return rows
