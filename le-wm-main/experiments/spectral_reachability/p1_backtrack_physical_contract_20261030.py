"""Fixed 16-row P1 physical resource contract; not a physical authorization."""
import hashlib
from copy import deepcopy

import numpy as np


ARMS = ("projected_ascent", "outward_free_tangent")
ROLES = ("incumbent", "model_selected")
ORDER = [(slot, arm, role) for slot in range(4) for arm in ARMS for role in ROLES]


def contract():
    # One independent reset/warmup and one 15-step trajectory per logical row.
    # Rendering is three overlays at 11 warmup + 16 plan observations per row.
    return dict(fixed_rows=16, known_roots=2, fresh_roots=0, warmup_steps=10,
                plan_steps=15, checkpoints=[0, 5, 10, 15], per_row_fresh_environment=True,
                caps=dict(constructor=16, close=16, reset=32, set_state=64,
                          goal_setter=48, goal_render=32, goal_bind=16,
                          public_step=400, physics_step=4064, explicit_render=1296),
                output_cap_bytes=512 * 1024**2, final_reserve_bytes=16 * 1024**2,
                disk_floor_bytes=512 * 1024**2, wall_seconds=3600,
                capability_probe_rows=0, training=False, new_neural_calls=0,
                P1_admission=False, independent_confirmation=False)


def plans(joined):
    if (joined.get("schema") != "p1_backtrack_physical_source_join_20261030"
            or joined.get("fixed_slots") != 4 or joined.get("fixed_roles") != 16
            or joined.get("known_roots") != 2 or joined.get("physical_authorized") is not False
            or joined.get("P1_admission") is not False or joined.get("freshness") is not False):
        raise ValueError("source join contract")
    tasks, rows = joined.get("task_rows"), joined.get("rows")
    if not isinstance(tasks, list) or len(tasks) != 4 or not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("fixed roster")
    result = []
    for index, (row, expected) in enumerate(zip(rows, ORDER, strict=True)):
        slot, arm, role = expected
        task = tasks[slot]
        if ((row.get("slot"), row.get("arm"), row.get("role")) != expected
                or task.get("slot") != slot
                or (task.get("root_slot"), task.get("condition")) != (slot // 2, ("off", "on")[slot % 2])
                or row.get("task_id") != task.get("task_id")
                or row.get("physical_root_sha256") != task.get("native_root_sha256")
                or task.get("warmup_model_history_exact_numpy") is not True):
            raise ValueError("task, role or native source mismatch")
        encoded = row.get("raw15_float32_hex")
        if not isinstance(encoded, str) or len(encoded) != 240 or encoded != encoded.lower():
            raise ValueError("raw15 literal bytes")
        try:
            data = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("raw15 literal bytes") from exc
        if len(data) != 120 or data.hex() != encoded:
            raise ValueError("raw15 literal bytes")
        action = np.frombuffer(data, dtype="<f4")
        if not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError("raw15 bounds")
        plan = deepcopy(row)
        plan.update(index=index, root_slot=task["root_slot"], condition=task["condition"],
                    raw15_sha256=hashlib.sha256(data).hexdigest())
        result.append(plan)
    return result
