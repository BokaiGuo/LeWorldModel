"""Read-only audit of durable prediction attempts, including incomplete calls."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_prediction_journal_v1 import digest


def audit_journal(directory):
    directory = Path(directory)
    contract = json.loads((directory / "contract.json").read_text())
    dim, packet, cap = (contract[k] for k in ("action_dim", "packet_length", "token_cap"))
    if min(dim, packet, cap) < 1:
        raise ValueError("invalid contract")
    attempts = sorted(directory.glob("*_attempt.json"))
    total, calls = 0, []
    for index, path in enumerate(attempts):
        record = json.loads(path.read_text())
        stem = f"{index:06d}"
        if path.name != f"{stem}_attempt.json" or record["index"] != index:
            raise ValueError("noncontiguous attempts")
        action_path = directory / f"{stem}_actions.npz"
        if record["actions"] != action_path.name or digest(action_path) != record["sha256"]:
            raise ValueError("action digest mismatch")
        with np.load(action_path, allow_pickle=False) as saved:
            actions = saved["actions"]
        if (actions.dtype != np.float32 or actions.ndim != 3 or not len(actions)
                or not actions.shape[1] or actions.shape[1] % packet or actions.shape[2] != dim
                or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
            raise ValueError("invalid recorded actions")
        cost = len(actions) * (actions.shape[1] // packet)
        if record["cost"] != cost:
            raise ValueError("incorrect recorded cost")
        total += cost
        returned = directory / f"{stem}_returned.json"
        failed = directory / f"{stem}_failed.json"
        if returned.exists() and failed.exists():
            raise ValueError("conflicting call outcomes")
        status = "incomplete"
        if returned.exists():
            outcome = json.loads(returned.read_text())
            result = directory / f"{stem}_result.npz"
            if (outcome["index"] != index or outcome["result"] != result.name
                    or digest(result) != outcome["sha256"]):
                raise ValueError("result digest mismatch")
            with np.load(result, allow_pickle=False) as saved:
                features, scores = saved["features"], saved["scores"]
                if (features.ndim != 2 or len(features) != len(actions)
                        or scores.shape != (len(actions),) or not np.isfinite(features).all()
                        or not np.isfinite(scores).all()):
                    raise ValueError("invalid recorded result")
            status = "returned"
        elif failed.exists():
            outcome = json.loads(failed.read_text())
            if outcome["index"] != index or not outcome["type"]:
                raise ValueError("invalid failed outcome")
            status = "failed"
        calls.append({"index": index, "cost": cost, "status": status})
    if total > cap:
        raise ValueError("budget exceeded")
    return {"calls": calls, "tokens": total, "cap": cap}
