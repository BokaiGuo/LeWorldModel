import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_predicted_vs_real_10x2_20261030_main as cmp


def inputs():
    metric = dict.fromkeys(cmp.pair.FIELDS, 1.0)
    metric["native_success"] = False
    rows, shards = [], []
    for i in range(24):
        rows.append({"index": i, "logical_pair": i//3, "root_slot": 0, "condition": "on",
                     "arm": "test", "role": "selected", "status": "probe_completed",
                     "accepted_suffix_steps": 10, "endpoint_metrics": metric})
        shards.append({"index": i})
    transition = [dict(step=i+1, actual_bits="00", state=[0], snapshot={"s": 0},
                       terminated=False, truncated=False) for i in range(5)]
    z = np.zeros(192, dtype="<f4")
    sha = hashlib.sha256(z.tobytes()).hexdigest()
    side = {"actual_latent": z.tolist(), "predicted_latent": z.tolist(),
            "actual_sha256": sha, "predicted_sha256": sha}
    return {"rows": rows}, {"rows": copy.deepcopy(rows)}, {"shards": shards}, {"shards": shards}, transition, side


def test_reject_mismatched_physical_history_and_latent():
    p, r, pr, rr, t, side = inputs()
    result = cmp.compare(p, r, pr, rr, lambda *_: (t, t), lambda *_: side)
    assert result["summary"]["common_first5_verified"] == 24
    bad = copy.deepcopy(t)
    bad[0]["state"] = [1]
    with pytest.raises(ValueError, match="first five"):
        cmp.compare(p, r, pr, rr, lambda *_: (t, bad), lambda *_: side)
    bad_side = dict(side, predicted_sha256="0"*64)
    with pytest.raises(ValueError, match="latent hash"):
        cmp.compare(p, r, pr, rr, lambda *_: (t, t), lambda *_: bad_side)
