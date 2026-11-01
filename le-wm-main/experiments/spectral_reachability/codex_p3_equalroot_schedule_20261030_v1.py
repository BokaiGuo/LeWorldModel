"""Fixed equal-per-root exploratory queue, no outcome or model inputs."""
import hashlib
import math
import numpy as np

POLICIES = ("uniform", "signal", "reverse")
TAG = "p3-equalroot-v1-fixed-cold-bank"


def banks(ball):
    if not math.isfinite(ball) or ball <= 0:
        raise ValueError("positive action ball")
    out = []
    for root in (0, 1):
        seed = int.from_bytes(hashlib.sha256(f"{TAG}:{root}".encode()).digest()[:8], "little")
        rng = np.random.Generator(np.random.PCG64(seed))
        bank = []
        for index in range(15):
            packets = []
            for _ in range(2):
                v = rng.standard_normal(10)
                norm = np.linalg.norm(v)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("invalid draw; no redraw")
                packets.append(v / norm * ball * rng.random() ** (1 / 10))
            logits = np.asarray(packets, np.float32)
            raw = np.tanh(logits).reshape(10, 2)
            bank.append(dict(index=index, kind="equalroot_pcg64_ball",
                             logits=logits.astype(float).tolist(),
                             commands=raw.astype(float).tolist()))
        out.append(bank)
    return out


def queue(policy, weights):
    if policy not in POLICIES or type(weights) not in (list, tuple) or len(weights) != 2 or any(
        type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights
    ):
        raise ValueError("policy and finite nonnegative weights")
    uniform = list(range(5)) + list(range(10, 15))
    jobs = []
    for root in (0, 1):
        if policy == "uniform" or weights[0] == weights[1]:
            indices = uniform
        else:
            high = weights[root] > weights[1-root]
            if policy == "reverse":
                high = not high
            indices = list(range(10)) if high else list(range(5, 15))
        jobs.extend(dict(root=root, index=i) for i in indices)
    return dict(policy=policy, quotas=[10, 10], jobs=jobs, candidates_per_root=10,
                per_root_neural_axis_cap=20, per_root_metric_cap=10,
                signal_role="arbitrary rank-to-window intervention, not original allocation",
                signal_acquisition="historically shared; full per-arm cost unknown",
                physical_release=False, scientific_pass=False)
