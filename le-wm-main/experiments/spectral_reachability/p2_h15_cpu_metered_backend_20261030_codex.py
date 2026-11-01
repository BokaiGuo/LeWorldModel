"""New CPU H15 spectral compiler/score backend using the durable hook driver.

No neural model or physical execution. The numerical equations mirror the
existing H15 spectral bank and bounded-ridge implementation; instrumentation
is explicit at each SVD, ridge solve and candidate score entry.
"""
import copy
import functools
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

CAP = .75 * np.sqrt(15)
RADII = (.25 * np.sqrt(15), CAP)


def fail_stop(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except BaseException:
            self.driver.failed = True
            raise
    return wrapped


def make_driver(directory, contract, contract_sha, original_caps, arm, run_id):
    """Bind metric calls to the contract selection cap for this exclusive backend.

    This native metric axis is reserved solely for score_candidate below.
    Other metric calls on the driver invalidate the adapter's interpretation.
    """
    spec.validate_contract(contract)
    caps = core.counters(original_caps)
    caps["metric_calls"] = min(caps["metric_calls"], contract["axes"]["selection_evaluations"]["cap"])
    return durable.Driver(directory, contract, contract_sha, caps, arm, run_id)


class Backend:
    """One compilation and ordered candidate scores, using real NumPy kernels."""
    def __init__(self, driver):
        core.require(type(driver) is durable.Driver and not driver.closed and not driver.failed,
                     "fresh durable driver")
        core.require(not driver.ledger.receipt()["events"], "fresh unshared ledger")
        self.driver = driver
        self.compiled = False
        self.next_candidate = 0
        self.started_ns = time.monotonic_ns()
        self.cpu_started_ns = time.process_time_ns()
        self.source_sha256 = spec.sha(Path(__file__).read_bytes())

    def scope(self, **fields):
        core.require(spec.sha(Path(__file__).read_bytes()) == self.source_sha256, "adapter source drift")
        return dict(fields, adapter_sha256=self.source_sha256)

    @fail_stop
    def compile(self, jacobian, nominal, goal):
        core.require(not self.compiled and self.next_candidate == 0, "compile exactly once")
        j = np.asarray(jacobian, dtype=np.float64)
        z = np.asarray(nominal, dtype=np.float64)
        g = np.asarray(goal, dtype=np.float64)
        core.require(j.shape == (192, 30) and z.shape == g.shape == (192,) and
                     np.isfinite(j).all() and np.isfinite(z).all() and np.isfinite(g).all(),
                     "finite H15 response, nominal and goal")
        d = self.driver

        def ridge():
            left, s, right = d.call(self.scope(phase="compile", work="ridge_svd"), "array_svd",
                                    lambda: np.linalg.svd(j, full_matrices=False))
            residual = g - z
            target = left.T @ residual
            def solution(multiplier):
                return right.T @ (s * target / (s * s + .002 + multiplier))
            multiplier = 0.
            step = solution(multiplier)
            if np.linalg.norm(step) > CAP:
                low, high = 0., 1.
                while np.linalg.norm(solution(high)) > CAP:
                    high *= 2
                for _ in range(70):
                    mid = (low + high) / 2
                    if np.linalg.norm(solution(mid)) > CAP:
                        low = mid
                    else:
                        high = mid
                multiplier, step = high, solution(high)
            remaining = residual - j @ step
            return dict(params=step.tolist(), singular_values=s.tolist(),
                        linear_terminal_cost=float(remaining @ remaining / len(residual)),
                        ridge_penalty=float(.002 * (step @ step) / len(residual)),
                        own_reach_value=float((remaining @ remaining + .002 * (step @ step)) / len(residual)),
                        radius=float(CAP), norm=float(np.linalg.norm(step)), ball_multiplier=multiplier)

        def compute():
            result = d.call(self.scope(phase="compile", work="ridge"), "solve", ridge)
            _, singular, right = d.call(self.scope(phase="compile", work="mode_svd"), "array_svd",
                                         lambda: np.linalg.svd(j, full_matrices=False))
            for row in right:
                if row[np.argmax(np.abs(row))] < 0:
                    row *= -1
            vectors = [np.zeros(30)] + [s * np.asarray(result["params"]) for s in (.25, .5, 1.)]
            labels = ["zero", "dls_.25", "dls_.5", "dls_1"]
            for mode in range(9):
                for ri, radius in enumerate(RADII):
                    for sign in (-1, 1):
                        vectors.append(sign * radius * right[mode])
                        labels.append(f"mode_{mode}_radius_{ri}_sign_{sign}")
            vectors = np.stack(vectors)
            core.require(vectors.shape == (40, 30) and
                         np.max(np.linalg.norm(vectors, axis=1)) <= CAP * (1 + 1e-12), "joint ball")
            return dict(labels=labels, logits=vectors.tolist(),
                        raw15=np.tanh(vectors).astype(np.float32).reshape(40, 15, 2).tolist(),
                        singular_values=singular.tolist(), right_vectors=right.tolist(), ridge=result)

        bank = d.call(self.scope(phase="compile", work="bank"), "program_compile", compute)
        self.goal = g.copy()
        self.compiled = True
        return bank

    @fail_stop
    def score_candidate(self, index, terminal, goal):
        core.require(self.compiled and type(index) is int and index == self.next_candidate and index < 40,
                     "ordered candidate after compilation")
        t, g = np.asarray(terminal, np.float64), np.asarray(goal, np.float64)
        core.require(t.shape == g.shape == (192,) and np.isfinite(t).all() and np.isfinite(g).all()
                     and np.array_equal(g, self.goal),
                     "finite score inputs")
        def score():
            delta = t - g
            value = float(delta @ delta / 192)
            core.require(np.isfinite(value), "finite score")
            return value
        value = self.driver.call(self.scope(phase="selection", candidate=index), "metric", score)
        self.next_candidate += 1
        return value

    def finish(self):
        """Return recorded timings for this CPU adapter, not full planner time."""
        wall_ns = time.monotonic_ns() - self.started_ns
        cpu_ns = time.process_time_ns() - self.cpu_started_ns
        if not self.compiled or self.next_candidate != 40:
            self.driver.failed = True
        outcome = self.driver.finish()
        outcome.update(cpu_adapter_wall_ns=wall_ns, cpu_process_ns=cpu_ns,
                       completed_scores=self.next_candidate, compilation_complete=self.compiled,
                       scope="CPU adapter lifetime only, excluding driver construction/finalization and prior work",
                       adapter_sha256=self.source_sha256,
                       FLOPs=None, neural_cost=None, physical_cost=None, coverage_verified=False)
        return copy.deepcopy(outcome)
