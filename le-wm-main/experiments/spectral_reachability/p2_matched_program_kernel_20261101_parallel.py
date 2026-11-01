"""Measured-resource matched P2 joint search versus two-stage composition.

Both arms evaluate sixteen complete ten-action candidates.  The composition
arm deliberately replays the first packet for every candidate instead of
sharing cached prefix rollouts.  This makes each candidate a complete program
evaluation and matches the joint arm on all instrumented neural/model,
Jacobian, SVD, and scoring counts.  It is not a FLOP or scientific-admission
claim.
"""
from copy import deepcopy
import math
import time

import numpy as np

from experiments.spectral_reachability import codex_p2_program_library_pair_20261030_v1 as old

NAME = "p2_matched_program_kernel_20261101_parallel"
CAPS = dict(
    packet=44,
    action_encoder=44,
    predict=44,
    predictor=44,
    pred_proj=44,
    jacobian=5,
    svd=5,
    score=17,
)
ARMS = ("joint_multistart", "prefix_chain")
MEASURED_AXES = tuple(CAPS)


def contract():
    return dict(
        schema=NAME + ".contract",
        arms=list(ARMS),
        order=[[base, arm] for base in range(4) for arm in ARMS],
        horizon=10,
        packets=2,
        candidates_per_arm=16,
        first_prefixes=4,
        local_directions_per_prefix=4,
        caps=deepcopy(CAPS),
        required_exact_counts=deepcopy(CAPS),
        measured_axes=list(MEASURED_AXES),
        wall_seconds=450.0,
        joint=(
            "four full-horizon anchors; each anchor has a full20 Jacobian and "
            "two right-singular directions with both signs"
        ),
        prefix_chain=(
            "four executed prefixes; each prefix has its own packet Jacobian "
            "and two local directions with both signs; every candidate replays "
            "its complete prefix and suffix"
        ),
        selection="terminal latent mean squared error; lowest index on ties",
        replay="whole selected two-packet winner once from original context",
        measured_resource_match=True,
        equal_total_compute=False,
        total_flops=None,
        energy=None,
        engine_backward_calls=None,
        known_roots=True,
        independent_confirmation=False,
        preregistered=False,
        after_previous_results_known=True,
        physical=False,
        P1_admission=False,
        P2_admission=False,
    )


class Budget:
    def __init__(self, emit, clock=time.monotonic):
        old.require(callable(emit) and callable(clock), "budget callbacks")
        self.emit = emit
        self.clock = clock
        self.start = self.last = clock()
        self.attempted = dict.fromkeys(CAPS, 0)
        self.returned = dict.fromkeys(CAPS, 0)
        self.failed = False
        self.depth = 0

    def tick(self):
        now = self.clock()
        old.require(math.isfinite(now) and now >= self.last, "monotonic clock")
        self.last = now
        return now - self.start

    def call(self, kind, fn, details=None):
        elapsed = self.tick()
        old.require(
            not self.failed
            and kind in CAPS
            and self.attempted[kind] < CAPS[kind]
            and elapsed < contract()["wall_seconds"]
            and callable(fn),
            "budget predispatch",
        )
        ordinal = self.attempted[kind] + 1
        self.emit(
            dict(
                kind="resource_attempt",
                resource=kind,
                ordinal=ordinal,
                depth=self.depth,
                elapsed_seconds=elapsed,
                details=details,
            )
        )
        self.attempted[kind] = ordinal
        self.depth += 1
        try:
            result = fn()
        except BaseException as exc:
            self.failed = True
            self.emit(
                dict(
                    kind="resource_failure",
                    resource=kind,
                    ordinal=ordinal,
                    depth=self.depth - 1,
                    error_type=type(exc).__name__,
                    message=str(exc)[:1000],
                )
            )
            raise
        finally:
            self.depth -= 1
        elapsed = self.tick()
        self.returned[kind] += 1
        self.emit(
            dict(
                kind="resource_return",
                resource=kind,
                ordinal=ordinal,
                depth=self.depth,
                elapsed_seconds=elapsed,
            )
        )
        if elapsed >= contract()["wall_seconds"]:
            self.failed = True
            raise RuntimeError("in-flight wall limit")
        return result

    def call_extra(self, kind, fn):
        return self.call(kind, fn)

    def snapshot(self):
        return dict(
            attempted=deepcopy(self.attempted),
            returned=deepcopy(self.returned),
            wall_seconds=self.tick(),
            failed=self.failed,
        )


def local_basis(jacobian):
    matrix = np.asarray(jacobian, dtype=np.float64)
    old.require(
        matrix.shape == (192, 10) and np.isfinite(matrix).all(),
        "local one-packet Jacobian",
    )
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    for row in right:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return singular, right


def run_branch(case, arm, adapter_factory, emit, model_scope, clock=time.monotonic):
    old.require(
        arm in ARMS
        and type(case["base"]) is int
        and case["base"] in range(4)
        and callable(adapter_factory)
        and callable(emit)
        and callable(model_scope),
        "fixed branch and callbacks",
    )
    initial = deepcopy(case["context"])
    goal = np.asarray(case["goal"], dtype=np.float64)
    old.require(goal.shape == (192,) and np.isfinite(goal).all(), "goal shape")

    def event(value):
        emit(dict(base=case["base"], arm=arm, event=value))

    budget = Budget(event, clock)
    event(
        dict(
            kind="branch_start",
            context_sha256=old.digest(initial),
            goal_sha256=old.digest(goal.tolist()),
            caps=CAPS,
        )
    )
    with model_scope(budget):
        adapter = adapter_factory(budget)
        root = adapter.root_jacobian(initial)
        root_singular, basis = budget.call(
            "svd",
            lambda: old.canonical_basis(root),
            dict(scope="root", shape=[192, 20]),
        )
        library, _ = old.candidates(basis)
        candidates = []
        contexts = []

        if arm == "joint_multistart":
            for prefix_index, first in enumerate(library[0]):
                anchor = np.asarray(first["logits"] + [0.0] * 10, dtype=np.float64)
                jacobian = adapter.full_jacobian_at(deepcopy(initial), anchor)
                singular, local = budget.call(
                    "svd",
                    lambda jacobian=jacobian: old.canonical_basis(jacobian),
                    dict(
                        scope="joint_anchor",
                        prefix_index=prefix_index,
                        shape=[192, 20],
                    ),
                )
                contexts.append(
                    dict(
                        prefix_index=prefix_index,
                        anchor_logits=anchor.tolist(),
                        jacobian=np.asarray(jacobian, dtype=np.float64).tolist(),
                        singular_values=singular.tolist(),
                        basis=local.tolist(),
                    )
                )
                event(
                    dict(
                        kind="joint_anchor_return",
                        prefix_index=prefix_index,
                        anchor_logits=anchor.tolist(),
                    )
                )
                for mode in range(2):
                    for sign in (-1, 1):
                        logits = anchor + sign * (old.CAP / math.sqrt(2)) * local[mode]
                        norm = float(np.linalg.norm(logits))
                        if norm > old.CAP:
                            logits = logits * (old.CAP / norm)
                        raw = np.tanh(logits).astype("<f4").reshape(10, 2)
                        mid = adapter.step(deepcopy(initial), raw[:5].copy())
                        final = adapter.step(deepcopy(mid), raw[5:].copy())
                        terminal = np.asarray(
                            final["latent_history"][-1], dtype=np.float64
                        )
                        old.require(
                            terminal.shape == (192,) and np.isfinite(terminal).all(),
                            "terminal latent",
                        )
                        cost = budget.call(
                            "score", lambda terminal=terminal: float(np.mean((terminal - goal) ** 2))
                        )
                        item = dict(
                            index=len(candidates),
                            prefix_index=prefix_index,
                            mode=mode,
                            sign=sign,
                            logits=logits.tolist(),
                            raw_float32_hex=raw.tobytes().hex(),
                            prefix_context_sha256=old.digest(mid),
                            prefix_context=deepcopy(mid),
                            terminal_context_sha256=old.digest(final),
                            terminal_context=deepcopy(final),
                            cost=cost,
                        )
                        event(dict(kind="candidate_return", candidate=deepcopy(item)))
                        candidates.append(item)
        else:
            for prefix_index, first in enumerate(library[0]):
                first_raw = np.frombuffer(
                    bytes.fromhex(first["raw_float32_hex"]), dtype="<f4"
                ).reshape(5, 2).copy()
                mid = adapter.step(deepcopy(initial), first_raw)
                mid_hash = old.digest(mid)
                event(
                    dict(
                        kind="prefix_return",
                        prefix_index=prefix_index,
                        program_id=first["program_id"],
                        raw_float32_hex=first["raw_float32_hex"],
                        context_sha256=mid_hash,
                    )
                )
                jacobian = adapter.packet_jacobian(deepcopy(mid))
                singular, local = budget.call(
                    "svd",
                    lambda jacobian=jacobian: local_basis(jacobian),
                    dict(scope="prefix", prefix_index=prefix_index, shape=[192, 10]),
                )
                contexts.append(
                    dict(
                        prefix_index=prefix_index,
                        context=deepcopy(mid),
                        context_sha256=mid_hash,
                        jacobian=np.asarray(jacobian, dtype=np.float64).tolist(),
                        singular_values=singular.tolist(),
                        basis=local.tolist(),
                    )
                )
                for mode in range(2):
                    for sign in (-1, 1):
                        logits = sign * (old.CAP / math.sqrt(2)) * local[mode]
                        second = np.tanh(logits).astype("<f4").reshape(5, 2)
                        candidate_mid = adapter.step(deepcopy(initial), first_raw.copy())
                        candidate_mid_hash = old.digest(candidate_mid)
                        old.require(
                            candidate_mid_hash == mid_hash,
                            "candidate prefix replay differs from compiled prefix",
                        )
                        candidate_index = len(candidates)
                        event(
                            dict(
                                kind="candidate_prefix_return",
                                candidate_index=candidate_index,
                                prefix_index=prefix_index,
                                raw_float32_hex=first["raw_float32_hex"],
                                context_sha256=candidate_mid_hash,
                            )
                        )
                        final = adapter.step(deepcopy(candidate_mid), second)
                        terminal = np.asarray(
                            final["latent_history"][-1], dtype=np.float64
                        )
                        old.require(
                            terminal.shape == (192,) and np.isfinite(terminal).all(),
                            "terminal latent",
                        )
                        cost = budget.call(
                            "score", lambda terminal=terminal: float(np.mean((terminal - goal) ** 2))
                        )
                        item = dict(
                            index=candidate_index,
                            prefix_index=prefix_index,
                            mode=mode,
                            sign=sign,
                            first_raw_float32_hex=first_raw.tobytes().hex(),
                            second_raw_float32_hex=second.tobytes().hex(),
                            prefix_context_sha256=mid_hash,
                            candidate_prefix_context_sha256=candidate_mid_hash,
                            terminal_context_sha256=old.digest(final),
                            terminal_context=deepcopy(final),
                            cost=cost,
                        )
                        event(dict(kind="candidate_return", candidate=deepcopy(item)))
                        candidates.append(item)

        winner = min(candidates, key=lambda item: (item["cost"], item["index"]))
        if arm == "joint_multistart":
            raw = np.frombuffer(
                bytes.fromhex(winner["raw_float32_hex"]), dtype="<f4"
            ).reshape(10, 2)
            first_raw, second_raw = raw[:5].copy(), raw[5:].copy()
        else:
            first_raw = np.frombuffer(
                bytes.fromhex(winner["first_raw_float32_hex"]), dtype="<f4"
            ).reshape(5, 2).copy()
            second_raw = np.frombuffer(
                bytes.fromhex(winner["second_raw_float32_hex"]), dtype="<f4"
            ).reshape(5, 2).copy()
        replay_mid = adapter.step(deepcopy(initial), first_raw)
        replay_end = adapter.step(deepcopy(replay_mid), second_raw)
        terminal = np.asarray(replay_end["latent_history"][-1], dtype=np.float64)
        replay_cost = budget.call(
            "score", lambda: float(np.mean((terminal - goal) ** 2))
        )
        old.require(
            old.digest(replay_mid) == winner["prefix_context_sha256"]
            and old.digest(replay_end) == winner["terminal_context_sha256"]
            and replay_cost == winner["cost"],
            "whole winner replay exact",
        )
        replay = dict(
            prefix_context_sha256=old.digest(replay_mid),
            terminal_context_sha256=old.digest(replay_end),
            cost=replay_cost,
        )
        event(dict(kind="winner_replayed", index=winner["index"], replay=replay))

    resources = budget.snapshot()
    old.require(
        not resources["failed"]
        and resources["attempted"] == resources["returned"] == CAPS,
        "exact measured resource vector required",
    )
    result = dict(
        schema=NAME + ".branch",
        base=case["base"],
        arm=arm,
        context_sha256=old.digest(initial),
        goal_sha256=old.digest(goal.tolist()),
        root_jacobian=np.asarray(root, dtype=np.float64).tolist(),
        root_singular_values=root_singular.tolist(),
        root_basis=basis.tolist(),
        candidates=candidates,
        winner=winner,
        replay=replay,
        prefix_contexts=contexts,
        resource_counts=resources,
        measured_resource_vector=deepcopy(CAPS),
        measured_resource_match=True,
        equal_total_compute=False,
        P2_admission=False,
        physical=False,
    )
    event(
        dict(
            kind="branch_complete",
            result_sha256=old.digest(result),
            resource_counts=resources,
        )
    )
    return result
