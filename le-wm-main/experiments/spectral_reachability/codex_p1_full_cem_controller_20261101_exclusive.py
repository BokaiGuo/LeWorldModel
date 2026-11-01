"""Prospective P1 full-episode CEM controller; adapter integration required.

No runtime launcher, model loader, environment constructor, or GPU access is
provided here. All callbacks are supplied by a separately reviewed adapter.
"""
import hashlib
import math

import numpy as np
import torch

from .codex_e1_proposal_durable_events_20261101 import ProposalJournal
from .codex_four_arm_raw_candidates_20261030 import _normal

BOUNDARIES = (0, 5, 10, 15)
HORIZONS = (3, 2, 1, 1)
ITERATIONS, SAMPLES, ELITES = 4, 16, 4


def tensor_sha(value):
    a = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class MeteredLeaf:
    def __init__(self, leaf, journal):
        self.leaf, self.journal = leaf, journal
        self.encode_attempts = 0

    def encode(self, info, *, phase="setup"):
        self.encode_attempts += 1
        with self.journal.attempt("encode", details={"calls": 1}):
            return self.leaf.encode(info, phase=phase)

    def action_encode(self, actions, *, phase):
        with self.journal.attempt("action_encode", details={"calls": 1}):
            return self.leaf.action_encode(actions, phase=phase)

    def predict(self, states, actions, *, phase):
        with self.journal.attempt("predict", details={"calls": 1, "samples": len(states)}):
            return self.leaf.predict(states, actions, phase=phase)


class Controller:
    """Sequential controller. An exception permanently stops this instance.

    observe(boundary, metered_leaf) supplies tensors states/history/goal/mean/std/matrix,
    rgb_sha256, provenance, identity and encoding_mode. The latter is
    "fresh_metered" (exactly one metered encode) or "inherited".
    issue(packet) returns a dictionary
    containing integer executed_steps and boolean stopped. The adapter must
    independently authenticate the identity and actual RGB/history: matching
    caller-provided strings alone cannot establish provenance.
    """
    def __init__(self, *, leaf, observe, issue, journal, run_id, identity, seed):
        required = {"model_sha256", "root_sha256", "goal_sha256", "normalizer_sha256",
                    "metric_sha256", "task_sha256"}
        if not isinstance(identity, dict) or set(identity) != required or not all(_hash(v) for v in identity.values()):
            raise ValueError("six fixed identity hashes required")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("invalid seed")
        self.identity, self.seed = dict(identity), seed
        self.leaf, self.observe, self.issue = leaf, observe, issue
        self.journal = ProposalJournal(journal, run_id=run_id, arm="full_cem",
                                       proposal_kind="p1_full_episode_cem")
        self.index, self.stopped = 0, False
        try:
            self.journal.append({"type": "contract", "identity": self.identity,
                "boundaries": BOUNDARIES, "horizons": HORIZONS,
                "iterations": ITERATIONS, "samples": SAMPLES, "elites": ELITES,
                "seed": seed, "identity_authenticated_by_adapter": False,
                "equal_cost": False, "shared_and_upstream_cost": "unknown_unallocated"})
        except BaseException:
            self.journal.close()
            raise

    def close(self):
        self.journal.close()

    def step(self):
        if self.stopped or self.index >= len(BOUNDARIES):
            raise RuntimeError("controller stopped or complete")
        boundary, horizon = BOUNDARIES[self.index], HORIZONS[self.index]
        try:
            leaf = MeteredLeaf(self.leaf, self.journal)
            with self.journal.attempt("observe", details={"boundary": boundary}):
                obs = self.observe(boundary, leaf)
            if not isinstance(obs, dict) or obs.get("identity") != self.identity:
                raise ValueError("identity mismatch")
            if (obs.get("encoding_mode"), leaf.encode_attempts) not in (("fresh_metered", 1), ("inherited", 0)):
                raise ValueError("encoding mode and metered calls mismatch")
            if not _hash(obs.get("rgb_sha256")) or not obs.get("provenance"):
                raise ValueError("actual observation provenance required")
            states, history, goal, mean, std, matrix = (obs[k] for k in
                ("states", "history", "goal", "mean", "std", "matrix"))
            tensors = (states, history, goal, mean, std, matrix)
            if not all(isinstance(t, torch.Tensor) and t.is_floating_point()
                       and t.device == states.device and torch.isfinite(t).all().item() for t in tensors):
                raise ValueError("finite aligned tensors required")
            d = states.shape[-1] if states.ndim == 3 else 0
            if (states.shape != (1, 3, d) or d < 1 or history.shape != (1, 2, 10)
                or goal.shape != (1, d) or mean.shape != (2,) or std.shape != (2,)
                or matrix.shape != (d, d) or not (std > 0).all().item()
                or any(t.dtype != states.dtype for t in (history, goal, mean, std))):
                raise ValueError("invalid context, objective or normalizer")
            if tensor_sha(goal) != self.identity["goal_sha256"] or tensor_sha(matrix) != self.identity["metric_sha256"]:
                raise ValueError("goal or metric bytes mismatch")
            normalizer = torch.stack((mean, std))
            if tensor_sha(normalizer) != self.identity["normalizer_sha256"]:
                raise ValueError("normalizer bytes mismatch")
            self.journal.append({"type": "context", "boundary": boundary,
                "rgb_sha256": obs["rgb_sha256"], "provenance": obs["provenance"],
                "states_sha256": tensor_sha(states), "history_sha256": tensor_sha(history),
                "encoding_mode": obs["encoding_mode"],
                "identity_authenticated_by_adapter": False})
            rng = np.random.default_rng(self.seed + self.index)
            center = np.zeros((horizon, 5, 2), np.float32)
            scale = np.full_like(center, .5)
            plans, scores = [], []
            def score(raw):
                with self.journal.attempt("rollout", details={"boundary": boundary, "horizon": horizon}):
                    plan = torch.as_tensor(raw, dtype=states.dtype, device=states.device).unsqueeze(0)
                    with torch.no_grad():
                        predicted = _normal(leaf, "verification", states, history, plan, mean, std)
                        residual = (predicted - goal).double()
                        value = ((residual @ matrix.double().T).square().mean()).item()
                    if not math.isfinite(value):
                        raise ValueError("nonfinite score")
                    return float(value)

            for iteration in range(ITERATIONS):
                bank = np.clip(rng.normal(center, scale, (SAMPLES, horizon, 5, 2)), -1, 1).astype(np.float32)
                bank[0] = center
                self.journal.append({"type": "bank", "boundary": boundary, "iteration": iteration,
                                     "raw": bank.tolist(), "mean": center.tolist(), "std": scale.tolist()})
                values = []
                for candidate in bank:
                    value = score(candidate)
                    values.append(value)
                    plans.append(candidate.copy())
                    scores.append(value)
                    self.journal.append({"type": "score", "boundary": boundary,
                                         "index": len(scores) - 1, "value": value})
                elites = sorted(range(SAMPLES), key=lambda i: (values[i], i))[:ELITES]
                center = bank[elites].mean(axis=0, dtype=np.float32)
                scale = np.clip(bank[elites].std(axis=0, dtype=np.float32), .05, 1.).astype(np.float32)
                self.journal.append({"type": "update", "boundary": boundary, "iteration": iteration,
                                     "elite_indices": elites, "mean": center.tolist(), "std": scale.tolist()})
            winner = min(range(len(scores)), key=lambda i: (scores[i], i))
            plan = plans[winner]
            self.journal.append({"type": "selection", "boundary": boundary, "index": winner,
                                 "score": scores[winner], "raw": plan.tolist()})
            replay = score(plan)
            if not np.isclose(replay, scores[winner], rtol=1e-6, atol=1e-8):
                raise ValueError("winner replay mismatch")
            self.journal.append({"type": "winner_replay", "boundary": boundary, "score": replay})
            packet = plan[0].copy()
            with self.journal.attempt("issue_packet", details={"boundary": boundary, "raw": packet.tolist()}):
                receipt = self.issue(packet)
            if (not isinstance(receipt, dict) or type(receipt.get("executed_steps")) is not int
                or not 0 <= receipt["executed_steps"] <= 5 or type(receipt.get("stopped")) is not bool
                or (receipt["executed_steps"] != 5 and not receipt["stopped"])):
                raise ValueError("invalid execution receipt")
            self.journal.append({"type": "issue_receipt", "boundary": boundary, "receipt": receipt})
            self.index += 1
            self.stopped = receipt["stopped"]
            return {"boundary": boundary, "horizon": horizon, "selected_index": winner,
                    "selected_raw": plan.copy(), "receipt": receipt}
        except BaseException as exc:
            self.stopped = True
            try:
                self.journal.append({"type": "controller_failure", "boundary": boundary,
                                     "exception_type": type(exc).__name__, "message": str(exc)})
            except BaseException:
                pass
            raise
