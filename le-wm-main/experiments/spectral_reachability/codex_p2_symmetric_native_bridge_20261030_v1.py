"""Native CUDA bridge preparation for the symmetric P2 kernel; no run CLI.

This is deliberately not a production runner: kernel dispatch/SVD events are
not durable, and its H3 domain is not the original four-root H2 contract.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_knownroot_host_cuda_20261030_v1 as host
from experiments.spectral_reachability import codex_p2_symmetric_budget_20261030_v1 as kernel

NAME = "codex_p2_symmetric_native_bridge_20261030_v1"
PREP = host.frontier.PROJECT / "results/codex_p2_knownroot_host_cuda_20261030_v1_prepared01"
PREP_SHA = "455038931b3db1a89f34cd5e95f9e40662b01291b84bfd4d4bd523d54892605e"


def bind_cases(protocol, compiled_rows):
    cases = protocol["context"]["cases"]
    host.need(tuple(c["root_sha256"] for c in cases) == host.frontier.ROOTS and
              [c["root_index"] for c in cases] == list(range(4)), "root census")
    index = {r["root_sha256"]: r["compiled"] for r in compiled_rows}
    host.need(len(compiled_rows) == len(index) == 4 and set(index) == set(host.frontier.ROOTS),
              "compiled root census")
    bound = []
    for c in cases:
        compiled = index[c["root_sha256"]]
        goal = compiled["goal_vectors"][0]
        projection = compiled["scoring"]["metric"]["projection"]
        host.need(host.frontier.digest(goal) == c["goal_vector_sha256"] and
                  host.frontier.digest(projection) == c["projection_sha256"], "goal/projection identity")
        source_context = compiled["input_context"]
        root_context = c["root"]["context"]
        host.need(source_context["emb"] == [root_context["latent_history"]] and
                  len(source_context["actions"]) == 1 and
                  source_context["actions"][0][-2:] == root_context["action_history"],
                  "compiled/root between-state context identity")
        g, q = np.asarray(goal, dtype=np.float64), np.asarray(projection, dtype=np.float64)
        host.need(g.shape == (192,) and q.shape == (192, 192) and
                  np.isfinite(g).all() and np.isfinite(q).all(), "goal/projection shape")
        host.need(c["action_domain"]["low"] == [-1., -1.] and
                  c["action_domain"]["high"] == [1., 1.] and
                  c["action_domain"]["action_block"] == 5, "action domain")
        bound.append(dict(root_index=c["root_index"], root_sha256=c["root_sha256"],
                          root_id=c["root"]["root_id"], context=c["root"]["context"],
                          context_sha256=host.frontier.digest(c["root"]["context"]),
                          goal=goal, projection=projection,
                          goal_sha256=c["goal_vector_sha256"],
                          projection_sha256=c["projection_sha256"],
                          action_domain=c["action_domain"]))
    return bound


def known_roots():
    manifest, protocol, _ = host.validate_prepare(PREP, PREP_SHA)
    rows = host.frontier.read(protocol["context"]["compilation"])["rows"]
    return manifest, protocol, bind_cases(protocol, rows)


def validate_logits(u, cap):
    u = np.asarray(u, np.float64)
    if u.shape != (30,) or not np.isfinite(u).all() or np.linalg.norm(u) > cap:
        raise ValueError("H3 joint L2 action-domain cap")
    return u


class NativeCudaBackend:
    """Native transition and full30 AD, with packet-level synchronous events.

    emit must durably write or raise; no fallback. This backend alone does not
    certify the enclosing planner, its SVD, full source closure, or run output.
    """
    def __init__(self, model, device, projection, emit, cap):
        import torch
        from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
        if device.type != "cuda" or not callable(emit):
            raise ValueError("CUDA device and durable event sink required")
        self.t, self.device, self.emit, self.cap = torch, device, emit, cap
        self.transition = compiler.make_transition(model, device)
        self.projection = np.asarray(projection, np.float64)
        if self.projection.shape != (192, 192) or not np.isfinite(self.projection).all():
            raise ValueError("projection")
        self.counts = dict(packet=0, jacobian=0)

    def clock(self):
        self.t.cuda.synchronize(self.device)
        return time.monotonic()

    def _terminal(self, context, u, capture):
        t = self.t
        state = t.tensor(context["latent_history"], dtype=t.float32, device=self.device)[None]
        history = t.tensor(context["action_history"], dtype=t.float32, device=self.device)[None]
        actions = t.tanh(u).reshape(3, 5, 2)
        contexts = []
        for i in range(3):
            self.emit(dict(kind="packet_attempt", index=self.counts["packet"], packet=i))
            self.counts["packet"] += 1
            state, history = self.transition(state, history, actions[i])
            self.clock()
            self.emit(dict(kind="packet_return", index=self.counts["packet"]-1, packet=i))
            if capture:
                c = dict(context, latent_history=state[0].detach().cpu().tolist(),
                         action_history=history[0].detach().cpu().tolist())
                contexts.append(c)
        return state[0, -1], contexts

    def rollout(self, context, u):
        u = validate_logits(u, self.cap)
        t = self.t
        with t.no_grad():
            terminal, contexts = self._terminal(context, t.tensor(u, dtype=t.float32, device=self.device), True)
        return contexts, self.projection @ terminal.detach().cpu().numpy().astype(np.float64)

    def jacobian(self, context, u):
        u = validate_logits(u, self.cap)
        t = self.t
        self.emit(dict(kind="jacobian_attempt", index=self.counts["jacobian"],
                       shape=[192, 30], mathematical_vjp_rows_requested=192))
        self.counts["jacobian"] += 1
        x = t.tensor(u, dtype=t.float32, device=self.device, requires_grad=True)
        j = t.autograd.functional.jacobian(lambda v: self._terminal(context, v, False)[0],
                                           x, vectorize=True)
        self.clock()
        self.emit(dict(kind="jacobian_return", index=self.counts["jacobian"]-1))
        return self.projection @ j.detach().cpu().numpy().astype(np.float64)


def preflight():
    _, protocol, cases = known_roots()
    return dict(schema=NAME+".preflight", root_count=len(cases),
                roots=[{k: c[k] for k in ("root_index", "root_sha256", "root_id",
                                          "context_sha256", "goal_sha256", "projection_sha256")}
                       for c in cases],
                policy=protocol["context"]["policy"], upstream_prepare_sha256=PREP_SHA,
                prior_horizon=2, proposed_horizon=3,
                projected_metric_adapter=True, production_ready=False,
                blockers=["kernel SVD/dispatch not durably metered",
                          "kernel action clipping does not enforce joint L2 cap",
                          "no independent saved consumer or complete replay-context check",
                          "new H3 task and full cost boundary not yet bound"],
                equal_total_compute=False, flops=None, P1_admission=False, P2_admission=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["preflight"])
    p.parse_args()
    print(json.dumps(preflight(), sort_keys=True, allow_nan=False))
