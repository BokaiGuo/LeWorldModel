"""Model-only terminal-Q incumbent guard; no physics, model calls or fitting.

Caller must certify both action domains and native prediction provenance.
This guard does not certify those facts or promise physical improvement.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

FIELDS = {'context_sha256', 'goal192', 'commands', 'predicted192', 'domain_verified', 'prediction_verified'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def evaluate(candidate):
    require(type(candidate) is dict and set(candidate) == FIELDS, 'model-only exact fields; no evaluator labels')
    h = candidate['context_sha256']
    require(type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h), 'actual context SHA')
    require(candidate['domain_verified'] is True and candidate['prediction_verified'] is True, 'caller must verify actual domain/prediction')
    raw = np.asarray(candidate['commands'], np.float32)
    goal = np.asarray(candidate['goal192'], np.float64)
    z = np.asarray(candidate['predicted192'], np.float64)
    require(raw.shape == (5, 2) and np.isfinite(raw).all() and (abs(raw) <= 1).all() and
            np.array_equal(raw.astype(np.float64), np.asarray(candidate['commands'], np.float64)), 'literal float32 raw5, no clipping/cast repair')
    require(goal.shape == z.shape == (192,) and np.isfinite(goal).all() and np.isfinite(z).all(), 'finite native goal/prediction')
    return float(np.mean((z-goal)**2))


def choose(incumbent, proposal):
    """Stable exact argmin: keep original incumbent on equality, no tolerance."""
    old_q, new_q = evaluate(incumbent), evaluate(proposal)
    require(incumbent['context_sha256'] == proposal['context_sha256'] and
            incumbent['goal192'] == proposal['goal192'], 'same actual causal context and original goal')
    key = 'proposal' if new_q < old_q else 'incumbent'
    candidate = deepcopy(proposal if key == 'proposal' else incumbent)
    selected_q = new_q if key == 'proposal' else old_q
    require(selected_q <= old_q, 'modeled terminal-Q monotonicity')
    payload = dict(selected=key, candidate=candidate, selected_Q=selected_q, incumbent_Q=old_q,
        proposal_Q=new_q, P1_admission=False, P2_admission=False, physical_improvement_guaranteed=False,
        provenance_certified_by_guard=False)
    payload['selection_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return payload
