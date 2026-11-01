"""Bounded, model-predicted action witness composition, never stage admission.

No neural calls are supplied here. A trusted caller provides a deterministic
replay adapter (HOST CUDA only for neural replay); arrays alone are fixtures.
Hashes bind declared content, not the truth or independence of its origin.
"""
from copy import deepcopy
import hashlib
import json
import math

PROVENANCE_KEYS = ("checkpoint_sha256", "encoder_sha256", "source_sha256",
                   "protocol_sha256", "metric_sha256", "candidate_bank_sha256")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit SHA256 required")


def _canonical(value):
    # Restrict to portable JSON, rejecting bool-as-number in numeric arrays below.
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("finite JSON content required") from exc


def digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _context(value):
    """Supported layout: L>=2 states with L-1 between-state actions.

    Missing incoming action before the first state is intentional, not padded.
    This validates a declared numeric layout, not completeness for a live model.
    """
    if not isinstance(value, dict) or not value:
        raise ValueError("nonempty full latent/action context required")
    if value.get("layout") != "states_with_between_state_actions_v1":
        raise ValueError("explicit supported context layout required")
    for key in ("context_length", "latent_dim", "action_dim"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError("positive declared context dimensions required")
    length = value["context_length"]
    if length < 2:
        raise ValueError("supported context requires at least two states")
    for key, rows, width in (("latent_history", length, value["latent_dim"]),
                             ("action_history", length-1, value["action_dim"])):
        matrix = value.get(key)
        if (not isinstance(matrix, list) or len(matrix) != rows or
                any(not isinstance(row, list) or len(row) != width or
                    any(not _number(x) for x in row) for row in matrix)):
            raise ValueError("rectangular finite numeric aligned histories required")
    _canonical(value)


def compose(*, root, provenance, action_domain, edges, replay, max_edges, max_actions):
    """Recompute each transition, bind exact contexts and concatenate actions.

    Actions are explicit per-environment-step commands [T,A], not latent goals
    or compiler parameters. No clipping, conversion, tolerance or waypoint
    substitution is performed. action_block records steps per compiler token.
    Output contexts include full predictor history, not merely terminal latent.
    Exact deterministic replay is intentional: a different runtime requires a
    separately frozen numerical contract rather than an implicit tolerance.
    """
    if any(type(n) is not int or n < 1 for n in (max_edges, max_actions)):
        raise ValueError("positive integer composition bounds required")
    if not callable(replay) or not isinstance(edges, list) or not 1 <= len(edges) <= max_edges:
        raise ValueError("bounded nonempty edges and trusted replay required")
    # Work exclusively on owned snapshots; replay receives separate copies.
    root, provenance, domain, edges = deepcopy((root, provenance, action_domain, edges))
    try:
        if not isinstance(root["root_id"], str) or not root["root_id"]:
            raise ValueError("root identity required")
        for key in ("root_rgb_sha256", "goal_rgb_sha256"):
            _sha(root[key])
        if set(provenance) != set(PROVENANCE_KEYS):
            raise ValueError("complete explicit provenance required")
        for value in provenance.values():
            _sha(value)
        if domain["name"] != "raw" or not isinstance(domain["units"], str) or not domain["units"]:
            raise ValueError("raw environment action coordinates and units required")
        _sha(domain["normalization_sha256"])
        block = domain["action_block"]
        if type(block) is not int or block < 1:
            raise ValueError("positive action block required")
        low, high = domain["low"], domain["high"]
        if (not isinstance(low, list) or not isinstance(high, list) or not low
                or len(low) != len(high) or any(not _number(x) for x in low + high)
                or any(l >= h for l, h in zip(low, high))):
            raise ValueError("finite ordered action bounds required")
        context = root["context"]
        _context(context)
        actions, transitions, ids = [], [], set()
        for edge in edges:
            identity = edge["program_id"]
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError("unique nonempty program identities required")
            ids.add(identity)
            if edge["kind"] not in ("progress", "reconfiguration"):
                raise ValueError("no arbitrary latent waypoint programs")
            if (edge["root_id"] != root["root_id"] or edge["provenance"] != provenance
                    or edge["action_domain"] != domain):
                raise ValueError("root/provenance/action-domain mismatch")
            _context(edge["input_context"])
            _context(edge["output_context"])
            if any(edge["output_context"][k] != context[k]
                   for k in ("layout", "latent_dim", "action_dim")):
                raise ValueError("context feature layout changed across transition")
            if _canonical(edge["input_context"]) != _canonical(context):
                raise ValueError("disconnected full intermediate context")
            chunk = edge["actions"]
            if not isinstance(chunk, list) or not chunk or len(chunk) % block:
                raise ValueError("nonempty complete action blocks required")
            if len(actions) + len(chunk) > max_actions:
                raise ValueError("action horizon exceeds declared bound")
            for row in chunk:
                if (not isinstance(row, list) or len(row) != len(low)
                        or any(not _number(x) or not l <= x <= h for x, l, h in zip(row, low, high))):
                    raise ValueError("finite in-domain action witness required")
            predicted = replay(deepcopy(context), deepcopy(chunk))
            _context(predicted)
            if _canonical(predicted) != _canonical(edge["output_context"]):
                raise ValueError("intermediate transition not generated by action witness")
            transitions.append(dict(program_id=identity, kind=edge["kind"],
                start_step=len(actions), end_step=len(actions)+len(chunk),
                input_context_sha256=digest(context), actions_sha256=digest(chunk),
                output_context_sha256=digest(predicted), edge_sha256=digest(edge)))
            actions.extend(chunk)
            context = deepcopy(predicted)
        receipt = dict(schema="d2p_composition_witness.v2", status="contract_replayed_only",
            root=root, provenance=provenance, action_domain=domain,
            max_edges=max_edges, max_actions=max_actions, actions=actions,
            actions_sha256=digest(actions), transitions=transitions, terminal_context=context,
            P2_admission=False, physical_efficacy_verified=False,
            provenance_independently_verified=False)
        receipt["receipt_sha256"] = digest(receipt)
        return receipt
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("malformed composition contract") from exc
