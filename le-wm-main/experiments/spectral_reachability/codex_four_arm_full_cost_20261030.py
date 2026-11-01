"""Four-arm prospective contract and in-memory, fail-stop dispatch accounting.

This is not instrumentation: every billable backend boundary must be routed
through dispatch, and reservations must be independently established upper
bounds. A hash is not proof of preregistration time or hook completeness.
"""
import copy
import hashlib
import json
import threading

NAME = "codex_four_arm_full_cost_20261030"
ARMS = ("structured", "native", "random", "immediate_progress")
IDENTITY = ("root", "task", "model", "action_domain", "history", "goal",
            "optimizer", "optimizer_config", "meter", "hardware")
AXES = ("encode_calls", "encode_samples", "action_encode_calls",
        "predict_calls", "predict_samples", "backward_calls", "vjp_rows",
        "jacobian_calls", "selection_evaluations", "solve_calls", "svd_calls",
        "replay_calls", "physical_steps")
PHASES = ("setup", "initialization", "jacobian", "screening", "trial",
          "common_suffix", "verification", "physical")
KINDS = {"encode": "encode_calls", "action_encode": "action_encode_calls",
         "predict": "predict_calls", "backward": "backward_calls",
         "jacobian": "jacobian_calls", "selection": "selection_evaluations",
         "solve": "solve_calls", "svd": "svd_calls", "replay": "replay_calls",
         "physical": "physical_steps"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def is_hash(x):
    return type(x) is str and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def vector(value):
    require(type(value) is dict and set(value) == set(AXES), "complete resource vector required")
    require(all(type(v) is int and v >= 0 for v in value.values()), "nonnegative integer resources required")
    return value


def preflight(contract, expected_sha):
    """Validate structure and external digest; never authorize production."""
    require(type(contract) is dict and set(contract) ==
            {"schema", "identity", "arms", "levels", "scope", "phase_policy_sha256"},
            "exact contract fields")
    require(contract["schema"] == NAME + ".contract.v1", "schema")
    require(contract["scope"] == "standalone_per_root_per_arm_including_attributable_setup",
            "standalone full-cost scope")
    require(type(contract["identity"]) is dict and set(contract["identity"]) == set(IDENTITY)
            and all(is_hash(x) for x in contract["identity"].values()), "identity hashes")
    require(is_hash(contract["phase_policy_sha256"]), "phase policy hash")
    arms = contract["arms"]
    require(type(arms) is dict and set(arms) == set(ARMS), "exact four arms")
    for arm in ARMS:
        item = arms[arm]
        require(type(item) is dict and set(item) == {"initializer_sha256", "optimizer_sha256",
                                                    "optimizer_config_sha256"}, "arm identity")
        require(all(is_hash(x) for x in item.values()), "arm hashes")
        require(item["optimizer_sha256"] == contract["identity"]["optimizer"]
                and item["optimizer_config_sha256"] == contract["identity"]["optimizer_config"],
                "common optimizer and configuration")
    levels = contract["levels"]
    require(type(levels) is list and len(levels) >= 2, "at least two prospective budget levels")
    previous = None
    names = set()
    for level in levels:
        require(type(level) is dict and set(level) == {"id", "caps"}, "level fields")
        name = level["id"]
        require(type(name) is str and name and name.strip() == name and name not in names, "level id")
        names.add(name)
        caps = vector(level["caps"])
        if previous is not None:
            require(all(caps[k] >= previous[k] for k in AXES)
                    and any(caps[k] > previous[k] for k in AXES), "strictly increasing vector curve")
        previous = caps
    require(is_hash(expected_sha) and digest(contract) == expected_sha, "contract digest mismatch")
    return {"contract_sha256": expected_sha, "levels": [x["id"] for x in levels],
            "arms": list(ARMS), "structurally_valid": True,
            "commitment_time_verified": False, "instrumentation_verified": False,
            "fairness_proven": False, "execution_authorized": False}


class Ledger:
    """Per-arm, per-level cooperative reservation guard; no persistence.

    Use leaf boundaries only; do not nest wrapped calls. Higher-level operations
    such as Jacobian need their own nonnested accounting event and their internal
    model/backward calls must also be separately routed and charged.
    """
    def __init__(self, contract, contract_sha, arm, level, run_id):
        preflight(contract, contract_sha)
        require(arm in ARMS and type(run_id) is str and run_id.strip() == run_id and run_id,
                "arm/run")
        matches = [x for x in contract["levels"] if x["id"] == level]
        require(len(matches) == 1, "unknown level")
        self.contract_sha, self.arm, self.level, self.run_id = contract_sha, arm, level, run_id
        self.caps = copy.deepcopy(matches[0]["caps"])
        self.charged = dict.fromkeys(AXES, 0)
        self.events = []
        self.failed = self.closed = False
        self.lock = threading.Lock()

    def dispatch(self, phase, kind, reservation, fn):
        if not self.lock.acquire(blocking=False):
            self.failed = True
            raise ValueError("nested or concurrent dispatch forbidden")
        try:
            require(not self.closed and not self.failed, "ledger inactive")
            require(phase in PHASES and kind in KINDS and callable(fn), "phase/kind/callback")
            vector(reservation)
            require(reservation[KINDS[kind]] > 0, "event must charge its operation")
            require((kind != "encode" or reservation["encode_samples"] > 0)
                    and (kind != "predict" or reservation["predict_samples"] > 0),
                    "batch samples required")
            reserved = dict(reservation)
            if any(self.charged[k] + reserved[k] > self.caps[k] for k in AXES):
                self.events.append({"phase": phase, "kind": kind, "status": "blocked",
                                    "reservation": reserved})
                self.failed = True
                raise ValueError("budget exceeded before dispatch")
            for k in AXES:
                self.charged[k] += reserved[k]
            event = {"phase": phase, "kind": kind, "status": "attempted", "reservation": reserved}
            self.events.append(event)
            try:
                result = fn()
                require(not self.failed, "nested or concurrent dispatch attempted")
            except BaseException:
                event["status"] = "failed"
                self.failed = True
                raise
            event["status"] = "completed"
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def close(self):
        require(self.lock.acquire(blocking=False), "ledger busy")
        try:
            require(not self.closed, "ledger closed")
            self.closed = True
            return {"schema": NAME + ".ledger.v1", "contract_sha256": self.contract_sha,
                    "arm": self.arm, "level": self.level, "run_id": self.run_id,
                    "status": "failed_or_blocked" if self.failed else "closed_unverified",
                    "charged_upper_bounds": dict(self.charged), "events": copy.deepcopy(self.events),
                    "hook_coverage_verified": False, "actual_usage_verified": False,
                    "fairness_proven": False, "execution_authorized": False}
        finally:
            self.lock.release()
