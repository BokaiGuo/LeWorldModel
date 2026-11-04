"""Trace actual v2 depth-3 beam pruning of privileged, constructed witnesses."""
import argparse
import ast
import hashlib
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN


ROOT = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
        "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
        "action_history": [[0., 0.]]}
NAMES = ("audit_prefix", "audit_middle", "audit_terminal")
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parents[2] / "docs/PC_ATLAS_DEPTH3_INJECTION_PRUNING_PROTOCOL_20261103_CODEX.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def beam_assignment_line():
    source, start = inspect.getsourcelines(AtlasSearch.search)
    tree = ast.parse(textwrap.dedent("".join(source)))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Name) and target.id == "beam"
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Call)
                and isinstance(node.value.value.func, ast.Name)
                and node.value.value.func.id == "sorted"):
            matches.append(start + node.lineno - 1)
    if len(matches) != 1:
        raise RuntimeError("expected one v2 sorted beam assignment")
    return matches[0]


def node_record(node, compiler):
    return {"names": [edge["name"] for edge in node["edges"]],
            "actions": node["actions"], "cost": compiler._cost(node["context"]),
            "context_sha256": digest(node["context"])}


def run_traced(search, root):
    """Capture actual locals immediately after the unmodified assignment."""
    if sys.gettrace() is not None:
        raise RuntimeError("exclusive Python tracing required")
    target = beam_assignment_line()
    pending = set()
    snapshots = []

    def trace(frame, event, arg):
        if frame.f_code is not AtlasSearch.search.__code__:
            return trace
        if event == "line":
            if frame in pending:
                pending.remove(frame)
                local = frame.f_locals
                if local["root"]["root_name"] == NAMES[0]:
                    expanded = local["expanded"]
                    beam = local["beam"]
                    expected = sorted(expanded, key=lambda n: search.compiler._cost(n["context"]))[
                        :search.config.beam_width]
                    if len(beam) != len(expected) or any(a is not b for a, b in zip(beam, expected)):
                        raise RuntimeError("observed beam differs from stable sorted slice")
                    ordered = sorted(enumerate(expanded),
                                     key=lambda item: search.compiler._cost(item[1]["context"]))
                    ranks = {id(node): rank for rank, (_, node) in enumerate(ordered, 1)}
                    snapshots.append({
                        "depth": len(expanded[0]["edges"]),
                        "expanded": [{**node_record(n, search.compiler), "stable_rank": ranks[id(n)],
                                      "retained": any(n is kept for kept in beam)}
                                     for n in expanded],
                        "beam": [node_record(n, search.compiler) for n in beam]})
            if frame.f_lineno == target:
                pending.add(frame)
        return trace

    try:
        sys.settrace(trace)
        result = search.search(root)
    finally:
        sys.settrace(None)
    if pending or [s["depth"] for s in snapshots] != [2, 3]:
        raise RuntimeError("failed to observe both beam assignments")
    return result, snapshots


class WitnessInjection(AtlasSearch):
    def __init__(self, compiler, config, actions, context_hashes):
        super().__init__(compiler, config)
        self.actions = actions
        self.context_hashes = context_hashes
        self.visits = [0, 0, 0]

    def _bank(self, context, rng):
        bank = super()._bank(context, rng)
        context_hash = digest(context)
        for i, expected in enumerate(self.context_hashes):
            if context_hash == expected:
                self.visits[i] += 1
                bank.append((NAMES[i], self.actions[i:i + 1].copy()))
        return bank


def make_actions(saved, variant):
    original = np.asarray(saved["gradient_actions"], dtype=np.float64)
    if original.shape != (2, 2) or original[1, 1] != 0 or not np.isfinite(original).all():
        raise ValueError("expected saved bounded two-action witness with zero final turn")
    if variant == "early":
        actions = np.vstack((original, [0., 0.]))
    elif variant == "delayed":
        actions = np.vstack((original[0], [-.05, 0.], original[1] + [.05, 0.]))
    else:
        raise ValueError("unknown variant")
    if np.any(actions < [-1., -TURN]) or np.any(actions > [1., TURN]):
        raise ValueError("constructed action out of bounds")
    return actions


def audit_case(saved, variant):
    actions = make_actions(saved, variant)
    backend = UnicycleBackend()
    # Five deliberately unmetered backend calls for witness construction/checks.
    first = backend.rollout(ROOT, actions[:1])
    second = backend.rollout(first, actions[1:2])
    third = backend.rollout(second, actions[2:])
    direct = backend.rollout(ROOT, actions)
    split = backend.rollout(first, actions[1:])
    if digest(direct) != digest(split) or digest(direct) != digest(third):
        raise ValueError("direct, two-part and packet-wise witness mismatch")
    context_hashes = [digest(ROOT), digest(first), digest(second)]
    if len(set(context_hashes)) != 3:
        raise ValueError("injection contexts must be distinct")
    compiler = LocalCompiler(
        backend, goal=[*saved["goal"], 0.], sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
        config=CompilerConfig(packet_steps=1, spectral_modes=2,
                              max_rollout_operations=1000, max_jacobian_operations=100))
    search = WitnessInjection(compiler, AtlasConfig(depth=3, beam_width=2, seed=42 + saved["index"]),
                              actions, context_hashes)
    result, snapshots = run_traced(search, ROOT)
    intermediate = [entry for entry in snapshots[0]["expanded"] if entry["names"] == list(NAMES[:2])]
    if len(intermediate) != 1:
        raise RuntimeError("expected unique injected intermediate candidate")
    node = intermediate[0]
    if node["context_sha256"] != context_hashes[2]:
        raise RuntimeError("traced intermediate does not bind to injected context")
    terminal = [entry for entry in snapshots[1]["expanded"] if entry["names"] == list(NAMES)]
    if len(terminal) != int(node["retained"]):
        raise RuntimeError("named continuation disagrees with actual intermediate beam")
    if terminal and terminal[0]["context_sha256"] != digest(direct):
        raise RuntimeError("terminal candidate differs from full witness")
    branch = next(p for p in result["programs"] if p["root_name"] == NAMES[0])
    selected = result["programs"][result["selected_index"]]
    return {"index": saved["index"], "goal": saved["goal"], "variant": variant,
            "actions": actions.tolist(), "witness_input_context_sha256": context_hashes,
            "witness_output_sha256": digest(direct), "witness_terminal_latent": direct["latent_history"][-1],
            "witness_terminal_cost": compiler._cost(direct),
            "intermediate_rank": node["stable_rank"], "intermediate_cost": node["cost"],
            "intermediate_retained_in_actual_beam": node["retained"],
            "exact_named_terminal_offered": bool(terminal),
            "exact_named_terminal_retained": bool(terminal and terminal[0]["retained"]),
            "context_injection_visits": search.visits,
            "branch_winner_names": [e["name"] for e in branch["edges"]],
            "branch_winner_cost": branch["terminal_cost"],
            "selected_names": [e["name"] for e in selected["edges"]],
            "selected_terminal_cost": selected["terminal_cost"],
            "compiler_counts": result["operation_counts"],
            "external_witness_rollouts": 5, "trace": snapshots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.input.read_text())
    if source.get("schema") != "pc_atlas_oracle_injection.v1":
        raise ValueError("unexpected source schema")
    rows = source["rows"]
    if [r["index"] for r in rows] != list(range(8)):
        raise ValueError("expected eight ordered exposed goals")
    print("Trace actual depth-3 beam for immediate and delayed privileged witnesses.")
    cases = [audit_case(row, variant) for row in rows for variant in ("early", "delayed")]
    report = {"schema": "pc_atlas_depth3_pruning_trace.v1",
              "scope": "constructed privileged injection; not Atlas performance or independent confirmation",
              "input_sha256": sha(args.input),
              "dependency_sha256": {p.name: sha(p) for p in (
                  Path(__file__), PROTOCOL, HERE / "d2p_atlas_search_v2.py",
                  HERE / "d2p_local_compiler.py", HERE / "d2p_composition_witness_v2.py",
                  HERE / "pc_atlas_unicycle_smoke_20261103.py",
                  HERE / "pc_exact_unicycle_development.py")},
              "numpy_version": np.__version__, "python_version": sys.version,
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({variant: {"retained": sum(c["intermediate_retained_in_actual_beam"]
                                               for c in cases if c["variant"] == variant),
                               "ranks": [c["intermediate_rank"] for c in cases if c["variant"] == variant]}
                      for variant in ("early", "delayed")}))


if __name__ == "__main__":
    main()
