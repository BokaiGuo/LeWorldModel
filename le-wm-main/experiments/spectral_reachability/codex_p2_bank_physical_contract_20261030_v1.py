"""Exploratory bank-bound physical preparation. Physical dispatch remains blocked.

The branch primitive below is testable with synthetic environments; it is not
an authorized production entry point or a substitute for a cross-entry claim.
"""
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import codex_p2_bank_bound_saved_consumer_20261030_v1 as saved
from experiments.spectral_reachability import p2_composition_predict_frontier_physical_20261027_v1 as physical

frontier = saved.frontier
need = saved.need
NAME = "codex_p2_bank_physical_contract_20261030_v1"


def build(prepared_dir, prepared_sha256, reports, source_bindings):
    """Read only sealed planning sources, physical references and four new reports.

    reports maps every fixed root to (absolute directory, externally retained
    report byte SHA256). No physical outcome file is read.
    """
    _, protocol, prepared = frontier.load_prepared(prepared_dir, prepared_sha256)
    saved.bridge.check_sources(source_bindings)
    need(set(reports) == set(frontier.ROOTS), "exact fixed four roots")
    required = set(protocol["source_paths"]) | set(protocol["bindings"])
    required |= {str(Path(__file__).resolve()), str(Path(saved.__file__).resolve()),
                 str(Path(physical.__file__).resolve()), str(Path(saved.bridge.__file__).resolve()),
                 str(Path(saved.bound.__file__).resolve())}
    need(required <= set(source_bindings), "complete required source inventory")
    for path, sha in protocol["bindings"].items():
        need(source_bindings[path] == sha, "inherited input binding")
    compilation = frontier.read(protocol["context"]["compilation"])
    cases = physical.io.index(protocol["context"]["cases"], ("root_sha256",))
    sources = physical.io.index(compilation["rows"], ("root_sha256",))
    need(set(cases) == set(sources) == {(r,) for r in frontier.ROOTS}, "source root census")
    refs = physical.helpers.validate_prepared(physical.PROJECT / physical.REFERENCES, physical.REFERENCES_SHA)
    packets = physical.io.index(refs["packets"], ("root_sha256",))
    need(set(packets) == set(cases), "physical reference census")
    models = refs["model_bindings"]
    need({str((Path(protocol["context"]["policy"]) / n).resolve()) for n in ("weights.pt", "config.json")} == set(models),
         "physical reference model identity")
    for path, sha in models.items():
        need(source_bindings.get(path) == sha, "reference model bytes")
    plans = []
    report_records = {}
    for index, root in enumerate(frontier.ROOTS):
        case = cases[root,]
        c = sources[root,]["compiled"]
        packet, ref = packets[root,], refs["references"][root]
        initial = dict(physical.planner.META, latent_history=c["input_context"]["emb"][0],
                       action_history=c["input_context"]["actions"][0][1:])
        need(case["root_index"] == index and case["root"]["root_id"] == root and
             case["root"]["context"] == initial == packet["root_context"] and
             packet["goal"] == c["goal_vectors"][0] and
             packet["projection"] == c["scoring"]["metric"]["projection"] and
             frontier.digest(packet["goal"]) == case["goal_vector_sha256"] and
             frontier.digest(packet["projection"]) == case["projection_sha256"] and
             c["scoring"]["metric"]["sha256"] == case["metric_sha256"] and
             case["root"]["root_rgb_sha256"] == ref["history_rgb_sha256"] and
             case["root"]["goal_rgb_sha256"] == ref["goal_rgb_sha256"], "physical root/goal/Q/RGB binding")
        directory, report_sha = reports[root]
        result = saved.consume(directory=directory, expected_report_sha256=report_sha, ready=case,
                               compiled=c, protocol=protocol, prepared=prepared, source_bindings=source_bindings)
        selected = {p["arm"]: p for p in result["plans"]}
        need(len(result["plans"]) == 2 and set(selected) == set(frontier.ARMS), "two-arm census")
        order = frontier.ARMS if index % 2 == 0 else frontier.ARMS[::-1]
        for arm in order:
            e = selected[arm]
            need(e["root_sha256"] == root and e["root_index"] == index and
                 e["root"] == case["root"] and e["action_domain"] == case["action_domain"] and
                 e["selection_uses_physical_outcome"] is False and
                 e["selection_sha256"] == frontier.digest({k: v for k, v in e.items() if k != "selection_sha256"}),
                 "selected source and selection binding")
            terminal = e["witness"]["receipt"]["terminal_context"]
            physical.planner.check_history(initial, terminal, e["commands"], protocol["context"]["normalization"])
            p = dict(schema=physical.NAME + ".plan", root_index=index, root_sha256=root, method=arm,
                     metric_label="source_Q", schedule="predict_frontier", active_metric_id=case["metric_sha256"],
                     planned_horizon_tokens=2, planned_commands=10, corrected_search_horizon=2,
                     commands=deepcopy(e["commands"]), goal=deepcopy(packet["goal"]),
                     predictor_context=deepcopy(initial), selected_predicted_context=deepcopy(terminal),
                     planner_report_sha256=report_sha, planner_journal_sha256=result["journal_sha256"],
                     selected_plan=deepcopy(e))
            p["selection_sha256"] = physical.digest(p)
            physical.validate_plan(p)
            plans.append(p)
        report_records[root] = dict(directory=str(directory), report_sha256=report_sha,
                                    journal_sha256=result["journal_sha256"])
    saved.bridge.check_sources(source_bindings)
    contract = dict(schema=NAME, plans=plans, reports=report_records, prepared_dir=str(prepared_dir),
                    prepared_sha256=prepared_sha256, source_bindings=deepcopy(source_bindings),
                    references=deepcopy(refs["references"]), root_seed=refs["root_seed"],
                    physical_limits=physical.contract(), status="prepared_physics_blocked",
                    physical_authorized=False, equal_total_compute=False, P1_admission=False, P2_admission=False)
    contract["contract_sha256"] = frontier.digest(contract)
    return contract


def checked_branch(*, plan, output, reset_and_warmup, warmup_counts, references,
                   verify_rgb_context, validate_reset_history, env, state_of, evaluate):
    """Internal primitive: only dispatch after exactly 15 returns and RGB/history checks."""
    physical.validate_plan(plan)
    def reset():
        before = deepcopy(warmup_counts())
        frames, history, goal = reset_and_warmup()
        after = warmup_counts()
        need(all(after[k] - before[k] == 15 for k in ("attempts", "returns")),
             "warmup must return exactly 15 steps")
        identity = verify_rgb_context(frames, goal, references[plan["root_sha256"]])
        identity.update(validate_reset_history(history, plan["predictor_context"]["action_history"]))
        return identity
    row = physical.physical_branch(plan=plan, output=output, reset=reset, env=env,
                                   state_of=state_of, evaluate=evaluate)
    physical.verify_branch(plan, row)
    return row


def execute_physics(*args, **kwargs):
    raise RuntimeError("BLOCKED: no reviewed cross-entry physical claim and complete runtime source seal")
