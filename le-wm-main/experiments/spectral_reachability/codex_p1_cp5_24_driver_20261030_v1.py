"""24-row fail-stop planner driver, with injected execution and persistence.

No CUDA, filesystem isolation, source closure or GPU lock is established here.
The caller must bind and enforce all of those properties separately.
"""
from . import codex_p1_three_prefix_blind_inputs_20261030_v1 as blind


def drive(bundle, run_row, persist):
    if (type(bundle) is not dict or set(bundle) !=
            {"schema", "rows", "fixed_roster", "planner_file_isolation_established",
             "GPU_run", "physical_run", "P1_admission"} or
            bundle["schema"] != "codex_p1_three_prefix_blind_inputs_v1" or
            bundle["fixed_roster"] != 24 or len(bundle["rows"]) != 24 or
            any(bundle[k] is not False for k in
                ("planner_file_isolation_established", "GPU_run", "physical_run", "P1_admission"))):
        raise ValueError("fixed blind bundle")
    for index, row in enumerate(bundle["rows"]):
        blind.validate_row(row)
        if row["index"] != index:
            raise ValueError("fixed roster order")
    if not callable(run_row) or not callable(persist):
        raise ValueError("driver callbacks")
    results = []
    for index, row in enumerate(bundle["rows"]):
        persist({"kind": "row_start", "index": index})
        try:
            result = run_row(row)
            if (type(result) is not dict or set(result) !=
                    {"index", "actions", "score", "ledger_receipt"} or result["index"] != index):
                raise ValueError("row result contract")
            persist({"kind": "row_complete", "index": index, "result": result})
        except BaseException as exc:
            persist({"kind": "row_failure", "index": index, "error_type": type(exc).__name__,
                     "message": str(exc)[:1000]})
            raise
        results.append(result)
    return results
