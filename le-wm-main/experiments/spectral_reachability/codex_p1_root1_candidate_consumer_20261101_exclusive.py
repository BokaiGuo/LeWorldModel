"""Independent saved-candidate consumer: no producer import, inference or physics."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_arm_consumer_20261030 as ledger

ROOT = Path(__file__).resolve().parents[2]
NAME = "codex_p1_root1_candidate_consumer_20261101_exclusive"
CONTRACT = ROOT / "results/codex_p1_root1_candidate_binding_20261101_exclusive_CODEX01.json"
CONTRACT_SHA = "2469d38911edec6777d20343b76dcecef93714fc841d9839ba123d2956446ee3"
IDENTITY = ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value, shape):
    value = np.asarray(value, dtype="<f4")
    require(value.shape == shape and np.isfinite(value).all() and (np.abs(value) <= 1).all(), "actions")
    return value


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def make_expected(original, optimized, point):
    original, optimized = array(original, (3, 5, 2)), array(optimized, (3, 5, 2))
    prefix, hold, tail = original[0], original[1:], optimized[:2]
    return {"prefix": prefix.tolist(), "hold": hold.tolist(), "optimized": tail.tolist(),
            "prefix_float32_sha256": digest(prefix), "hold_float32_sha256": digest(hold),
            "optimized_float32_sha256": digest(tail),
            "required_cp5": {key: point[key] for key in IDENTITY}}


def check_entry(entry, original, optimized, point):
    expected = make_expected(original, optimized, point)
    require(all(entry[k] == v for k, v in expected.items()), "candidate/identity mismatch")


def check_selection(event, role):
    bank = np.asarray(event["bank"], dtype="<f4")
    require(bank.ndim == 4 and bank.shape[1:] == (3, 5, 2)
            and np.isfinite(bank).all() and (np.abs(bank) <= 1).all(), "bank")
    scores = np.asarray(event["stage_scores"], dtype=np.float64)
    require(event["role"] == role and scores.shape == (len(bank),) and np.isfinite(scores).all(), "role/scores")
    idx = event["original_selected_index"]
    require(type(idx) is int and 0 <= idx < len(bank), "index")
    chosen = np.zeros((3, 5, 2), dtype="<f4") if role == "zero" else bank[
        int(np.argmin(scores)) if role == "greedy" else idx]
    require(np.array_equal(array(event["full_selected"], (3, 5, 2)), chosen), "selection")


def event(rows, kind):
    hits = [r["payload"] for r in rows if r["kind"] == kind]
    require(len(hits) == 1, "trace event")
    return hits[0]


def verify():
    require(sha(CONTRACT) == CONTRACT_SHA, "contract digest")
    contract = json.loads(CONTRACT.read_bytes())
    summaries = []
    for role in ("selected", "zero", "greedy"):
        entry = contract["candidates"][role]
        source = entry["source"]
        report_path = Path(source["report"])
        require(sha(report_path) == source["report_sha256"], "report digest")
        report = json.loads(report_path.read_bytes())
        require(report["protocol"]["root_slot"] == 1 and report["protocol"]["prefix_role"] == role
                and report["cp10_goal_binding"] == contract["goal_binding"]
                and report["model_before"] == report["model_after"] == contract["model"], "identity")
        require(sha(report["prepared"]) == source["prepare_sha256"], "prepare digest")
        row = report["rows"][0]
        point = next(p for p in row["checkpoints"] if p["step"] == 5)
        frame = Path(point["path"])
        require(sha(frame) == point["file_sha256"], "frame digest")
        with Image.open(frame) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"], "RGB")
        plans = {}
        for boundary in (0, 5):
            node = next(n for n in report["nodes"]["s1_structured"] if n["boundary"] == boundary)
            seal = json.loads((report_path.parent / f"s1_structured_b{boundary:02}_seal.json").read_bytes())
            receipt = node["receipt"]
            require(node["status"] == "sealed" and seal["receipt"] == receipt, "node seal")
            trace = Path(receipt["trace"])
            require(sha(trace) == receipt["trace_sha256"] == source[f"boundary{boundary}_trace_sha256"],
                    "trace digest")
            rows = ledger._parse(trace.read_bytes())
            require(rows[-1]["kind"] == "terminal" and rows[-1]["payload"]["status"] == "complete_node",
                    "trace status")
            plan = event(rows, "selected_full_plan")
            require(plan["boundary"] == boundary, "boundary")
            plans[boundary] = plan["raw"]
            if boundary == 0:
                prefix_event = event(rows, "prefix_intervention")
                check_selection(prefix_event, role)
                require(prefix_event["full_selected"] == plans[0], "prefix plan")
            account = receipt["account"]
            saved = ledger.consume(receipt["ledger"]["journal"], receipt["ledger"]["sha256"],
                                   account["contract_sha256"], "structured", "node", account["run_id"])
            require(saved == account and account["charged_upper_bounds"] ==
                    entry["historical_node_charged_upper_bounds"][str(boundary)], "ledger")
            require(np.array_equal(array(receipt["commands"], (5, 2)),
                                   array(plans[boundary], (3, 5, 2))[0]), "commands")
        require(np.array_equal(array(row["executed_raw"][:5], (5, 2)),
                               array(plans[0], (3, 5, 2))[0]), "prefix dispatch")
        check_entry(entry, plans[0], plans[5], point)
        summaries.append({"role": role, "prefix_sha256": entry["prefix_float32_sha256"],
                          "hold_sha256": entry["hold_float32_sha256"],
                          "optimized_sha256": entry["optimized_float32_sha256"],
                          "cp5_pixel_sha256": point["pixel_sha256"]})
    return {"status": "verified_saved_candidates_only", "contract_sha256": CONTRACT_SHA,
            "roles": summaries, "no_neural_or_physical_replay": True, "independent_confirmation": False,
            "execution_authorization": False, "full_cost_verified": False}


def main():
    output = ROOT / "results" / (NAME + "_CODEX01.json")
    result = verify()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output)}))


if __name__ == "__main__":
    main()
