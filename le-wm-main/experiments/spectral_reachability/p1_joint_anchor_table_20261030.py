"""Bind fixed P1 prefix identities to independently optimized physical outcomes."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "h10_prepare": ("results/p1_joint_h10_20261030_prepare_RUN01.json", "71579e186113a71d5571006de8519633950b05c32e887bb86563db1bba946b34"),
    "physical_prepare": ("results/p1_joint_cp5_physical20_20261030_prepare_RUN01.json", "21d8f77908968543e7257cf1ff77d5cd855def1ae9c016f08e8b7a2add70b372"),
    "physical_consumed": ("results/p1_joint_cp5_physical20_20261030_consume_RUN01.json", "fa76f75b95a48a32c161b9937e6a4c0e307f127f16226c5c6915269bcc6e796b"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(state, goal):
    if state is None:
        return None
    require(len(state) == len(goal) == 7, "state dimensions")
    require(all(math.isfinite(float(x)) for x in state + goal), "nonfinite state")
    joint = math.dist(state[:4], goal[:4])
    raw = abs(state[4] - goal[4])
    native_angle = min(raw, 2 * math.pi - raw)
    return {
        "object_position_error": math.dist(state[2:4], goal[2:4]),
        "object_angle_error_wrapped": abs(math.atan2(math.sin(state[4] - goal[4]), math.cos(state[4] - goal[4]))),
        "pusher_position_error": math.dist(state[:2], goal[:2]),
        "native_joint_xy_error": joint,
        "native_angle_term": native_angle,
        "native_angle_negative": native_angle < 0,
        "native_success": joint < 20 and native_angle < math.pi / 9,
    }


def build(prepared, plan, observed):
    anchors, roster, rows = prepared["anchors"], plan["roster"], observed["rows"]
    require(len(anchors) == len(roster) == len(rows) == 24, "incomplete roster")
    require(plan["P1_admission"] is False and observed["P1_admission"] is False, "admission")
    output = []
    for index, (anchor, item, row) in enumerate(zip(anchors, roster, rows, strict=True)):
        require(all(x["index"] == index for x in (anchor, item, row)), "index")
        for key in ("root_slot", "condition", "arm", "role"):
            require(anchor[key] == item[key] == row[key], "identity " + key)
        require(anchor["logical_pair"] == item["logical_pair"]
                and anchor["task_id"] == item["task_id"], "task identity")
        require(anchor["prefix5_float32_hex"] == item["prefix5_float32_hex"]
                and anchor["checkpoint"]["pixel_sha256"] == item["cp5_pixel_sha256"], "prefix identity")
        root = metrics(anchor["root_state"], anchor["goal"])
        prefix = metrics(anchor["checkpoint"]["state"], anchor["goal"])
        endpoint = metrics(row["completed_endpoint"], anchor["goal"])
        require((endpoint is not None) == (row["status"] == "probe_completed"), "endpoint censoring")
        if endpoint is not None:
            actual = row["last_accepted_state_metrics"]
            require(endpoint["native_success"] == actual["native_success"], "native success mismatch")
            require(math.isclose(endpoint["object_position_error"], actual["object_position_error"],
                                 rel_tol=1e-12, abs_tol=1e-10), "object position mismatch")
        names = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error",
                 "native_joint_xy_error", "native_angle_term")
        output.append({
            "index": index, "logical_pair": anchor["logical_pair"],
            "root_slot": anchor["root_slot"], "condition": anchor["condition"],
            "arm": anchor["arm"], "role": anchor["role"], "task_id": anchor["task_id"],
            "physical_root_sha256": anchor["physical_root_sha256"],
            "prefix_action_sha256": anchor["prefix5_sha256"],
            "prefix_cp5_pixel_sha256": anchor["checkpoint"]["pixel_sha256"],
            "root_metrics": root, "prefix_metrics": prefix,
            "prefix_minus_root": {k: prefix[k] - root[k] for k in names},
            "prefix_native_success_transition": [root["native_success"], prefix["native_success"]],
            "independent_suffix_20step": {"endpoint": endpoint, "status": row["status"],
                                           "returned_suffix": row["returned_suffix"],
                                           "raw10_sha256": item["raw10_sha256"]},
            "common_suffix": None,
            "common_suffix_status": "full panel not yet consumed",
        })
    return {"schema": "p1_joint_anchor_table_20261030", "rows": output,
            "fixed_anchors": 24, "P1_admission": False, "equal_budget_certified": False,
            "independent_confirmation": False,
            "limitations": "Known roots; duplicated trajectories; native angle follows original unwrapped formula; common panel pending."}


def main():
    loaded, digests = {}, {}
    for name, (relative, expected) in FILES.items():
        path = ROOT / relative
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected, "input hash " + name)
        loaded[name], digests[name] = json.loads(raw), actual
    result = build(loaded["h10_prepare"], loaded["physical_prepare"], loaded["physical_consumed"])
    result["input_sha256"] = digests
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = ROOT / "results/p1_joint_anchor_table_20261030_RUN01.json"
    with output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False, indent=2) + "\n").encode())
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "rows": len(result["rows"]), "common_panel_complete": False}))


if __name__ == "__main__":
    main()
