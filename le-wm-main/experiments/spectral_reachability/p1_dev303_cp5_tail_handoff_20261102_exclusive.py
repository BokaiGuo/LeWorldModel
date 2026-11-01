"""Read-only dev303 tail handoff and fail-closed replay gate; no physical runner."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as planner
from . import p1_dev303_cp5_rgb_action_export_20261102_exclusive as export
from . import p1_dev303_rgb_chunk_saved_20261102_exclusive as saved

ROOT = planner.ROOT
NAME = "p1_dev303_cp5_tail_handoff_20261102_exclusive"
PREP = ROOT / "results/p1_dev303_cp5_encoded_v13_replan_20261102_exclusive_PREP_DEV01.json"
REPORT = ROOT / "results/p1_dev303_cp5_encoded_v13_replan_20261102_exclusive_RUN_DEV01/report.json"
PREP_SHA = "e5e5cfc630bb32e03b7b6a88e7c2fc2d8f23793182ab06e0124e2791f0e1154e"
REPORT_SHA = "817e86b6855dcab634791fc9361c5362742d9081ce4c8138b108adcd2fdc51fd"
ARMS = ("rgb_chunk", "zero_prefix")
BANK_SHA = "ab0ca17ef053daad3b31f795d53ce48a0579237e3f481c2d02ec436921ce31ea"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def actions(value, shape):
    raw = np.asarray(value, dtype=np.float64)
    require(raw.shape == shape and np.isfinite(raw).all()
            and np.all(np.abs(raw) <= 1), "action shape, finite values or range")
    result = raw.astype("<f4")
    require(np.array_equal(result.astype(np.float64), raw), "actions not exact float32")
    return result


def build():
    """Verify original physical prefix and sealed model-only planning evidence."""
    prep = export.decode(planner.parent.read_fixed(PREP, PREP_SHA))
    report = export.decode(planner.parent.read_fixed(REPORT, REPORT_SHA))
    require(prep == planner.protocol(), "planner protocol source closure")
    saved.verify(export.PREP, export.PREP_SHA, export.RUN_DIR, export.REPORT_SHA)
    metadata, arrays = export.build()
    require(report["schema"] == planner.NAME + ".report.v1"
            and report["status"] == "model_only_complete_not_admitted"
            and report["protocol_sha256"] == PREP_SHA
            and report["encode_report_sha256"] == planner.ENCODE_REPORT_SHA
            and report["encoded_contexts_sha256"] == planner.CONTEXTS_SHA
            and report["host"]["device"] == "cuda:0"
            and report["model_before"] == report["model_after"] == prep["model_hash"]
            and report["physical_execution"] is False
            and report["physical_actions_authorized"] is False
            and report["P1_admission"] is False
            and report["independent_confirmation"] is False, "planning report boundary")
    require([r["role"] for r in report["rows"]] == list(ARMS)
            and [r["role"] for r in prep["roles"]] == list(ARMS)
            and [r["arm"] for r in metadata["roles"]] == list(ARMS), "fixed arm roster")
    original = export.decode(export.read_regular(export.RUN_DIR / "report.json"))
    roles = []
    for model_row, role, origin, meta in zip(
            report["rows"], prep["roles"], original["rows"], metadata["roles"], strict=True):
        arm = role["role"]
        history = actions(role["actual_prefix"], (5, 2))
        require(np.array_equal(history, arrays[arm + "_raw_action_history"][-5:])
                and digest(history.tobytes()) == role["actual_prefix_sha256"]
                == model_row["actual_prefix_sha256"], "actual action history binding")
        require(role["cp5_pixel_sha256"] == model_row["cp5_pixel_sha256"]
                == meta["cp5_pixel_sha256"]
                == digest(arrays[arm + "_rgb_history"][-1].tobytes()), "cp5 RGB binding")
        require(model_row["budget"] == {"action_encoder_calls": 10, "predict_calls": 10,
                                       "predict_samples": 3200}
                and model_row["hooks"] == {"action_encoder": 10, "predict": 10}
                and model_row["initial_bank_sha256"] == BANK_SHA
                and [x["iteration"] for x in model_row["iterations"]] == list(range(5)),
                "equal budget and initialization record")
        tail = actions(model_row["selected"]["actions"], (2, 5, 2)).reshape(10, 2)
        checkpoints = [p for p in origin["observations"] if p["phase"] == "control"]
        require([p["step"] for p in checkpoints] == list(range(1, 6)),
                "original replay checkpoint roster")
        roles.append({"arm": arm, "prefix": history.tolist(), "tail": tail.tolist(),
                      "tail_sha256": digest(tail.tobytes()),
                      "cp5_pixel_sha256": role["cp5_pixel_sha256"],
                      "prefix_checkpoints": checkpoints})
    return {"schema": NAME + ".handoff.v1", "classification": "development_only",
            "planner_report_sha256": REPORT_SHA, "planner_preparation_sha256": PREP_SHA,
            "prefix_report_sha256": export.REPORT_SHA, "roles": roles,
            "P1_admission": False, "independent_confirmation": False,
            "physical_actions_authorized": False,
            "limitation": "no physical runner or saved-output verifier implemented"}


class ReplayGate:
    """Pure per-arm replay gate. No I/O, no environment, no action dispatch.

    A consumer must separately check reset, warmup, goal, storage, source closure,
    real action receipts and handle errors. This class does not authorize execution.
    """

    def __init__(self, role):
        self.prefix = actions(role["prefix"], (5, 2))
        self.tail = actions(role["tail"], (10, 2))
        require(digest(self.tail.tobytes()) == role["tail_sha256"], "tail hash")
        self.checkpoints = role["prefix_checkpoints"]
        require([x["step"] for x in self.checkpoints] == list(range(1, 6)),
                "checkpoint roster")
        self.index = 0
        self.failed = False

    def accept_prefix(self, action, observation, *, terminated=False, truncated=False):
        try:
            require(not self.failed and self.index < 5, "prefix gate state")
            actual = actions([action], (1, 2))[0]
            expected = self.checkpoints[self.index]
            require(np.array_equal(actual, self.prefix[self.index])
                    and not terminated and not truncated
                    and all(observation[k] == expected[k]
                            for k in ("state", "snapshot", "pixel_sha256")),
                    "prefix replay mismatch")
            self.index += 1
        except Exception:
            self.failed = True
            raise

    def fixed_tail(self):
        require(not self.failed and self.index == 5, "prefix replay not verified")
        return self.tail.copy()
