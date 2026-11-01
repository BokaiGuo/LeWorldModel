"""Outcome-free dev303 physical-prefix contract bridge; no execution entry point.

The selector integration must normalize its sealed report to REPORT_SCHEMA.
The caller must obtain ``source`` through ``verified_source``; build_contract
is separately exposed to facilitate CPU-only contract tests.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

NAME = "p1_dev303_physical_prefix_bridge_20261101_exclusive"
REPORT_SCHEMA = NAME + ".selector-report.v1"
SEED = 301000003
QUEUE_SHA = "ec5656ea3c907844d3f8374af7f55aee9c65c9fe5196bbcd195717248618eb0e"
RECEIPT_SHA = "4d7c23525d745278ba29f46676c88dddd1a95aae44c4e46241fb71cdd60af78d"
ROLES = ("selected", "zero", "task_greedy_direct", "random")
LABELS = ("zero", "spectral", "gn", "rgb_nearest_contact",
          "rgb_goal_support_direct", "rgb_goal_support_top_route",
          "rgb_goal_support_bottom_route", "random_control",
          "rgb_delayed_task_support", "rgb_delayed_support_x12",
          "rgb_delayed_support_y12", "rgb_delayed_support_x23",
          "rgb_delayed_support_y23", "rgb_delayed_x12_pos06",
          "rgb_delayed_x12_neg06", "rgb_delayed_x23_pos06",
          "rgb_delayed_x23_neg04")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def verified_source():
    """Verify the existing queue/receipt chain and return only source identities."""
    from . import p1_dev303_blind_adapter_20261101_codex as adapter

    require(adapter.QUEUE_SHA == QUEUE_SHA, "dev303 queue identity")
    arrays, receipt_sha, audit = adapter.source()
    require(receipt_sha == RECEIPT_SHA and audit["seed"] == SEED
            and audit["P1_admission"] is False, "dev303 source identity")
    directory = adapter.QUEUE.parent / (adapter.QUEUE.name + ".native_reset") / "dev_holdout03_0000.warmup"
    receipt_raw = (directory / "source_receipt.json").read_bytes()
    require(digest(receipt_raw) == receipt_sha, "receipt drift")
    receipt = decode(receipt_raw)
    reset_sha = digest((directory / "reset_receipt.json").read_bytes())
    require(receipt["history"]["reset_receipt_sha256"] == reset_sha, "reset binding")
    return {
        "seed": SEED, "queue_sha256": QUEUE_SHA, "source_receipt_sha256": receipt_sha,
        "reset_receipt_sha256": reset_sha,
        "current_rgb_pixel_sha256": digest(arrays["rgb_history"][-1].tobytes()),
        "goal_rgb_pixel_sha256": digest(arrays["goal_rgb"].tobytes()),
    }


def _actions(row):
    try:
        arr64 = np.asarray(row["actions"], dtype=np.float64)
        require(arr64.shape == (5, 2) and np.isfinite(arr64).all()
                and (np.abs(arr64) <= 1).all(), "action shape/range")
        arr = arr64.astype("<f4")
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid actions") from exc
    require(digest(arr.tobytes()) == row["actions_sha256"], "action SHA")
    return arr


def build_contract(source, prepare_raw, prepare_sha, report_raw, report_sha):
    """Validate a normalized report against a sealed V12 preparation and source.

    ``source`` is trusted only when produced by verified_source().  Caller-supplied
    SHA values identify seals but do not by themselves establish seal authority.
    """
    require(digest(prepare_raw) == prepare_sha and digest(report_raw) == report_sha,
            "sealed input SHA")
    prep, report = decode(prepare_raw), decode(report_raw)
    require(source["seed"] == SEED and source["queue_sha256"] == QUEUE_SHA
            and source["source_receipt_sha256"] == RECEIPT_SHA, "source identity")
    require(prep["schema"] == "p1_dev303_v12_adapter_20261101_exclusive.prepare.v1"
            and prep["source_receipt_sha256"] == RECEIPT_SHA
            and prep["pixel_sha256"] == [source["current_rgb_pixel_sha256"],
                                          source["goal_rgb_pixel_sha256"]]
            and prep["candidate_status"] == "ready" and prep["P1_admission"] is False,
            "prepared source binding")
    require(report["schema"] == REPORT_SCHEMA and report["status"] == "model_only_sealed"
            and report["prepare_sha256"] == prepare_sha and report["source"] == source
            and report["selection_input"] == "rgb_only"
            and report["simulator_state_used"] is False
            and report["physical_outcomes_used"] is False
            and report["P1_admission"] is False and report["errors"] == [],
            "selector seal and RGB-only boundary")
    candidates = prep["prefix_bank"]["candidates"]
    scores = report["scores"]
    require(len(candidates) == len(scores) == len(LABELS), "complete candidate roster")
    require([r["index"] for r in candidates] == list(range(len(LABELS)))
            and [r["label"] for r in candidates] == list(LABELS), "candidate identity")
    arrays = [_actions(row) for row in candidates]
    require(not arrays[0].any(), "zero baseline")
    require(all(type(s["robust_score"]) in (int, float)
                and type(s["cp5_q"]) in (int, float)
                and math.isfinite(s["robust_score"]) and math.isfinite(s["cp5_q"])
                for s in scores), "finite scores")
    feasible = [i for i, row in enumerate(candidates) if row["contact_feasible_prior"] is True]
    require(feasible, "selector abstention")
    indices = {
        "selected": min(feasible, key=lambda i: (scores[i]["robust_score"], i)),
        "zero": 0,
        "task_greedy_direct": 4,
        "random": 7,
    }
    require(report["role_indices"] == indices, "selector decision mismatch")
    roles = [{"role": role, "candidate_index": indices[role],
              "candidate_label": candidates[indices[role]]["label"],
              "actions": arrays[indices[role]].tolist(),
              "actions_sha256": candidates[indices[role]]["actions_sha256"]}
             for role in ROLES]
    return {"schema": NAME + ".contract.v1", "source": dict(source),
            "prepare_sha256": prepare_sha, "selector_report_sha256": report_sha,
            "roles": roles, "selection_input": "rgb_only", "feedback_boundary": 5,
            "physical_execution_authorized": False, "independent_confirmation": False,
            "P1_admission": False}


def prepare_contract(prepare_path, prepare_sha, report_path, report_sha):
    """Read only explicitly named, sealed preparation and selector report files."""
    def read(path):
        path = Path(path)
        require("outcome" not in str(path).lower() and "physical" not in str(path).lower(),
                "outcome/physical input forbidden")
        require(path.is_file() and not path.is_symlink(), "regular sealed input")
        return path.read_bytes()
    return build_contract(verified_source(), read(prepare_path), prepare_sha,
                          read(report_path), report_sha)
