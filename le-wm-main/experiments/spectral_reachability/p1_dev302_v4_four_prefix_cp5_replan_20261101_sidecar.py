"""Dev302 V4 four-prefix actual-RGB equal-budget cp5 tail replanning.

This sidecar accepts an externally SHA-bound four-prefix physical report.  It
requires every physical prefix to match the already sealed V4 selected, zero,
same-bank-greedy, and random candidates byte-for-byte.  Each arm then receives
the same robust CEM budget for a ten-step (two packet) model-only tail.

Preparation is CPU/saved-only.  ``--run`` is host-CUDA only and is intentionally
not invoked by this implementation turn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_dev302_mixed_bank_cuda_v4_20261101_codex as v4
from . import p1_dev302_mixed_prefix_physical_v4_20261101_codex as physical_v4
from . import p1_unified_cp5_feedback_cem_20261101_codex as common
from . import preflight_pusht_d2p_p0 as p0


ROOT = blind.ROOT
NAME = "p1_dev302_v4_four_prefix_cp5_replan_20261101_sidecar"
V4_PREP = ROOT / "results/p1_dev302_mixed_bank_cuda_v4_20261101_codex_PREP_MAIN01.json"
V4_PREP_SHA256 = "ac0fd2082a1ab12582df58da35d847a0396355e3a5a4e6d391b150c1e8c268d1"
V4_REPORT = ROOT / "results/p1_dev302_mixed_bank_cuda_v4_20261101_codex_RUN_MAIN01/report.json"
V4_REPORT_SHA256 = "73a8cab5ac90aef32df33bbb1b02676fd373592cd1c8706a75d168a95c0a0db3"
PHYSICAL_PREP = (
    ROOT / "results/p1_dev302_mixed_prefix_physical_v4_20261101_codex_PREP_MAIN01.json"
)
PHYSICAL_PREP_SHA256 = "231d8ff9fa28d18a6de9446f6d2a4282a9fa8a9a089169add629504bd4220293"
PHYSICAL_REPORT = (
    ROOT
    / "results/p1_dev302_mixed_prefix_physical_v4_20261101_codex_RUN_MAIN01/report.json"
)
PHYSICAL_REPORT_SHA256 = "3c49e8ca83dce02b71bb44a6d41d38eab3490ef8adb9fd32119c062563186759"
MODEL_SHA256 = "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546"

ROLES = ("selected", "zero", "same_bank_greedy", "random")
ROLE_INDEX_FIELDS = {
    "selected": "selected_index",
    "zero": "zero_index",
    "same_bank_greedy": "same_bank_greedy_index",
    "random": "random_index",
}
ITERATIONS, SAMPLES, ELITES, VARIANTS, HORIZON = 5, 64, 8, 5, 2
INITIAL_STD, MIN_STD, SEED = 0.25, 0.03, 30253021101
WALL_SECONDS = 1800

AXIS_CAPS = {
    "rgb_decode": {"calls": 1, "samples": 1},
    "rgb_encode": {"calls": 1, "samples": 1},
    "candidate_generate": {"calls": ITERATIONS, "samples": ITERATIONS * SAMPLES},
    "variant_expand": {"calls": ITERATIONS, "samples": ITERATIONS * SAMPLES * VARIANTS},
    "action_encoder": {
        "calls": ITERATIONS * HORIZON,
        "samples": ITERATIONS * HORIZON * SAMPLES * VARIANTS,
    },
    "predict": {
        "calls": ITERATIONS * HORIZON,
        "samples": ITERATIONS * HORIZON * SAMPLES * VARIANTS,
    },
    "metric": {"calls": ITERATIONS, "samples": ITERATIONS * SAMPLES * VARIANTS},
    "robust_reduce": {"calls": ITERATIONS, "samples": ITERATIONS * SAMPLES},
    "elite_update": {"calls": ITERATIONS, "samples": ITERATIONS * ELITES},
    "selection": {"calls": 1, "samples": ITERATIONS * SAMPLES},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_sha(actions) -> str:
    value = np.asarray(actions, np.float32)
    return hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_bytes())
    require(type(value) is dict, "JSON object")
    return value


def write_new(path: Path, value) -> str:
    path = Path(path)
    payload = canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            require(count > 0, "short write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(payload).hexdigest()


def read_physical_journal(directory: Path, report: dict) -> list[dict]:
    path = directory / "journal.jsonl"
    require(sha256(path) == report["journal_sha256"], "physical journal SHA")
    raw = path.read_bytes()
    require(raw and raw.endswith(b"\n"), "complete physical journal")
    rows = [json.loads(line) for line in raw.splitlines()]
    previous = None
    for index, row in enumerate(rows):
        body = {key: value for key, value in row.items() if key != "sha256"}
        require(
            row.get("index") == index
            and row.get("previous") == previous
            and row.get("sha256") == digest(body),
            "physical journal hash chain",
        )
        previous = row["sha256"]
    require(previous == report["journal_tail"], "physical journal tail")
    return rows


def _v4_material() -> tuple[dict, dict, dict[int, dict]]:
    require(sha256(V4_PREP) == V4_PREP_SHA256, "V4 prep SHA")
    require(sha256(V4_REPORT) == V4_REPORT_SHA256, "V4 report SHA")
    prepared, report = read_json(V4_PREP), read_json(V4_REPORT)
    require(
        prepared == v4.protocol()
        and prepared["schema"] == v4.NAME + ".protocol.v1"
        and report["schema"] == v4.NAME + ".report.v1"
        and report["status"] == "model_only_complete"
        and report["protocol_sha256"] == V4_PREP_SHA256
        and report["physical_outcomes_used"] is False
        and report["model_before"] == report["model_after"]
        and report["model_before"]["original_state_dict"] == MODEL_SHA256
        and report["P1_admission"] is False,
        "sealed V4 planner",
    )
    rows = {row["index"]: row for row in report["rows"]}
    require(len(rows) == 8 and set(rows) == set(range(8)), "V4 candidate roster")
    return prepared, report, rows


def _read_cp5(directory: Path, row: dict) -> tuple[dict, Path]:
    points = [
        point
        for point in row.get("observations", [])
        if point.get("phase") == "prefix" and point.get("step") == 5
    ]
    require(len(points) == 1, "one actual cp5 observation")
    receipt = points[0]
    path = (directory / receipt["file"]).resolve(strict=True)
    require(path.is_relative_to(directory) and not path.is_symlink(), "cp5 path")
    require(sha256(path) == receipt["file_sha256"], "cp5 file SHA")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), np.uint8)
    require(
        pixels.shape == (224, 224, 3)
        and hashlib.sha256(pixels.tobytes()).hexdigest() == receipt["pixel_sha256"],
        "cp5 pixel SHA",
    )
    return receipt, path


def _unique_tails(rows: list[dict]) -> list[np.ndarray]:
    tails = [np.asarray(row["tail_selected"]["actions"], np.float32) for row in rows]
    tails.append(np.zeros((HORIZON, 5, 2), np.float32))
    unique = []
    for tail in tails:
        require(
            tail.shape == (HORIZON, 5, 2)
            and np.isfinite(tail).all()
            and np.max(np.abs(tail)) <= 1,
            "V4 seed tail",
        )
        if not any(np.array_equal(tail, old) for old in unique):
            unique.append(tail)
    require(2 <= len(unique) < SAMPLES, "fixed seed bank")
    return unique


def protocol(physical_directory: Path, physical_report_sha256: str) -> dict:
    directory = Path(physical_directory).resolve(strict=True)
    require(
        directory.parent == ROOT / "results"
        and not directory.is_symlink()
        and len(physical_report_sha256) == 64,
        "physical directory and external SHA",
    )
    report_path = directory / "report.json"
    require(sha256(report_path) == physical_report_sha256, "physical report SHA")
    report = read_json(report_path)
    require(sha256(PHYSICAL_PREP) == PHYSICAL_PREP_SHA256, "V4 physical prep SHA")
    physical_contract = read_json(PHYSICAL_PREP)
    require(
        physical_contract == physical_v4.build()
        and physical_contract.get("schema") == physical_v4.NAME + ".contract.v1"
        and physical_contract["planner"]["prep_sha256"] == V4_PREP_SHA256
        and physical_contract["planner"]["report_sha256"] == V4_REPORT_SHA256
        and report.get("schema") == physical_v4.NAME + ".report.v1"
        and report.get("status") == "recorded_development_v3_prefixes"
        and report.get("prepare_sha256") == PHYSICAL_PREP_SHA256
        and report.get("planner_prep_sha256") == V4_PREP_SHA256
        and report.get("planner_report_sha256") == V4_REPORT_SHA256
        and report.get("errors") == []
        and report.get("post_prefix_outcomes_used_for_planning") is False
        and report.get("new_neural_calls_during_physical") == 0
        and report.get("development_only") is True
        and report.get("independent_confirmation") is False
        and report.get("P1_admission") is False,
        "V4 four-prefix physical boundary",
    )
    read_physical_journal(directory, report)
    prepared, v4_report, v4_rows = _v4_material()
    physical_rows = report.get("rows", [])
    require(
        len(physical_rows) == 4
        and [row.get("role") for row in physical_rows] == list(ROLES),
        "ordered four-prefix roster",
    )
    expected_indices = {
        role: v4_report[field] for role, field in ROLE_INDEX_FIELDS.items()
    }
    canonical_v4_rows = [v4_rows[expected_indices[role]] for role in ROLES]
    seed_tails = _unique_tails(canonical_v4_rows)
    rows = []
    bindings = {
        str(path.resolve()): sha256(path.resolve())
        for path in (
            Path(__file__).resolve(),
            Path(v4.__file__).resolve(),
            Path(physical_v4.__file__).resolve(),
            Path(common.__file__).resolve(),
            Path(blind.__file__).resolve(),
            Path(p0.__file__).resolve(),
            Path(locking.__file__).resolve(),
            V4_PREP,
            V4_REPORT,
            PHYSICAL_PREP,
            report_path,
            directory / "journal.jsonl",
        )
    }
    for physical_row, role in zip(physical_rows, ROLES, strict=True):
        expected = v4_rows[expected_indices[role]]
        actual = np.asarray(
            [transition["action"] for transition in physical_row.get("transitions", [])],
            np.float32,
        )
        planned = np.asarray(expected["actions"], np.float32)
        require(
            physical_row.get("candidate_index") == expected["index"]
            and physical_row.get("candidate_label") == expected["label"]
            and physical_row.get("status") == "prefix_completed"
            and physical_row.get("error") is None
            and physical_row.get("plan_returned") == 5
            and actual.shape == planned.shape == (5, 2)
            and np.array_equal(actual, planned)
            and physical_row.get("planned_actions_sha256") == action_sha(planned),
            f"V4 physical prefix identity: {role}",
        )
        cp5, cp5_path = _read_cp5(directory, physical_row)
        bindings[str(cp5_path)] = cp5["file_sha256"]
        old_tail = np.asarray(expected["tail_selected"]["actions"], np.float32)
        rows.append(
            {
                "role": role,
                "candidate_index": expected["index"],
                "candidate_label": expected["label"],
                "actual_prefix": actual.tolist(),
                "actual_prefix_sha256": action_sha(actual),
                "cp5_png": str(cp5_path),
                "cp5_file_sha256": cp5["file_sha256"],
                "cp5_pixel_sha256": cp5["pixel_sha256"],
                "old_v4_tail": old_tail.tolist(),
                "old_v4_tail_sha256": action_sha(old_tail),
            }
        )
    for path, expected in prepared["bindings"].items():
        require(path not in bindings or bindings[path] == expected, "binding collision")
        bindings[path] = expected
    require(all(sha256(Path(path)) == value for path, value in bindings.items()), "source closure")
    return {
        "schema": NAME + ".protocol.v1",
        "physical_directory": str(directory),
        "physical_report_sha256": physical_report_sha256,
        "v4_prepare_sha256": V4_PREP_SHA256,
        "v4_report_sha256": V4_REPORT_SHA256,
        "physical_prepare_sha256": PHYSICAL_PREP_SHA256,
        "roles": rows,
        "task": prepared["task"],
        "policy": prepared["policy"],
        "model_sha256": MODEL_SHA256,
        "seed_tails": [tail.tolist() for tail in seed_tails],
        "algorithm": {
            "iterations": ITERATIONS,
            "samples": SAMPLES,
            "elites": ELITES,
            "variants": VARIANTS,
            "horizon_packets": HORIZON,
            "remaining_steps": HORIZON * 5,
            "initial_std": INITIAL_STD,
            "minimum_std": MIN_STD,
            "common_random_numbers": True,
            "iteration0_seeds": "same canonical V4 four-role tails plus zero for every arm",
            "score": "mean five terminal latent goal MSEs plus max-min range",
            "selection": "global minimum robust score with stable iteration/candidate ties",
        },
        "per_arm_axis_caps": AXIS_CAPS,
        "total_axis_caps": {
            axis: {
                key: value * len(ROLES)
                for key, value in cap.items()
            }
            for axis, cap in AXIS_CAPS.items()
        },
        "wall_seconds": WALL_SECONDS,
        "information_boundary": (
            "per-arm actual cp5 RGB and actual five-action prefix; sealed V4 task and tails; "
            "no post-cp5 physical outcome"
        ),
        "equal_cem_and_robust_budget": True,
        "post_cp5_physical_outcomes_used": False,
        "physical_execution": False,
        "training": False,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.fd = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        self.index = 0
        self.previous = None

    def emit(self, kind: str, **fields) -> str:
        body = {
            "index": self.index,
            "previous": self.previous,
            "kind": kind,
            **fields,
        }
        row = {**body, "sha256": digest(body)}
        payload = canonical(row) + b"\n"
        view = memoryview(payload)
        while view:
            count = os.write(self.fd, view)
            require(count > 0, "short journal write")
            view = view[count:]
        os.fsync(self.fd)
        self.index += 1
        self.previous = row["sha256"]
        return self.previous

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class AxisMeter:
    def __init__(self, role: str, journal: Journal, caps=None):
        require(role in ROLES, "meter role")
        self.role = role
        self.journal = journal
        self.caps = caps or AXIS_CAPS
        self.counts = {
            axis: {"attempted_calls": 0, "completed_calls": 0, "attempted_samples": 0,
                   "completed_samples": 0}
            for axis in self.caps
        }

    def call(self, axis: str, samples: int, function):
        require(axis in self.caps and type(samples) is int and samples > 0, "meter axis")
        row = self.counts[axis]
        cap = self.caps[axis]
        require(
            row["attempted_calls"] + 1 <= cap["calls"]
            and row["attempted_samples"] + samples <= cap["samples"],
            f"axis cap: {self.role}/{axis}",
        )
        call_index = row["attempted_calls"]
        started = time.monotonic()
        row["attempted_calls"] += 1
        row["attempted_samples"] += samples
        self.journal.emit(
            "resource_attempt",
            role=self.role,
            axis=axis,
            call_index=call_index,
            samples=samples,
            started_seconds=started,
        )
        value = function()
        ended = time.monotonic()
        row["completed_calls"] += 1
        row["completed_samples"] += samples
        self.journal.emit(
            "resource_return",
            role=self.role,
            axis=axis,
            call_index=call_index,
            samples=samples,
            started_seconds=started,
            ended_seconds=ended,
        )
        return value

    def settle(self) -> dict:
        for axis, cap in self.caps.items():
            row = self.counts[axis]
            require(
                row["attempted_calls"] == row["completed_calls"] == cap["calls"]
                and row["attempted_samples"] == row["completed_samples"] == cap["samples"],
                f"unsettled exact axis: {self.role}/{axis}",
            )
        return json.loads(json.dumps(self.counts))


def candidate_file_rows(path: Path) -> list[dict]:
    raw = Path(path).read_bytes()
    require(raw and raw.endswith(b"\n"), "complete candidate file")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == ITERATIONS * SAMPLES, "complete candidate denominator")
    return rows


def write_candidates(path: Path, rows: list[dict]) -> str:
    require(len(rows) == ITERATIONS * SAMPLES, "candidate denominator")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        for row in rows:
            payload = canonical(row) + b"\n"
            view = memoryview(payload)
            while view:
                count = os.write(fd, view)
                require(count > 0, "short candidate write")
                view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256(path)


def run(prepared: Path, prepared_sha256: str, output: Path) -> str:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    prepared = Path(prepared).resolve(strict=True)
    value = read_json(prepared)
    require(
        sha256(prepared) == prepared_sha256
        and value == protocol(Path(value["physical_directory"]), value["physical_report_sha256"]),
        "prepared protocol",
    )
    output = Path(output)
    require(
        output.parent == ROOT / "results"
        and output.name.startswith(NAME + "_RUN_")
        and not os.path.lexists(output),
        "exclusive run output",
    )
    output.mkdir(mode=0o700)
    journal = Journal(output / "events.jsonl")
    started = time.monotonic()
    report = {
        "schema": NAME + ".report.v1",
        "status": "failed",
        "error": None,
        "protocol_sha256": prepared_sha256,
        "physical_report_sha256": value["physical_report_sha256"],
        "rows": [],
        "model_before": None,
        "model_after": None,
        "host": None,
        "module_hooks": None,
        "journal_tail": None,
        "journal_events": None,
        "wall_seconds": None,
        "equal_cem_and_robust_budget": True,
        "post_cp5_physical_outcomes_used": False,
        "physical_execution": False,
        "training": False,
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    handles = []
    try:
        journal.emit("run_started", protocol_sha256=prepared_sha256, roles=list(ROLES))
        device = torch.device("cuda:0")
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        report["host"] = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(value["policy"], device=device).eval().requires_grad_(False)
        report["model_before"] = blind.host.signal.model_hash(model)
        require(report["model_before"]["original_state_dict"] == MODEL_SHA256, "model identity")
        hooks = {"action_encoder": 0, "predict": 0}
        handles = [
            model.action_encoder.register_forward_pre_hook(
                lambda _m, _a: hooks.__setitem__(
                    "action_encoder", hooks["action_encoder"] + 1
                )
            ),
            model.predictor.register_forward_pre_hook(
                lambda _m, _a: hooks.__setitem__("predict", hooks["predict"] + 1)
            ),
        ]
        task = value["task"]
        seed_tails = np.asarray(value["seed_tails"], np.float32)
        tensor = lambda item: torch.as_tensor(item, dtype=torch.float32, device=device)
        mean_action, std_action = tensor(task["action_mean"]), tensor(task["action_std"])
        goal = tensor(task["goal"])[None]
        expected_hook_calls = 0
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(
            SDPBackend.MATH
        ):
            for role_row in value["roles"]:
                require(time.monotonic() - started < WALL_SECONDS, "wall cap")
                role = role_row["role"]
                meter = AxisMeter(role, journal)
                journal.emit(
                    "arm_started",
                    role=role,
                    cp5_pixel_sha256=role_row["cp5_pixel_sha256"],
                    actual_prefix_sha256=role_row["actual_prefix_sha256"],
                )

                def decode():
                    with Image.open(role_row["cp5_png"]) as image:
                        pixels = np.asarray(image.convert("RGB"), np.uint8)
                    require(
                        pixels.shape == (224, 224, 3)
                        and hashlib.sha256(pixels.tobytes()).hexdigest()
                        == role_row["cp5_pixel_sha256"],
                        "runtime cp5 pixels",
                    )
                    return pixels

                pixels = meter.call("rgb_decode", 1, decode)

                def encode():
                    image = p0._image_tensor(
                        pixels, transform=p0.image_transform(), device=device
                    )
                    latent = model.encode({"pixels": image})["emb"][0, 0]
                    require(
                        latent.shape == (192,) and bool(torch.isfinite(latent).all()),
                        "cp5 latent",
                    )
                    return latent

                latent = meter.call("rgb_encode", 1, encode)
                states = tensor(
                    task["root_context"]["latent_history"][-2:] + [latent.cpu().tolist()]
                )[None]
                prefix = tensor(role_row["actual_prefix"])
                prefix_token = ((prefix - mean_action) / std_action).reshape(1, 1, 10)
                history = torch.cat(
                    (tensor(task["root_context"]["action_history"][-1:])[None], prefix_token),
                    dim=1,
                )
                old_tail = np.asarray(role_row["old_v4_tail"], np.float32)
                mean = old_tail.copy()
                std = np.full_like(mean, INITIAL_STD)
                rng = np.random.default_rng(SEED)
                candidates, iterations = [], []
                for iteration in range(ITERATIONS):
                    bank = meter.call(
                        "candidate_generate",
                        SAMPLES,
                        lambda: common.make_bank(
                            rng, mean, std, seed_tails, iteration
                        ),
                    )
                    variants = meter.call(
                        "variant_expand",
                        SAMPLES * VARIANTS,
                        lambda: np.stack(
                            [common.action_variants(candidate) for candidate in bank]
                        ),
                    )
                    require(
                        variants.shape == (SAMPLES, VARIANTS, HORIZON, 5, 2),
                        "variant bank",
                    )
                    flat = tensor(
                        variants.reshape(SAMPLES * VARIANTS, HORIZON, 5, 2)
                    )
                    batch = SAMPLES * VARIANTS
                    z = states.repeat(batch, 1, 1)
                    past = history.repeat(batch, 1, 1)
                    for packet in range(HORIZON):
                        context = torch.cat(
                            (
                                past,
                                (
                                    (flat[:, packet] - mean_action) / std_action
                                ).reshape(batch, 1, 10),
                            ),
                            dim=1,
                        )
                        encoded = meter.call(
                            "action_encoder",
                            batch,
                            lambda context=context: model.action_encoder(context),
                        )
                        predicted = meter.call(
                            "predict",
                            batch,
                            lambda z=z, encoded=encoded: model.predict(z, encoded),
                        )
                        require(
                            predicted.shape == z.shape
                            and bool(torch.isfinite(predicted).all()),
                            "model output",
                        )
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    variant_scores = meter.call(
                        "metric",
                        SAMPLES * VARIANTS,
                        lambda: (
                            (z[:, -1] - goal)
                            .square()
                            .mean(dim=1)
                            .reshape(SAMPLES, VARIANTS)
                            .cpu()
                            .numpy()
                        ),
                    )
                    robust = meter.call(
                        "robust_reduce",
                        SAMPLES,
                        lambda: common.robust_scores(variant_scores),
                    )
                    for candidate in range(SAMPLES):
                        candidates.append(
                            {
                                "iteration": iteration,
                                "candidate": candidate,
                                "nominal_q": float(variant_scores[candidate, 0]),
                                "variant_q": variant_scores[candidate].tolist(),
                                "robust_score": float(robust[candidate]),
                                "actions": bank[candidate].tolist(),
                            }
                        )
                    mean, std, elite_indices = meter.call(
                        "elite_update",
                        ELITES,
                        lambda: common.update(bank, robust),
                    )
                    iterations.append(
                        {
                            "iteration": iteration,
                            "elite_indices": elite_indices,
                            "mean": mean.tolist(),
                            "std": std.tolist(),
                        }
                    )
                selected = meter.call(
                    "selection",
                    ITERATIONS * SAMPLES,
                    lambda: min(
                        candidates,
                        key=lambda item: (
                            item["robust_score"],
                            item["iteration"],
                            item["candidate"],
                        ),
                    ),
                )
                ledger = meter.settle()
                filename = f"{role}.candidates.jsonl"
                candidate_sha = write_candidates(output / filename, candidates)
                report["rows"].append(
                    {
                        "role": role,
                        "candidate_index": role_row["candidate_index"],
                        "candidate_label": role_row["candidate_label"],
                        "status": "replanned",
                        "cp5_pixel_sha256": role_row["cp5_pixel_sha256"],
                        "actual_prefix": role_row["actual_prefix"],
                        "actual_prefix_sha256": role_row["actual_prefix_sha256"],
                        "old_v4_tail": role_row["old_v4_tail"],
                        "selected": selected,
                        "iterations": iterations,
                        "candidate_file": filename,
                        "candidate_file_sha256": candidate_sha,
                        "axis_ledger": ledger,
                    }
                )
                expected_hook_calls += AXIS_CAPS["predict"]["calls"]
                journal.emit(
                    "arm_complete",
                    role=role,
                    selected_iteration=selected["iteration"],
                    selected_candidate=selected["candidate"],
                    selected_robust_score=selected["robust_score"],
                    candidate_file_sha256=candidate_sha,
                    axis_ledger=ledger,
                )
        torch.cuda.synchronize(device)
        report["model_after"] = blind.host.signal.model_hash(model)
        report["module_hooks"] = hooks
        require(
            hooks
            == {
                "action_encoder": expected_hook_calls,
                "predict": expected_hook_calls,
            }
            and report["model_before"] == report["model_after"],
            "module hooks and frozen model",
        )
        require(
            all(sha256(Path(path)) == expected for path, expected in value["bindings"].items()),
            "post-run source closure",
        )
        report["status"] = "model_only_complete_not_admitted"
        journal.emit("run_complete", arms=len(report["rows"]), P1_admission=False)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        try:
            journal.emit("run_failed", error=report["error"], P1_admission=False)
        except BaseException:
            pass
    finally:
        for handle in handles:
            handle.remove()
        report["journal_tail"] = journal.previous
        report["journal_events"] = journal.index
        report["wall_seconds"] = time.monotonic() - started
        journal.close()
        report_sha = write_new(output / "report.json", report)
    require(report["status"] == "model_only_complete_not_admitted", "run failed; output retained")
    return report_sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-directory", required=True)
    parser.add_argument("--physical-report-sha256", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive outputs")
    value = protocol(Path(args.physical_directory), args.physical_report_sha256)
    prepared_sha = write_new(prepared, value)
    print(
        json.dumps(
            {"prepared": str(prepared), "sha256": prepared_sha, "CUDA_started": False},
            sort_keys=True,
        ),
        flush=True,
    )
    if args.run:
        with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            report_sha = run(prepared, prepared_sha, output)
        print(
            json.dumps(
                {"output": str(output), "report_sha256": report_sha},
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
