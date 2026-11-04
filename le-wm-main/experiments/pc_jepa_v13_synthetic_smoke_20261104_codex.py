"""Isolated JEPA feedback worker smoke on deterministic synthetic inputs."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as v13
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_jepa_v13_synthetic_isolation_smoke_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V13_SYNTHETIC_ISOLATION_SMOKE_PROTOCOL_20261104_CODEX.md"
METADATA = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex/input/model_metadata.json"
LAUNCH = ROOT / "experiments/pc_jepa_v13_synthetic_smoke_launch_20261104_codex.py"
WORKER = ROOT / "experiments/pc_jepa_v13_contact_feedback_worker_20261104_codex.py"
ARMS = ("response", "cem")


def write(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    deps = seal.capture([Path(__file__), PROTOCOL, METADATA, LAUNCH, WORKER,
        Path(v13.__file__), ROOT / "experiments/spectral_reachability/pc_recursive_response_v12.py",
        ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"])
    OUT.mkdir()
    rng = np.random.default_rng(2029132200)
    rows = []
    for step in range(3):
        rgb = rng.integers(0, 256, size=(3, 224, 224, 3), dtype=np.uint8)
        goal = rng.integers(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
        prop = rng.normal(0, .1, size=(3, 4)).astype(np.float32)
        packets = rng.normal(0, .1, size=(2, 10)).astype(np.float32)
        carry = (np.empty((0, 2), np.float32) if step == 0
                 else rng.uniform(-.4, .4, size=(15-5*step, 2)).astype(np.float32))
        for arm in ARMS:
            base = OUT / f"{arm}_{step}"
            base.mkdir()
            (base / "input").mkdir()
            (base / "output").mkdir()
            with (base / "input/context.npz").open("xb") as f:
                np.savez_compressed(f, rgb_history=rgb, goal_rgb=goal, normalized_proprio=prop,
                    normalized_history_packets=packets, action_mean=np.zeros(2, np.float32),
                    action_std=np.ones(2, np.float32), carry=carry)
            write(base / "input/config.json", {"arm": arm, "step": step})
            write(base / "input/model_metadata.json", json.loads(METADATA.read_text()))
            rows.append({"arm": arm, "step": step, "input_sha256": seal.digest(base / "input/context.npz")})
    seal.verify(deps)
    write(OUT / "manifest.json", {"dependencies": deps, "cases": rows, "seed": 2029132200,
        "scope": "synthetic inputs only; no physical observations or actions"})


def run():
    manifest = json.loads((OUT / "manifest.json").read_text())
    seal.verify(manifest["dependencies"])
    if (OUT / "runs.json").exists():
        raise FileExistsError("runs exist")
    rows = []
    for case in manifest["cases"]:
        arm, step = case["arm"], case["step"]
        base = OUT / f"{arm}_{step}"
        if seal.digest(base / "input/context.npz") != case["input_sha256"]:
            raise ValueError("input changed")
        started = time.monotonic()
        try:
            proc = subprocess.run([str(ROOT / ".venv/bin/python"), "-m",
                "experiments.pc_jepa_v13_synthetic_smoke_launch_20261104_codex", str(base)],
                cwd=ROOT, capture_output=True, text=True, timeout=600)
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc, stdout, stderr = None, str(exc.stdout or ""), str(exc.stderr or "")
        output = base / "output/decision.json"
        rows.append({"arm": arm, "step": step, "returncode": rc,
            "stdout": stdout[-4000:], "stderr": stderr[-4000:],
            "seconds": time.monotonic()-started,
            "decision_sha256": seal.digest(output) if output.exists() else None})
        print(f"arm={arm} step={step} returncode={rc} output={output.exists()}", flush=True)
    seal.verify(manifest["dependencies"])
    write(OUT / "runs.json", {"manifest_sha256": seal.digest(OUT / "manifest.json"), "rows": rows})


def audit():
    if (OUT / "audit.json").exists():
        raise FileExistsError("audit exists")
    manifest, runs = (json.loads((OUT / name).read_text()) for name in ("manifest.json", "runs.json"))
    seal.verify(manifest["dependencies"])
    if runs["manifest_sha256"] != seal.digest(OUT / "manifest.json"):
        raise ValueError("manifest mismatch")
    rows = []
    for record, case in zip(runs["rows"], manifest["cases"], strict=True):
        if (record["arm"], record["step"]) != (case["arm"], case["step"]):
            raise ValueError("case mismatch")
        arm, step = record["arm"], record["step"]
        base = OUT / f"{arm}_{step}"
        if record["returncode"] != 0:
            rows.append({"arm": arm, "step": step, "status": "worker_failed",
                         "returncode": record["returncode"]})
            continue
        output = base / "output/decision.json"
        if seal.digest(output) != record["decision_sha256"]:
            raise ValueError("decision hash mismatch")
        data = json.loads(output.read_text())
        with np.load(base / "input/context.npz", allow_pickle=False) as f:
            carry = None if step == 0 else f["carry"].copy()
        horizon = 15-5*step
        config = v13.Config(action_dim=2, segment_length=5, segments=horizon//5,
            nominals=tuple(tuple((float(a), float(b)) for _ in range(5))
                           for a, b in ((.25, 0.), (.25, .5))),
            token_cap=1920, seed=2027110401+step*10000)
        bank = v13.proposal_bank(config, carry)
        np.testing.assert_array_equal(np.asarray(data["evaluations"][0]["actions"], np.float32), bank)
        if data["input_sha256"] != case["input_sha256"] or data["arm"] != arm or data["step"] != step:
            raise ValueError("decision input mismatch")
        token, full, matched = 0, [], []
        chain = np.asarray(data["decision"]["selected"]["chain"], np.float32)
        if chain.shape != (horizon, 2) or np.max(np.abs(chain)) > 1:
            raise ValueError("invalid selected chain")
        for batch in data["evaluations"]:
            actions, scores = np.asarray(batch["actions"], np.float32), np.asarray(batch["scores"])
            if actions.ndim != 3 or actions.shape[1] % 5 or np.max(np.abs(actions)) > 1 or not np.isfinite(scores).all():
                raise ValueError("invalid evaluation")
            token += len(actions)*(actions.shape[1]//5)
            if actions.shape[1] == horizon:
                full.extend(scores.tolist())
                matched.extend(float(s) for a, s in zip(actions, scores, strict=True) if np.array_equal(a, chain))
        score = data["decision"]["selected"]["terminal_score"]
        if token != data["meter"]["prediction_token_samples"] or token > 1920 or not matched:
            raise ValueError("selection or accounting mismatch")
        if not any(abs(x-score) < 1e-6 for x in matched) or score > min(full)+1e-5:
            raise ValueError("selected chain is not best scored incumbent")
        pair = data["evaluations"][-1]
        if np.asarray(pair["actions"]).shape != (2, horizon, 2):
            raise ValueError("invalid final pair")
        rows.append({"arm": arm, "step": step, "status": "passed", "tokens": token,
                     "score": score, "bank_samples": len(bank)})
    for step in range(3):
        with np.load(OUT / f"response_{step}/input/context.npz", allow_pickle=False) as first, \
                np.load(OUT / f"cem_{step}/input/context.npz", allow_pickle=False) as second:
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
    seal.verify(manifest["dependencies"])
    write(OUT / "audit.json", {"runs_sha256": seal.digest(OUT / "runs.json"), "rows": rows,
        "scope": "synthetic interface check, not physical control or scientific evidence"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "run", "audit"):
        raise SystemExit("choose prepare, run or audit")
    {"prepare": prepare, "run": run, "audit": audit}[sys.argv[1]]()
