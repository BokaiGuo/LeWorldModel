"""Execute the unified covariance--predictor--memory--input--output protocol.

The repository keeps historical theory/temporal paths for reproducibility, but
this runner treats geometry, lifted transfer, contact control, and the claim
matrix as one study.  It creates an immutable manifest before execution,
never overwrites an existing protocol directory, runs every stage with an
explicit command list (no shell), and records stage logs plus a final report.

The output is evidence bookkeeping and reproducible execution state.  It does
not promote a claim merely because a stage completed; the generated six-goal
matrix remains the scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = "covariance_predictor_memory_input_output.v2"
DEFAULT_SOURCE = Path(
    "results/protocol_corrected_temporal_export_20261012/"
    "jepa_pusht_windows4_metrics_enriched_export.json"
)
DEFAULT_SEEDS = tuple(range(20261012, 20261022))
DEFAULT_ADAPTER_SEEDS = (9173, 9181)


@dataclass(frozen=True)
class Stage:
    name: str
    purpose: str
    commands: tuple[tuple[str, ...], ...]
    expected: tuple[str, ...]


def _rel(path: Path, repo: Path) -> str:
    return str(path.resolve().relative_to(repo.resolve()))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _command(python: str, script: str, *args: object) -> tuple[str, ...]:
    return (python, script, *(str(arg) for arg in args))


def build_stages(
    *,
    repo: Path,
    output_root: Path,
    source: Path,
    split_seeds: Iterable[int],
    adapter_seeds: Iterable[int],
) -> tuple[Stage, ...]:
    """Build the complete stage graph with fixed inputs and output paths."""

    python = sys.executable
    geometry_json = output_root / "geometry" / "gauge_nonnormal_audit.json"
    controlled_json = output_root / "controlled_alignment" / "diagnostic.json"
    legacy_dir = output_root / "transfer" / "legacy_memory"
    action_history_dir = output_root / "transfer" / "action_history_memory"
    summary_action_dir = output_root / "transfer" / "summary_action_memory"
    fourier_dir = output_root / "transfer" / "fourier_state_memory"
    adaptive_dir = output_root / "transfer" / "adaptive_event_memory"
    legacy_paths = tuple(
        legacy_dir / f"pusht_splitseed{seed}.json" for seed in split_seeds
    )
    history_paths = tuple(
        action_history_dir / f"pusht_splitseed{seed}.json" for seed in split_seeds
    )
    summary_action_paths = tuple(
        summary_action_dir / f"pusht_splitseed{seed}.json" for seed in split_seeds
    )
    fourier_paths = tuple(
        fourier_dir / f"pusht_splitseed{seed}.json" for seed in split_seeds
    )
    adaptive_paths = tuple(
        adaptive_dir / f"pusht_splitseed{seed}.json" for seed in split_seeds
    )
    legacy_summary = legacy_dir / "summary.json"
    history_summary = action_history_dir / "summary.json"
    summary_action_summary = summary_action_dir / "summary.json"
    fourier_summary = fourier_dir / "summary.json"
    adaptive_summary = adaptive_dir / "summary.json"

    transfer_commands: list[tuple[str, ...]] = []
    for seed, path in zip(split_seeds, legacy_paths):
        transfer_commands.append(
            _command(
                python,
                "experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py",
                "--source", _rel(source, repo), "--output", _rel(path, repo),
                "--split-seed", seed, "--legacy-memory-input",
            )
        )
    for seed, path in zip(split_seeds, history_paths):
        transfer_commands.append(
            _command(
                python,
                "experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py",
                "--source", _rel(source, repo), "--output", _rel(path, repo),
                "--split-seed", seed,
            )
        )
    for seed, path in zip(split_seeds, summary_action_paths):
        transfer_commands.append(
            _command(
                python,
                "experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py",
                "--source", _rel(source, repo), "--output", _rel(path, repo),
                "--split-seed", seed, "--memory-action-mode", "summary",
            )
        )
    for seed, path in zip(split_seeds, fourier_paths):
        transfer_commands.append(
            _command(
                python,
                "experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py",
                "--source", _rel(source, repo), "--output", _rel(path, repo),
                "--split-seed", seed, "--memory-action-mode", "summary",
                "--memory-state-mode", "fourier_residual",
            )
        )
    for seed, path in zip(split_seeds, adaptive_paths):
        transfer_commands.append(
            _command(
                python,
                "experiments/empirical_spectral_audit/run_adaptive_time_frequency_memory_audit.py",
                "--source", _rel(source, repo), "--output", _rel(path, repo),
                "--split-seed", seed,
            )
        )
    transfer_commands.extend(
        (
            _command(
                python,
                "experiments/empirical_spectral_audit/summarize_protocol_corrected_temporal_audit.py",
                "--input", *(_rel(path, repo) for path in legacy_paths),
                "--output", _rel(legacy_summary, repo),
            ),
            _command(
                python,
                "experiments/empirical_spectral_audit/summarize_protocol_corrected_temporal_audit.py",
                "--input", *(_rel(path, repo) for path in history_paths),
                "--output", _rel(history_summary, repo),
            ),
            _command(
                python,
                "experiments/empirical_spectral_audit/summarize_protocol_corrected_temporal_audit.py",
                "--input", *(_rel(path, repo) for path in summary_action_paths),
                "--output", _rel(summary_action_summary, repo),
            ),
            _command(
                python,
                "experiments/empirical_spectral_audit/summarize_protocol_corrected_temporal_audit.py",
                "--input", *(_rel(path, repo) for path in fourier_paths),
                "--output", _rel(fourier_summary, repo),
            ),
            _command(
                python,
                "experiments/empirical_spectral_audit/summarize_adaptive_time_frequency_memory_audit.py",
                "--input", *(_rel(path, repo) for path in adaptive_paths),
                "--output", _rel(adaptive_summary, repo),
            ),
        )
    )

    adapter_commands = tuple(
        _command(
            python,
            "experiments/run_pusht_lifted_contact_adapter.py",
            "--output",
            _rel(output_root / "control" / f"pose_adapter_seed{seed}", repo),
            "--seed", seed,
            "--selection-mode", "pose_regressor",
            "--sequence-steps", 4,
            "--contact-recovery-fallback",
            "--pose-xy-weight", 1.0,
            "--pose-angle-weight", 4.0,
        )
        for seed in adapter_seeds
    )

    matrix_output = output_root / "claim_matrix"
    matrix_command = _command(
        python,
        "experiments/build_six_goal_evidence_matrix.py",
        "--output", _rel(matrix_output, repo),
        "--temporal-summary", _rel(legacy_summary, repo),
        "--temporal-action-history-summary", _rel(history_summary, repo),
        "--temporal-fourier-summary", _rel(fourier_summary, repo),
    )

    return (
        Stage(
            "geometry",
            "Run gauge-aware, Schur, non-normal and transient diagnostics.",
            (_command(
                python,
                "experiments/empirical_spectral_audit/run_gauge_nonnormal_audit.py",
                "--output", _rel(geometry_json, repo),
            ),),
            (_rel(geometry_json, repo),),
        ),
        Stage(
            "controlled_alignment",
            "Run the fixed-spectrum covariance--predictor controlled benchmark.",
            (_command(
                python,
                "experiments/empirical_spectral_audit/run_covariance_predictor_diagnostic_benchmark.py",
                "--output", _rel(controlled_json, repo),
                "--repeats", 128,
            ),),
            (_rel(controlled_json, repo),),
        ),
        Stage(
            "learned_transfer",
            "Run legacy and action-history memory inputs on identical seeded splits.",
            tuple(transfer_commands),
            tuple(_rel(path, repo) for path in (*legacy_paths, *history_paths, *summary_action_paths, *fourier_paths, *adaptive_paths, legacy_summary, history_summary, summary_action_summary, fourier_summary, adaptive_summary)),
        ),
        Stage(
            "contact_control",
            "Run root-disjoint successor-free lifted pose adapters with contact recovery.",
            adapter_commands,
            tuple(
                _rel(output_root / "control" / f"pose_adapter_seed{seed}" / "lifted_contact_adapter.json", repo)
                for seed in adapter_seeds
            ),
        ),
        Stage(
            "claim_matrix",
            "Join stage artifacts into the six-goal evidence matrix without promotion by completion.",
            (matrix_command,),
            (_rel(matrix_output / "six_goal_evidence_matrix.json", repo), _rel(matrix_output / "six_goal_evidence_matrix.md", repo)),
        ),
    )


def build_manifest(
    *,
    repo: Path,
    output_root: Path,
    source: Path,
    split_seeds: Iterable[int] = DEFAULT_SEEDS,
    adapter_seeds: Iterable[int] = DEFAULT_ADAPTER_SEEDS,
) -> dict[str, Any]:
    split_seeds = tuple(int(seed) for seed in split_seeds)
    adapter_seeds = tuple(int(seed) for seed in adapter_seeds)
    stages = build_stages(
        repo=repo,
        output_root=output_root,
        source=source,
        split_seeds=split_seeds,
        adapter_seeds=adapter_seeds,
    )
    return {
        "artifact": "unified_research_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "study_scope": "covariance_predictor_memory_input_output",
        "claim_boundary": (
            "Completion of a command is not scientific support. Geometry, transfer, "
            "contact control and visual planning remain separate evidence gates."
        ),
        "repository": str(repo.resolve()),
        "output_root": str(output_root.resolve()),
        "source_export": str(source.resolve()),
        "split_seeds": list(split_seeds),
        "adapter_seeds": list(adapter_seeds),
        "stages": [
            {
                "name": stage.name,
                "purpose": stage.purpose,
                "commands": [list(command) for command in stage.commands],
                "expected": list(stage.expected),
            }
            for stage in stages
        ],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    """Write an immutable manifest and return its content hash."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen unified manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    digest = _sha256_bytes(encoded)
    payload = dict(manifest)
    payload["manifest_sha256"] = digest
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest


def _run_stage(stage: dict[str, Any], *, repo: Path, output_root: Path) -> dict[str, Any]:
    name = str(stage["name"])
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = stage["commands"]
    command_logs: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        print(f"[{name}] purpose: {stage['purpose']}")
        print(f"[{name}] command {index + 1}/{len(commands)}: {' '.join(command)}")
        log_path = log_dir / f"{name}_{index:03d}.log"
        with log_path.open("w") as handle:
            completed = subprocess.run(
                command,
                cwd=repo,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=dict(os.environ),
            )
        command_logs.append({
            "command": command,
            "returncode": int(completed.returncode),
            "log": str(log_path),
        })
        if completed.returncode != 0:
            raise RuntimeError(f"stage {name} command {index} failed; see {log_path}")
    missing = [path for path in stage["expected"] if not (repo / path).exists()]
    if missing:
        raise FileNotFoundError(f"stage {name} completed but expected artifacts are missing: {missing}")
    return {"name": name, "status": "completed", "commands": command_logs, "expected": stage["expected"]}


def execute(manifest_path: Path, *, repo: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    output_root = Path(manifest["output_root"])
    stages = manifest["stages"]
    results = []
    for stage in stages:
        results.append(_run_stage(stage, repo=repo, output_root=output_root))
    report = {
        "artifact": "unified_research_protocol_execution",
        "protocol_version": manifest["protocol_version"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "status": "completed",
        "stages": results,
        "claim_boundary": manifest["claim_boundary"],
    }
    report_path = output_root / "execution_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--adapter-seeds", type=int, nargs="+", default=list(DEFAULT_ADAPTER_SEEDS))
    parser.add_argument("--run", action="store_true", help="execute the frozen manifest after creating it")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to use non-empty unified protocol directory: {output_root}")
    manifest = build_manifest(
        repo=repo,
        output_root=output_root,
        source=(repo / args.source).resolve() if not args.source.is_absolute() else args.source.resolve(),
        split_seeds=args.split_seeds,
        adapter_seeds=args.adapter_seeds,
    )
    manifest_path = output_root / "manifest.json"
    digest = write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": digest}, indent=2))
    if args.run:
        print(json.dumps(execute(manifest_path, repo=repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
