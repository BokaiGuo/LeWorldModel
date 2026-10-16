# Storage cleanup audit — 2026-10-16

This cleanup removes only rebuildable or superseded local material. It does not
remove model checkpoints, source data, formal result JSON, negative-result
reports, or the retained `recovery` / `downstream_final2` evidence surfaces.

## Retained canonical evidence

- `le-wm-main/results/full_spectral_transfer_cuda_recovery_20261014/`
- `le-wm-main/results/full_spectral_transfer_cuda_downstream_final2_20261014/`
- all checkpoint and dataset roots under `le-wm-main/data/`
- all source, tests, documentation, and existing result summaries

## Removed material

- `results/full_spectral_transfer_cuda_review_20261014/chunks/`: interrupted
  resumable-export chunks; the run status, event log, progress log, and theory
  outputs remain, while the completed recovery export is retained.
- `results/full_spectral_transfer_cuda_20261014/chunks/`: stale interrupted
  phase-1 export chunks from the superseded run; its status, logs, and theory
  summaries remain for provenance.
- `results/full_spectral_transfer_cuda_downstream_20261014/chunks/`: temporary
  chunks for a completed export whose consolidated latent and downstream
  evidence are retained elsewhere.
- `results/full_spectral_transfer_cuda_b256_20261014/chunks/` and
  `results/full_spectral_transfer_cuda_recovery_20261014/chunks/`: completed
  export intermediates; their consolidated `latents.npz`, manifests, metrics,
  and status/event records remain.
- `results/tworoom_full_spectral_cuda_20261015/full_latents/chunks/`: 83
  completed latent-export chunks superseded by the consolidated
  `full_latents/latents.npz`; the run status is completed and the planning
  alignment artifact remains.
- `results/tworoom_full_spectral_cuda_20261015/chunks/`, the GPU batch/recovery
  probe chunk directories, and `results/pusht/lewm/chunks/`: small completed
  or probe-only intermediates with their summaries retained.
- `results/full_spectral_transfer_cuda_downstream_final2_20261014/planning_chunks/`:
  completed planning intermediates superseded by `planning_alignment_raw.npz`
  and its summary JSON.
- `results/full_spectral_transfer_cuda_downstream_final_20261014/`: superseded
  duplicate result directory; its latent file was a hard link to the retained
  canonical export.
- `.cache/dinov2/`, `.cache/torch/`, Python `__pycache__` directories, and
  `.pytest_cache/`: rebuildable runtime caches.

The cleanup intentionally does not remove archived negative-result directories
or the dataset archives, because their scientific/provenance role has not been
verified as redundant.

## Follow-up cleanup — 2026-10-16

After the first inventory, a second pass removed only items verified to be
rebuildable or probe-only:

- all `__pycache__/` and `.pytest_cache/` directories below `le-wm-main/`;
- completed-run `planning_chunks/` under the spectral and TwoRoom runs, whose
  consolidated planning/latent outputs remain;
- `results/gpu_batch_probe_20261014/`,
  `results/gpu_batch_probe_256_20261014/`,
  `results/gpu_batch_probe_256_rich_20261014/`, and
  `results/gpu_recovery_probe_20261014/`; these were unreferenced probe-only
  outputs rather than formal evidence.

The follow-up freed approximately 4.8 GB according to the filesystem free-space
measurement. No source, checkpoint, dataset, formal JSON/report, or retained
negative-result directory was removed. The remaining large result directories
are intentionally retained until their scientific redundancy is verified.

## Current pass — 2026-10-16

The following verified disposable material was removed:

- all `__pycache__/` directories below `le-wm-main/` (about 110 MiB; Python
  recreates them automatically);
- `results/theory_operator_alignment_iclr_oral_20260829/official_iclr/iclr2026.zip`,
  after confirming the extracted `official_iclr/iclr2026/` tree is present;
- the archive's `official_iclr/__MACOSX/` metadata directory;
- the empty `results/tmp/` directory.

No dataset archive, checkpoint, source file, formal result, or negative-result
artifact was removed in this pass.

## Current pass 2 — 2026-10-16

The following disposable runtime material was removed:

- all newly regenerated `__pycache__/` and `.pytest_cache/` directories below
  `le-wm-main/` (about 118 MiB; Python recreates them automatically);
- stale `results/pusht/lewm/status_monitor.pid` and
  `results/pusht/lewm/full_run.pid`, after verifying that neither PID referred
  to a running process.

The CIFAR10, CIFAR100, and STL10 archives were inspected but retained: they
are currently the only local copies of those source datasets. No source code,
checkpoint, dataset, formal result, or negative-result artifact was removed.

## Current pass 3 — 2026-10-16

The following verified disposable material was removed:

- `results/full_spectral_transfer_cuda_downstream_20261014/latents.npz` and
  `results/full_spectral_transfer_cuda_downstream_postpatch_20261014/latents.npz`:
  both had the same SHA-256 as the retained recovery/final2 canonical latent
  export; their run metadata and downstream summaries remain.
- 13 completed `enriched_jepa_*/run.log` files larger than 1 MiB, while each
  corresponding run directory retains its `metrics.json` and other result
  artifacts.
- all regenerated Python `__pycache__/` and `.pytest_cache/` directories under
  `le-wm-main/`.

No source, checkpoint, dataset, formal JSON/report, or negative-result
directory was removed. The duplicate latent payloads, logs, and caches are
rebuildable or retained elsewhere in canonical form.

## Current pass 4 — 2026-10-16

After checking that no experiment process was running, the following disposable
material was removed:

- 546 rebuildable `__pycache__/` directories and `.pytest_cache/` directories
  below the repository, including the virtual environment; Python recreates
  them automatically.
- 45 raw training logs under
  `results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/`
  and `wall_temporal_gate/`; the corresponding metrics JSON files remain.
- 24 raw logs under
  `results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/logs/`;
  every log had a same-stem metrics JSON under `metrics/`.
- `jepa_wm_pointmaze_episode_shuffle_metrics_fixed.json`, which was byte-for-byte
  identical to the retained `jepa_wm_pointmaze_episode_shuffle_metrics.json`.
- `calibration/khodakarami2026_operator_spectral_bias.pdf`, which was
  byte-for-byte identical to the retained copy in `section_reference_papers/`.

The cleanup freed approximately 0.8–0.9 GiB by filesystem accounting. Dataset
archives, checkpoints, latent exports, source code, formal result JSON/report
files, and negative-result directories were not removed.

## Current pass 5 — 2026-10-16

After verifying that no experiment process was running and that the two large
latent files had identical SHA-256 values, the duplicate payload was
deduplicated with a hard link:

- `results/full_spectral_transfer_cuda_recovery_20261014/latents.npz` now
  shares its inode with the identical payload in
  `results/full_spectral_transfer_cuda_downstream_final2_20261014/`.
- Both result paths and their independent run metadata remain available; the
  latent bytes are stored only once.
- Newly regenerated `__pycache__/` and `.pytest_cache/` directories were
  removed again as rebuildable runtime caches.

No source, checkpoint, dataset, formal JSON/report, or negative-result
directory was removed in this pass.

## Current pass 6 — 2026-10-16

The following verified disposable material was removed:

- all currently regenerated `__pycache__/` and `.pytest_cache/` directories
  below `le-wm-main/` (548 Python cache directories and one pytest cache;
  these are automatically rebuildable);
- `le-wm-main/data/cifar100/cifar-100-python.tar.gz`, after `tar` reported an
  unexpected end of file and the directory contained no extracted dataset;
  this was an incomplete download rather than usable source data.
- 26 raw experiment logs larger than 1 MiB (about 50.3 MiB total) under the
  negative-diagnosis result roots, after verifying that each containing
  directory retains a structured JSON/JSONL result.

The CIFAR-10 and STL-10 archives were integrity-checked and retained because
their directories contain no extracted copy. No checkpoint, latent export,
source file, formal result, negative-result artifact, or active experiment was
removed.

## Current pass 7 — 2026-10-16

After verifying that no experiment process was running, the following disposable
material was removed:

- all currently regenerated `__pycache__/` directories below `le-wm-main/`;
- `le-wm-main/.pytest_cache/`;
- seven zero-byte files under `le-wm-main/results/` (empty log placeholders).

No source, checkpoint, dataset, latent export, formal result, or negative-result
artifact was removed. The result tree currently has no empty files and no Python
cache directories; both classes are automatically rebuildable.

## Current pass 8 — 2026-10-16

After confirming that no experiment process was running, the following cleanup
was performed:

- removed 549 rebuildable Python cache directories below `le-wm-main/`,
  including `__pycache__`, `.pytest_cache`, `.mypy_cache`, and `.ruff_cache`;
- scanned result files larger than 1 MiB by SHA-256 and replaced 313 exact
  duplicate payloads with hard links, reclaiming approximately 1.66 GiB while
  preserving every result path and file contents.

The deduplication was restricted to `le-wm-main/results/`. Dataset archives,
checkpoints, source code, latent exports, formal result JSON/report files, and
negative-result directories were not deleted. No running experiment was
interrupted.

## Current pass 9 — 2026-10-16

After confirming that no experiment process was running, the following
disposable material was removed:

- all currently regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`,
  and `.ruff_cache/` directories below `le-wm-main/`;
- `le-wm-main/outputs/`, which contained only five ignored Hydra `eval.log`
  files from old runs and no structured result, checkpoint, or source artifact.

The STL-10 and CIFAR-10 archives, the virtual environment, datasets,
checkpoints, latent exports, formal result JSON/report files, and negative-result
directories were retained. The removed cache and output files are rebuildable
and are not needed by the current experiment registry.

## Current pass 10 — 2026-10-16

After the Cube temporal CUDA panel and regression tests completed, all
regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/`
directories below `le-wm-main/` were removed again. No running experiment was
present, and no result, checkpoint, dataset, source file, or report was removed.

## Current pass 11 — 2026-10-16

After confirming that no experiment process was running, this pass removed only
verified disposable material:

- all currently regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`,
  and `.ruff_cache/` directories below `le-wm-main/` (approximately 133 MiB;
  Python recreates them automatically);
- six raw logs under
  `results/negative_cause_diagnosis_20261011/`, each after verifying that its
  same-named run directory retains a structured `rgb_local_effect_adapter.json`:
  `aligned_goal_dynamic_observer_seed9173.log`,
  `aligned_goal_dynamic_terminal_ranker_seed9173.log`,
  `aligned_goal_dynamic_torque_prior_seed9173.log`,
  `centroid_contact_relaxed_dynamic_seed9173.log`,
  `centroid_local_side_repair_cap016_seed9173.log`, and
  `dynamic_readout_fail_closed_seed9173.log`.

## Current pass 12 — 2026-10-16

After confirming that no experiment process was running, this pass removed only
rebuildable Python bytecode:

- all `*.pyc` files below `le-wm-main/` (about 116 MiB), plus the remaining
  root `.pytest_cache/`; Python and pytest recreate these files automatically.

No source, checkpoint, dataset, latent export, formal result, or
negative-result artifact was removed. The remaining large files were checked:
the only project archives are the three required model checkpoints, and the
large latent/data/result payloads are either unique or already hard-linked
duplicates.

## Current pass 12 — 2026-10-16

After confirming that no experiment process was running, this pass removed the
currently regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
`.ruff_cache/` directories below `le-wm-main/` (311 directories, about 48.7
MiB). A content-level scan found no additional disposable large file: the
remaining large files are unique datasets, checkpoints, latent exports, or
formal/negative-result artifacts referenced by the active protocols. They were
retained to preserve cross-environment reruns and provenance.

## Current pass 12 — 2026-10-16

After confirming that no experiment process was running, removed only empty
directories under `results/`. These were failed or probe-only run directories
with no remaining files, including empty smoke-run directories and empty log
directories. No file content was removed in this pass. Empty dataset
directories, the virtual environment, archives, checkpoints, latent exports,
source code, formal result files, and negative-result files were retained.

The unmatched `observer_fail_closed_full_seed9173.log` was retained. Dataset
archives, checkpoints, latent exports, source code, formal result JSON/report
files, and negative-result artifacts were not removed.

## Current pass 12 — 2026-10-16

After the third CUDA feedback panel and regression tests completed, all newly
regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/`
directories below `le-wm-main/` were removed again. No experiment process was
running, and no source, checkpoint, dataset, latent export, formal result, or
negative-result artifact was removed.

## Current pass 13 — 2026-10-16

After the independent TwoRoom CUDA panel and regression tests completed, all
newly regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
`.ruff_cache/` directories below `le-wm-main/` were removed again. No source,
checkpoint, dataset, latent export, formal result, or negative-result artifact
was removed.

## Current pass 15 — 2026-10-16

After the PointMaze representation diagnostic and regression tests completed,
all newly regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
`.ruff_cache/` directories below `le-wm-main/` were removed again. No source,
checkpoint, dataset, latent export, formal result, or negative-result artifact
was removed.

## Current pass 14 — 2026-10-16

After the independent Reacher CUDA panel and regression tests completed, all
newly regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
`.ruff_cache/` directories below `le-wm-main/` were removed again. No source,
checkpoint, dataset, latent export, formal result, or negative-result artifact
was removed.

## Current pass 16 — 2026-10-16

After confirming that no experiment process was running, removed only verified
rebuildable or redundant material:

- 364 Python cache directories (`__pycache__/`, `.pytest_cache/`,
  `.mypy_cache/`, and `.ruff_cache/`) below `le-wm-main/`, about 55 MiB;
- `results/negative_cause_diagnosis_20261011/observer_fail_closed_fullroots_short_seed9173.log`,
  after verifying that the same run retains `rgb_local_effect_adapter.json`,
  `rgb_local_effect_adapter.pt`, and its Markdown report.

No source, checkpoint, dataset, latent export, formal result, or negative-result
artifact was removed. The large result directories and dataset archives remain
because they are still referenced by the experiment registry or are the only
local reproducibility copies.

## Current pass 17 — 2026-10-16

After confirming that no experiment process was running, removed only
rebuildable or empty material:

- all currently present `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` directories below `le-wm-main/` (about 129 MiB before
  filesystem rounding); Python recreates these automatically;
- empty directories below `le-wm-main/results/`.

A fresh duplicate-size/hash audit found no additional large duplicate payload
that could be removed safely. The two equal-size checkpoint files have
different SHA-256 values, while the equal latent exports are already hard
linked. Dataset archives, checkpoints, latent exports, source code, formal
result JSON/report files, and negative-result artifacts were retained.

## Current pass 18 — 2026-10-16

After the full regression suite completed, removed the newly regenerated
`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/` directories
below `le-wm-main/`. No source, checkpoint, dataset, latent export, formal
result, or negative-result artifact was removed.

## Current pass 19 — 2026-10-16

After confirming that no experiment process was running, scanned the current
workspace before deletion. Removed the currently regenerated Python/pytest/lint
cache directories below `le-wm-main/`; these are rebuildable and are not
research artifacts. A hash audit of result files larger than 100 MiB found only
the already hard-linked pair of canonical latent exports and no new duplicate
payload. The remaining large files are retained: dataset archives have no local
extracted copy, and the virtual environment contains the CUDA/PyTorch runtime
needed for GPU experiments. No source, checkpoint, dataset, latent export,
formal result, or negative-result artifact was removed.

## Current pass 20 — 2026-10-16

After confirming that no experiment process was running, removed only
rebuildable or empty material:

- all currently present `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` directories below `le-wm-main/`;
- empty directories below `le-wm-main/results/` and `le-wm-main/data/`.

A fresh size and hash audit found no additional safe GB-scale deletion. The
remaining large files are unique datasets, CUDA/PyTorch runtime files,
checkpoints, latent exports, or formal/negative-result artifacts. No source,
checkpoint, dataset, latent export, formal result, or negative-result artifact
was removed in this pass.

## Current pass 21 — 2026-10-16

After confirming that no LeWorldModel experiment process was running, removed
the remaining rebuildable Python bytecode/test caches and empty directories
below `le-wm-main/`. This pass reclaimed approximately 112.8 MiB.

An exact-hash scan found 313 duplicate result payload groups, but all were
already sharing inodes from the earlier deduplication pass, so no additional
result bytes were removed. The STL-10 and CIFAR-10 archives were retained
because their directories contain no extracted copy; deleting them would
remove the only local source data. Checkpoints, datasets, latent exports,
source code, formal result JSON/report files, and negative-result directories
were not removed.

## Current pass 22 — 2026-10-16

After confirming that no experiment process was running, removed 702 currently
regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/`
directories below `le-wm-main/`, approximately 119 MiB. These directories are
automatically rebuildable.

The result-tree scan found no new duplicate within `le-wm-main/results/`.
The retained Overleaf package did contain 14 byte-identical result payloads;
these were hard-linked across the two retained paths, reclaiming about 59.4
MiB without deleting either copy or changing any result content. No source,
checkpoint, dataset, latent export, formal result, negative-result directory,
or environment package was removed.

## Current pass 23 — 2026-10-16

After confirming that no LeWorldModel experiment process was running and that
the files were not needed by any active result, removed the following
redownloadable or duplicate local payloads:

- `le-wm-main/data/stl10/stl10_binary.tar.gz` (2.64 GB): not extracted locally;
  the experiment code can download it again when needed;
- `le-wm-main/data/cifar10/cifar-10-python.tar.gz` (170 MB): not extracted
  locally; the experiment code can download it again when needed;
- `le-wm-main/.cache/dinov2/dinov2-small/pytorch_model.bin` (about 85 MB):
  duplicate Transformers serialization; `model.safetensors` remains as the
  canonical local weight file.

This pass reclaimed approximately 2.8 GB. Source code, checkpoints, unique
datasets, latent exports, formal result JSON/report files, and negative-result
artifacts were retained.

## Current pass 24 — 2026-10-16

After confirming that no LeWorldModel experiment process was running, removed
only rebuildable runtime material:

- all current `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` contents below `le-wm-main/`;
- empty cache/result/data directories left after earlier cleanup passes.

The DINOv2 cache was inspected but retained: its `model.safetensors` is the
local canonical model weight and is not disposable cache data. No source,
checkpoint, dataset, latent export, formal result, negative-result artifact,
or virtual-environment package was removed. The remaining archives are either
model checkpoints or package-internal assets; no project archive was found
that could be deleted after extraction.

## Current pass 25 — 2026-10-16

After confirming that no LeWorldModel experiment process was running, removed
only generated build/test intermediates:

- `le-wm-main/.pytest_cache/`;
- LaTeX auxiliary files (`.aux`, `.out`, `.fdb_latexmk`, `.fls`) from the
  retained theory result and Overleaf working tree.

These files are rebuildable and contain no source or scientific measurements.
The DINOv2 canonical `model.safetensors`, all datasets, checkpoints, latent
exports, formal result JSON/report files, source code, and negative-result
artifacts were retained. A fresh scan found no project archive or large exact
duplicate that could be safely deleted.

## Current pass 26 — 2026-10-16

After confirming that no LeWorldModel experiment process was running, removed
only rebuildable runtime material:

- 836 regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` directories below `le-wm-main/` and the repository root;
  approximately 133 MiB was reclaimable and Python recreates these caches;
- stale Hugging Face metadata and lock files for the previously removed
  `pytorch_model.bin` DINOv2 serialization; the canonical
  `model.safetensors` remains in place.

Empty result/data directories were also removed. No source, checkpoint,
dataset, latent export, formal result JSON/report, or negative-result artifact
was removed. The remaining 85 MiB DINOv2 cache is the canonical local model
weight required by the GPU experiments; the large result/data/environment
trees were retained because their scientific or reproducibility role is not
redundant.

## Current pass 27 — 2026-10-16

While the CUDA Cube hybrid panel was still running, removed only rebuildable
runtime caches below `le-wm-main/`:

- all regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` directories, including caches inside the virtual environment;
- no active experiment output directory was touched.

The result tree was checked for empty files, stale project archives, temporary
download files, and large exact duplicates. No additional disposable result
payload was identified; the only large exact duplicate pair is the already
hard-linked canonical latent export. Source code, checkpoints, datasets,
formal results, negative-result artifacts, and the active CUDA output were
retained.

## Current pass 28 — 2026-10-16

After confirming that no LeWorldModel experiment process, active PID file, or
zero-byte result file was present, removed only rebuildable runtime material:

- 835 regenerated `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and
  `.ruff_cache/` directories below `le-wm-main/`;
- `le-wm-main/.cache/dinov2/` (about 85 MiB), a redownloadable model cache.

The project tree was rechecked for archives, partial downloads, empty files,
and stale PID files. No disposable project archive or partial download was
found; remaining `.pth.tar` files are model checkpoints and were retained.
No source, checkpoint, dataset, latent export, formal result, or
negative-result artifact was removed.

## Current pass 29 — 2026-10-16

After confirming that no LeWorldModel experiment process was running, removed
only currently regenerated rebuildable caches:

- 833 `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and `.ruff_cache/`
  directories below `le-wm-main/`, including the virtual environment (about
  130 MiB); this includes the repository-level `le-wm-main/.pytest_cache/`.

An exact-content audit found that repeated result payloads are already hard
linked, including repeated adapter checkpoints and canonical exports, so no
result path was deleted. Dataset files, checkpoints, latent exports, source,
formal JSON/report files, and negative-result directories were retained.
