# P1 three-prefix H10: full saved-analysis handoff

This handoff prepares analysis only. During inspection the runner process was
active and its final `run_report.json` was absent. No partial branch results
were read or interpreted. Recheck completion after the producer exits; do not
infer completion or failure solely from this point-in-time observation.

## Fixed inputs and CLI

Working directory: `/home/gbk/code/LeWorldModel/le-wm-main`.

- Run directory: `results/codex_p1_three_prefix_h10_runner_20261030_run_MAIN01`
- Prepare: `results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json`
- Prepare SHA256, independently rehashed during preparation:
  `ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30`
- The run-report SHA256 is **not yet available**. Obtain it from the completed
  producer's final receipt, cross-check the stable `run_report.json` bytes,
  and substitute it in the command below. A hash of an incomplete or
  untrusted report does not establish completeness.

After the producer exits, a final report exists, its pinned SHA is obtained,
and the intended output name is confirmed unused:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_three_prefix_h10_analysis_20261030 \
  --directory /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_three_prefix_h10_runner_20261030_run_MAIN01 \
  --report-sha256 '<COMPLETED_RUN_REPORT_SHA256>' \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json \
  --prepare-sha256 ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_three_prefix_h10_analysis_20261030_FULL_CODEX01.json
```

The literal placeholder is not executable until replaced. Output must be a
new absolute path directly inside `results`, named with the analysis module
prefix; the analyzer writes exclusively and does not overwrite an existing
file. Coordinate ownership of this prospective output name before running.
No separate saved-consumer invocation is necessary: analysis invokes it first.

## All-or-nothing requirements

The source requires a matching externally pinned run-report hash, complete
`recorded_not_consumed` run status, all 24 ordered shard reports with 42
ordered branches each, and a matching run binding and frozen prepare/source
hashes. The saved consumer verifies shard report and journal hashes, action
bytes and ordered transitions, source traces, RGB pixel/file checks, branch
terminal/censor evidence, operation ledger and resource caps. It rechecks
source and metadata for drift. Any missing or invalid shard rejects the whole
analysis; existence of a report or a zero producer exit alone is insufficient.

The output preserves 8 logical groups, 336 group/probe records and 1008 branch
statuses. Native stops and proven out-of-domain censoring remain missing
endpoints, with no padding or imputation; missing zero baselines propagate
to within-role contrasts. Duplicate groups are not independent samples.
Known roots and post-outcome design mean exploratory evidence only:
`P1_admission`, freshness, independent confirmation and equal-budget
certification remain false. Saved checks do not physically replay or establish
unrecorded host/solver-cache authenticity.

## Verification completed without production results

CLI `--help` confirmed all five required options. Thirteen existing
analysis tests passed using synthetic fixtures and no GPU; no final analyzer
was run against the live directory:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_codex_p1_three_prefix_h10_analysis_20261030.py
```
