# E3 cell10 single-host gate: implementation boundary

This is a bounded source trace, not a proof of an exhaustive set of callers,
attempt history, a runtime fence, or scientific admission. No production
claim, GPU, model, or physical episode was started for this work.

## Known call paths and claims

| Path or family | Relevant claim behavior |
| --- | --- |
| `agent_e3_chunked_identity_host_20261029_v1.run` | Direct shared lifecycle; exclusive caller-selected output, no cell-wide claim. |
| `agent_e3_identity_host_20261029_v1.run` | Direct shared lifecycle; exclusive caller-selected output, no cell-wide claim. |
| `agent_e3_continuous_host_launcher_20261029_v1.run` | Direct shared lifecycle; exclusive caller-selected output, no cell-wide claim. |
| `agent_e3_integrated_host_saved_20261029_v1.run` | Direct shared lifecycle and host performer; no cell-wide claim. |
| `agent_e3_startup_closure_20261029_v1.run` | CPU precheck then calls chunked host; no cell-wide claim. |
| `e3_cell02_exclusive_candidate_codex_20261030_v1.run` | Its own fixed cell02 marker and output; neither coordinates with cell10 marker nor prevents direct host calls. |
| `e3_cell03_exclusive_host_codex_20261030_v1.run` | Its own fixed cell03 marker and output; existing marker/partial must remain untouched. |
| `agent_e3_encoder_cuda_preflight_20261029_v{1,2,3}.run` | Separate CUDA encoder preflights, not feedback episodes; no unified cell10 marker, still compete for resources. |
| `agent_e3_gd_host_cuda_20261029_v1.run_cuda` | Separate control5 numerical producer, not closed-loop feedback; no unified cell10 marker. |
| `agent_e3_authorized_episode_20261029_v{1,2}.run`, `agent_e3_reset_identity_20261029_v1.run`, `agent_e3_live_episode_join_20261029_v1.execute` | Lower-level callable lifecycle/episode components can be imported directly; no cell10 marker. |
| `e3_cell10_durable_claim_codex_20261030_v1.claim` | Fixed cooperative cell10 marker, but not called by the above routes. |

The results-root filename scan found no cell10 feedback host or cell10 claim
matching its limited patterns at inspection time. It cannot exclude arbitrary
output names, other hosts, in-flight work, direct imports, or earlier attempts.
The cell10 control5 report and its receipt are upstream inputs, not proof of a
cell10 feedback episode.

## New executable boundary

`e3_cell10_single_host_gate_codex_20261030_v1` exposes a `run` command that
always refuses before creating a claim, and a `fixture` command confined to a
canonical `/tmp` subdirectory. Fixture route labels share the prior fixed
claim primitive, then start only a hard-coded inert interpreter. They do not
dispatch real entrypoints. A failed child retains the marker. The five route
labels do not mean those real entrypoints participate in the gate.

Before production, an authorized operator must adjudicate all history and
establish an independently verified, continuously enforced administrative
fence covering old CLIs, direct imports and all relevant hosts; then review
filesystem O_EXCL/fsync behavior, independently seal the final launcher and
actual runtime modules, coordinate GPU and complete cost/resource accounting.
Only then can a reviewed real-child handoff be integrated. No boolean flag or
local process/file scan in this prototype supplies that external authority.
Prior claims, results, fixed scientific gates and the 16-cell denominator
remain unchanged.

## Offline conflict tests

Run from `le-wm-main`:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_e3_cell10_single_host_gate_codex_20261030_v1.py \
  tests/test_e3_cell10_durable_claim_codex_20261030_v1.py \
  tests/test_e3_cell10_feedback_prepare_codex_20261030_v1.py
```

The process-race test uses five independent processes and one temporary
marker. It establishes neither network-filesystem behavior nor non-cooperative
exclusion. Production is intentionally unavailable; there is no safe
production run command yet.
