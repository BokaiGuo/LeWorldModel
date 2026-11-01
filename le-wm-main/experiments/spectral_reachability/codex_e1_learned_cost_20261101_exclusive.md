# E1 learned new-run call accounting

The exclusive journal records `model_call` attempts and returns for the owned
encoder, projector, action encoder, low predictor/projection, and the three
high modules. Trace-based attempts are recorded before the original trace's
guards and buffer persistence, so an unmatched attempt does not prove model
dispatch. Forward-hook attempts are recorded before module forward; an
unmatched attempt can indicate failure or interruption. A return counts only
after the corresponding instrumentation completes. Count attempts and returns
separately by `details.model`; never infer a zero from a missing record.
Pixel sublayers and aggregate cost/search operations are not extra model calls.

`cost_scope` explicitly marks shared costs `shared_unallocated`, with unknown
amount and allocation. The journal does not measure time, FLOPs, memory,
money, upstream costs, or historical runs. Counts from nested modules are not
additive compute costs. The original runner still controls its own report,
manifest verification, path restrictions and CUDA authorization.

CPU-only tests, from `le-wm-main`:

```sh
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_codex_e1_learned_durable_runner_20261101_exclusive.py \
  tests/test_codex_e1_proposal_durable_events_20261101.py
```

For a separately authorized future CUDA run, first replace every placeholder.
The manifest must be an existing verified runner manifest; the output must be
a new absolute path under the runner's `results/` directory with the runner's
required `agent_hilewm_rgb_high_low_20261028_v1_` prefix. The journal must be
a separate nonexistent file in an existing directory, not inside the output
directory. This command starts GPU work; it was not executed for this change:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. .venv/bin/python -m \
  experiments.spectral_reachability.codex_e1_learned_durable_runner_20261101_exclusive \
  --manifest /ABS/PATH/manifest.json \
  --manifest-sha256 REPLACE_WITH_VERIFIED_SHA256 \
  --output /ABS/PATH/TO/le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_NEW \
  --journal /ABS/PATH/TO/EXISTING_DIRECTORY/NEW-events.jsonl \
  --run-id REPLACE_WITH_UNIQUE_RUN_ID --arm learned --main-authorized
```

Read and validate the journal with `read_events(path)` from
`codex_e1_proposal_durable_events_20261101`; an interrupted or malformed
tail must not be silently treated as a complete result.
