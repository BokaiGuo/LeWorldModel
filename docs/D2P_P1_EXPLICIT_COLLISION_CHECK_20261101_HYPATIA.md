# P1 explicit candidate identity collision check

This is a CPU-only check against two fixed, incomplete historical projections.
It does not construct, select, approve, or execute a candidate. No collision
means only no match in these inputs, not freshness, development independence,
training exclusion, sampling independence or geometric eligibility. Input
identity is a submitter claim; this tool does not authenticate its origin.
Existing P1 and source-authority requirements remain unchanged.

The tool verifies the existing historical and coherent-root artifacts and their
bound input/implementation hashes before and after comparison. It compares
both candidate seed fields against all projected root/reset seeds and half-open
reserved reset spans. Shared reset seed is conservatively reported, not asserted
to imply an identical physical root. It compares state7 little-endian float64
bits and physical pose (first four coordinates at `1e-9`, wrapped angle at
`1e-12`) against 80 historical prefix records, four historical source bridges,
the coherent development root and its requested reset state. It compares
candidate rows against one another too. Prefix states and reset requests are
retained under their own labels; a match does not prove they are sampling roots.
Every hit references a specific input JSON pointer. Missing fields, incomplete
history and training membership remain unknown even with zero hits.

## Input contract

An externally provided UTF-8 JSON file contains exactly this structure. The
illustration uses `null` because no real candidate was supplied; it is not an
executable candidate or evidence:

```json
{
  "schema": "p1_explicit_collision_check_20261101_hypatia.candidates.v1",
  "candidates": [
    {"candidate_id": "SOURCE_ID", "root_seed": null, "reset_seed": null, "state7": null}
  ]
}
```

Replace the illustration with actual, source-authorized identity values; at
least one of the three identity fields must be present per row. `state7` is
the actual source's seven finite numbers in the existing PushT state order;
do not substitute a seed-generated prediction for an observed state.
`candidate_id` is a unique nonempty identity label. The tool rejects extra
fields, including scores or outcomes, booleans as seeds, duplicate IDs,
nonfinite state values and empty rows. A partial record can be inspected but
the missing fields are explicitly reported. Pin the supplied input bytes and
retain the source-to-state/reset provenance separately.

## Run and verify

Run from `le-wm-main` with a canonical absolute candidate path, independently
retained candidate SHA, and a new absolute output path whose filename starts
with the module name. Creation is exclusive and a failed partial file is never
reused. Replace the capitalized values with actual paths and digests:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_explicit_collision_check_20261101_hypatia check --candidates /ABSOLUTE/CANDIDATES.json --candidates-sha256 CANDIDATE_SHA256 --output /ABSOLUTE/EXISTING_DIR/p1_explicit_collision_check_20261101_hypatia_NEW.json
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_explicit_collision_check_20261101_hypatia verify --report /ABSOLUTE/EXISTING_DIR/p1_explicit_collision_check_20261101_hypatia_NEW.json --sha256 REPORT_SHA256
```

Retain the printed report SHA externally. Verification rebuilds the report from
the same pinned candidate and historical bytes, rather than trusting its
embedded statuses. The two upstream saved manifests must remain available at
their existing paths. No GPU, environment reset, model call, training or
outcome-based filtering is needed. Do not interpret this as permission to
launch a prospective experiment.

Tests:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_explicit_collision_check_20261101_hypatia.py tests/test_p1_coherent_development_identity_20261101_hypatia.py tests/test_p1_known_history_exclusion_20261101_hypatia.py
```
