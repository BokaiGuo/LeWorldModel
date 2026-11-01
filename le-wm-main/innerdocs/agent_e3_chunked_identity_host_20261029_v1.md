# E3 chunked identity HOST candidate

This separately named candidate loads the original model via the v3 chunked binding, then passes that same bound object to the existing identity HOST guarded same-environment episode. Existing scientific gates, event and storage limits, and HOST authorization are unchanged. The binding is chunked under the existing per-event limit; its conservative up-front category/whole-budget check is not a reservation for later encoder, guard or physics records. The prior HOST's one-record capacity proof is inapplicable and is not claimed.

From `le-wm-main`, first create a fresh authority from an already verified identity authority:

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_e3_chunked_host_authority_20261029_v1 prepare --source-authority /ABS/IDENTITY.json --source-sha256 IDENTITY_SHA --output /ABS/NEW_HOST_AUTHORITY.json
.venv/bin/python -m experiments.spectral_reachability.agent_e3_chunked_host_authority_20261029_v1 verify --source-authority /ABS/IDENTITY.json --source-sha256 IDENTITY_SHA --output /ABS/NEW_HOST_AUTHORITY.json --output-sha256 NEW_SHA
```

The source authority's own original fixed control5/resolver and source checks remain in force. The new host and chunk transport are additionally pinned and rehashed. The authority preparation module is also pinned. Creation is exclusive and does not overwrite. Failed writes can leave a partial new file, which is not valid authority.

Only with separate explicit HOST authorization, the future entry is `python -m experiments.spectral_reachability.agent_e3_chunked_identity_host_20261029_v1 run --authority /ABS/NEW_HOST_AUTHORITY.json --authority-sha256 NEW_SHA --output /ABS/FRESH_JOB --authorization MAIN_HOST_AUTHORIZATION`, replacing the last token with the original launcher's exact authorization. This document does not grant it. Saved inspection uses `inspect` with the same authority/output and explicit `--report-sha256` and `--completion-sha256` values. It verifies the original saved consumer, journal and source checks, a distinct host marker, chunk ordering/hash, loader identity and encoder-boundary joins. A partial unsealed binding is only partial; old host records cannot satisfy the marker. Runtime identity is source-trusted, not a cryptographic proof of process memory.

Tests are CPU fixtures: two bound encoder callbacks through the actual loader fixture, source authority creation/verification, non-overwrite, tampering and partial/marker rejection, plus an inspector seam fixture. The inspector seam stubs the prior consumer, so it is not a full end-to-end physical episode. No actual CUDA, model loading, environment, physics, neural parity, or scientific evaluation was run. Existing `full_saved_strict` and scientific admission stay false. Before any production decision, review the real source closure and execute an independently authorized, bounded same-environment host run and independently inspect its persisted evidence; these fixtures do not close physical observability, counterfactual controls or real multi-step evidence.
