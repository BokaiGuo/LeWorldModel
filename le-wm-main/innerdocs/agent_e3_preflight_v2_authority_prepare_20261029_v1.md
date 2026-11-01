# E3 v2 preflight authority builder

This saved-only CLI reads a canonical absolute existing identity HOST authority and its explicit SHA. It invokes the original authority reader and source rehash, retains its exact cell/control5/resolver contract, and adds current SHA pins for preflight v2, array transport, capture, encoder and this builder. Existing conflicting pins fail closed. It writes a new authority using the original five-field ABI, exclusively (`xb`), then verifies by rebuilding it from the separately supplied old authority and validating the new authority with the original reader and the v2 preflight's source closure. Existing source and output files are never overwritten.

This inherits the original reader's saved report/receipt joins, not a fresh upstream full strict or proof of model execution. It does not load a neural model, invoke CUDA, perform physics, or admit any scientific gate. A write or post-write verification failure can leave a partial/new file; do not reuse that path without inspecting it. All source and output arguments must be canonical absolute paths, and MAIN supplies the actual pre-existing authority and SHA; no authority path or SHA is guessed here.

From `/home/gbk/code/LeWorldModel/le-wm-main`, with placeholders replaced by actual absolute values:

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_e3_preflight_v2_authority_prepare_20261029_v1 prepare --source-authority /ABS/EXISTING_AUTHORITY.json --source-sha256 EXISTING_SHA256 --output /ABS/NEW_V2_AUTHORITY.json
.venv/bin/python -m experiments.spectral_reachability.agent_e3_preflight_v2_authority_prepare_20261029_v1 verify --source-authority /ABS/EXISTING_AUTHORITY.json --source-sha256 EXISTING_SHA256 --output /ABS/NEW_V2_AUTHORITY.json --output-sha256 NEW_SHA256
```

Prepare prints the new SHA; verify independently requires that SHA and the original authority SHA. This file is an input to the separately reviewed v2 preflight, not its authorization to run.
