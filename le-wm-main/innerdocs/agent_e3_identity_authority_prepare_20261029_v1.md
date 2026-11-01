# E3 identity authority: saved control5 inputs

This saved-only builder creates the original five-field identity HOST authority from explicit canonical paths and SHA256s. It checks fixed control5 prepare and resolver pins, actually reads and hashes the resolver, joins the original control5 prepare/report/completion/strict receipt and row using the original reader, rehashes the prepared source closure, and adds source pins required by the original launcher, integrated reader and identity host plus this builder. It writes a new file exclusively and verifies it by complete reconstruction and the original identity authority reader. It does not use a fixture to construct the authority.

The reader inherits the old saved strict receipt and checks its joins: it does not rerun the upstream full strict, execute a model, prove future device/encoder identity, or admit a scientific gate. No input or output is overwritten; failed publication or post-write verification may leave a new partial file for inspection. Source files and saved inputs can change between operations and then verification fails closed. The selection below is the already-saved cell00, not a fresh root or outcome-selected replacement.

From `/home/gbk/code/LeWorldModel/le-wm-main`, the following uses the existing MAIN01 saved files. Change only the new output path if it already exists; do not overwrite it. Prepare prints the new identity authority SHA.

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_e3_identity_authority_prepare_20261029_v1 prepare \
 --prepare /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha256 ea8e4feae44e31876852c960db03ce4613184c55fc0745ef8f890f70485f8425 \
 --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01/report.json --report-sha256 e6d465652eb993e31d19ce11e5e0f494cb18bf54e65d4d9f0cac983a733e7385 \
 --completion /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_cell00_MAIN01/completion.json --completion-sha256 ca309a42a5bf7ab8344b8c1672b66829d3b8bc3b4a82fac3650f04ce5d360b1b \
 --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_gd_host_cuda_20261029_v1_consume_cell00_MAIN01.json --receipt-sha256 9b5b8751e38138b6f127d05de331e54bd355075501cdaf6914fb1fb3e63b945a \
 --resolver /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json --resolver-sha256 b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26 \
 --cell 0 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_identity_authority_prepare_20261029_v1_MAIN01.json
```

For independent verification, run the **same arguments** replacing `prepare` immediately after the module name with `verify` and append `--output-sha256 THE_PRINTED_SHA256`. Then supply the verified identity file and SHA as `--source-authority` and `--source-sha256` to `agent_e3_preflight_v2_authority_prepare_20261029_v1 prepare`; its distinct output must also be a new canonical absolute path. This does not itself authorize any CUDA preflight.

Fixture command:

```sh
.venv/bin/python -m pytest -q tests/test_agent_e3_identity_authority_prepare_20261029_v1.py tests/test_agent_e3_preflight_v2_authority_prepare_20261029_v1.py
```
