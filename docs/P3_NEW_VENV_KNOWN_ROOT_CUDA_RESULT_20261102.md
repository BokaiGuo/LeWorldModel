# P3 new-venv known-root CUDA run

The new environment's prepared task and candidate bank were run on host CUDA
and separately consumed from the saved journal. Consumer status is
`saved_semantics_checked_known_roots`; model hash and GPU identity checks
passed. Old/new runtime equivalence remains unknown.

The signal and reverse allocation arms used [15, 5] and [5, 15] respectively.
Each records 48 predictor samples, 2 Jacobians, 384 VJP directions and 20
metric evaluations. Their winning predicted costs by task are
`[0.972351, 0.229728]` and `[1.008468, 0.229728]`. These are model-space
development costs on two already known roots, not real physical task
outcomes, an independent test, or proof of equal total compute. In particular,
this does not pass P3 admission.

Run report SHA256:
`19425dc87593b4354f93a391246f53fd227c0d9be26bdb85f51c6e4b72e3743f`.
Saved consumer:
`le-wm-main/results/p3_venv_live_host_20261102_codexexclusive_CONSUME_01.json`.
