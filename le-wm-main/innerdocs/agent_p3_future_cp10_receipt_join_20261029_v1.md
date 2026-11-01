# Candidate future cp10 receipt join

This is a **new proposed ABI**, not an existing producer or a production consumer. There is no actual future producer/receipt to validate yet. It connects the existing owned projection and saved physical/PNG/latent per-row verifier without spoofing the historical bridge. The historical bridge source SHA is refused; different source bytes do not prove a new root, trusted provenance or future execution.

`join(refs)` accepts five externally SHA-bound canonical absolute descriptors named `prepare`, `report`, `completion`, `receipt`, `source`, each exactly `{path,sha256}`. Candidate producer namespace is `agent_p3_future_cp10_bridge_20261029_v1`, with `.prepare`, `.report`, `.completion`, `.consume` schemas. The prepare has `producer_source`, `expected_model_hash`, and five complete row specs as documented in the projection component. Report has the same source, prepared path/SHA, model before/after, five encoded rows (`source`, `status`, pixel and latent descriptors), complete status/errors/blocked, denominator, resource counts and literal false flags. Completion binds prepare/report SHA, publication status and the pixel/latent artifact hashes. Receipt binds report/completion, exact completion object, source, five projected endpoint tasks, same status/counts and absolute SHA `bindings` for producer, prepare/report/completion, physical reports/receipts and ten array files. The exact candidate count dictionary is `{encoder:5,projector:5,predict:0,physics:0,training:0}`; it is a declared consumer ABI, not a complete resource ledger or proof of calls. All actual producer evidence must later be checked by its own reviewed full saved consumer.

For each slot, the physical receipt's report SHA/binding and full rows match the report; selected physical row must have the existing fixed guard/history/task/action semantics. The row checker hashes and decodes all three PNGs, verifies snapshots and physical trajectory frame hashes, hashes and reads saved pixel/latent NPY files and checks latent bits against the task. Projection reconstructs all five complete endpoints using real task compact-digest encoding and original float32 normalization; whole projected endpoints must equal receipt tasks. Missing or early-stop cp10 cannot be substituted: this fixed ABI fails closed, with no reduced denominator or fallback. It cannot by itself decide whether a physical receipt was honestly generated or whether its claimed guards actually executed; the independently reviewed full physical and producer receipts are inherited. Caller-supplied source SHA is a byte pin, **not** a trust policy or independent authorization.

Returns `source_authority_verified=false`, `execution_authorized=false`, `trusted_timestamp_proven=false`, `fresh_roots_certified=false`, `scientific_admission=false`. A future producer must actually implement and seal this ABI, obtain independently reviewed full receipts and source closures, bind the pre-dispatch cohort and executor, and establish external order/controlled launch separately. No timing, novelty, P1/P3 admission or benefits are inferred. Historical data are not relabeled. This implementation only provides RAM/temporary-disk fixture proof and must not be advertised as a production-ready executor.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_future_cp10_receipt_join_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_future_cp10_receipt_join_20261029_v1 \
 --prepare /ABS/FUTURE_PREPARE --prepare-sha ACTUAL_SHA \
 --report /ABS/FUTURE_REPORT --report-sha ACTUAL_SHA \
 --completion /ABS/FUTURE_COMPLETION --completion-sha ACTUAL_SHA \
 --receipt /ABS/FUTURE_RECEIPT --receipt-sha ACTUAL_SHA \
 --source /ABS/FUTURE_SOURCE --source-sha ACTUAL_SHA
```

CLI prints only a compact join SHA and blocked status; it writes nothing, runs no NN, physics, optimizer or upstream full strict. Future paths and SHA must not be invented.
