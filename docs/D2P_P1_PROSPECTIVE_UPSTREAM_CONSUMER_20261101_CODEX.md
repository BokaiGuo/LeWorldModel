# P1 prospective upstream saved-plan consumer

This is a read-only CPU consumer for the **existing coherent saved formats**. It does not run the model, gym, CUDA, or physical actions, and does not modify old assets. It does not write an execution manifest and never authorizes confirmation.

Given four externally SHA-bound report paths (structured root, same-bank greedy, five-prefix own-suffix, single-greedy own-suffix), the consumer recomputes:

- Root trace hash chain, initial eight packets, prefix protection, all 64 exchange argmins, eight final scores, chosen full plan and command, and durable root-node charge.
- Greedy report/prepare/binding/model/root identity, same initial bank, finite eight scores, first-index argmin and exact charged action-encoding/predict/replay/selection ledger.
- S/G/Z own-suffix input identities, actual normalized warmup/prefix action history, latent-history structure, fixed goal/normalizer, prefix-plus-zero hold, 64 exchange argmins, nine-way selected suffix and node/encoding durable ledger.
- Current source bytes against all four recorded closures, reporting each missing/changed hash. Hash comparisons are not proof of authorship, model-score correctness, GPU kernel accounting or full source closure.
- Optionally, `--physical-manifest` and `--physical-manifest-sha256` read the physical seam's input contract, check every role prefix/suffix and cp5 state/RGB hash, root state/goal/RGB, and bind the four report hashes as planning evidence. Dynamics hashes are required by that physical input schema but are **not** independently verified here.

The historical formats bind the greedy input to a structured report that also contains physical outcomes; an offline consumer cannot prove the producer was blind or that files were sealed before outcomes. Nor can it establish source authority, complete alias exclusion, independent root families, genuine cp5 RGB encoding, exact dynamics, or a fair strong baseline. Therefore `no_physical_outcome_leakage`, `chronological_precommit`, and `source_independence` remain `unknown`, `confirmation_ready=false` and `P1_admission=false` even when structural checks pass. Source drift additionally yields `blocked_source_drift` and a nonzero CLI exit. `--require-confirmation-ready` also exits nonzero; a self-declared timestamp or boolean is not accepted as proof.

Known development bundle only, example:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.p1_prospective_upstream_consumer_20261101_codex \
  --structured results/p1_centered_h4_coherent_20261101_launcher_structured_RUN_COHERENTMAIN01/report.json --structured-sha256 5be4ad99b9140af2f3e0d73389f411fb08007eb71518f1b52d9984a6aeb43a1f \
  --greedy results/p1_coherent_same_bank_greedy_20261101_codex_RUN_MAIN01/report.json --greedy-sha256 df39207feda1a2239c2b2bcee120a78680e99a7c91b9393c7572b53731e9cc76 \
  --own results/p1_coherent_own_suffix_20261101_codex_RUN_MAIN01/report.json --own-sha256 e14a4e701ed4983e48801b2ed813f9daf8c0373db1771af4415118690b19cdc7 \
  --samebank results/p1_coherent_own_suffix_samebank_20261101_codex_RUN_MAIN01/report.json --samebank-sha256 7402564db52792ed3a7027b63aa7328d6c9c7b59b0e7ca881a000a16515919e0
```

To make this an independently confirmatory seam, a new upstream producer must isolate the pre-outcome root/model artifacts from future physical outcomes, bind every actual input read and seal the candidate bank, scores, prefixes, cp5 receipt and suffixes in an externally auditable order before the relevant physical suffixes. It also needs source-authorized development-disjoint roots with an audited exclusion registry, frozen protocol and thresholds, real RGB context/dynamics, full costs, and the strong baseline. This consumer does not retrofit those properties onto already opened data.
