# P2 prefix-recompile host CUDA result

The fixed four known-context, two-arm panel completed eight of eight slots
on the host NVIDIA GeForce RTX 5060 Ti under the existing cooperative
CUDA lock. The supervisor held the original lock inode through child exit;
this mutual exclusion applies only to cooperating processes. The model hash
before and after the run was
`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`.
The child reports 4.194 s model loading and 10.245 s total child wall,
including loading. Dynamic loaded-source closure remains unknown.

The separate-process saved reader reconstructed the sealed input and
source inventory, supervisor and child journal chains, device and model
identity, all eight branch files, Jacobian/SVD and candidate formulas,
context hashes, winner replay, and resource events. Its receipt is
`results/codex_p2_prefix_recompile_saved_20261030_v1_RECEIPT_01.json`,
SHA256 `d58d1938435d1be3251e0c0b280ab46a2f0834f3b3370c89b9d78455baf508ca`.
It shares the producer's formulas and does not independently re-execute the
neural model, verify autodiff numerical accuracy, or prove complete dynamic
source closure.

| Base | joint winner model Q | prefix-chain winner model Q |
|---:|---:|---:|
| 0 | 0.952102 | 0.788026 |
| 1 | 0.511733 | 0.614643 |
| 2 | 0.752304 | 0.508096 |
| 3 | 0.752304 | 0.508096 |

Q is a predicted terminal latent MSE, not physical performance. Bases 2 and
3 share the same input context and outcomes; these are not independent trials.

Each arm had the same upper limits, without padding. Each actual joint arm
attempted and returned 36 packet calls, one full20 Jacobian, one SVD and
17 scores. Each prefix-chain arm attempted and returned 28 packet calls,
one full20 and four independent single-packet Jacobians, five SVDs and
17 scores. Each packet nested 1 action encoder, predict, predictor and
prediction projection entry; those nested entries are not additive FLOPs.
Per-arm synchronized wall was 0.367–0.574 s for joint and 0.447–0.456 s
for prefix-chain. Model loading is shared and reported separately; inherited
acquisition, engine backward calls, FLOPs and energy remain unknown.
Common ceilings are not equal total compute.

The new contract was fixed only after H25 results were known, not
preregistered, and is not an independent confirmation. The arms have
different candidate supports and compilation patterns; this does not isolate
composition. No physical replay, training, P1 admission, or P2 admission
occurred under this new contract. Original runs remain unchanged.

Inputs and outputs:

- `results/codex_p2_prefix_recompile_host_20261030_v1_PREP_01.json`,
  SHA256 `531333a5316dfdf20c4bcb628c9e0894d8c58e8af4d24e3af6157993e29ffd84`.
- `results/codex_p2_prefix_recompile_host_20261030_v1_RUN_01/report.json`,
  SHA256 `cec7be120e93f51f9e4fdd69ec0d29ab4898b5bcf479249550fd26793a684d88`.
- The same run directory contains eight branch files and both durable
  journals. Local test suite: 17 passed; fixture tests are not GPU evidence.
