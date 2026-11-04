# Implementation lock for regenerated-RGB paired planning

This addendum fixes previously unspecified implementation details before
any new planning or suffix execution. It inherits the full
`PC_POINTMAZE_CP12_REGENERATED_RGB_PAIRED_PLANNING_PROTOCOL_20261103_CODEX.md`.

- Branch order is root order 8343,8345,8349,8351, then raw/spectral, then
  candidate 2,3,4,5. Branch index i is 0–31. Generate a CPU float32 Torch
  normal tensor `[3,16,10]` using `torch.Generator("cpu")` seed
  `2026110300+i`; transfer its same values to each CUDA arm. Sample zero
  at each iteration is overwritten by the current mean, as in existing CEM.
- Use raw-zero prior, population 16, 3 iterations, 4 elites, float32
  bounded-candidate update, sample std correction 1, floor 1e-5 and final
  two-way prior/mean comparison as specified in the parent. A fixed
  deterministic `argsort(stable=True)` resolves elite ties by sample index.
- Use one shared goal encode and 96 observation encodes (2,7,12 per branch)
  in planning. Require two separately counted one-frame wrapper parity
  checks on the first branch at batch 16 and 2: wrapper and explicit unroll
  each evaluate those batches, 36 parity samples total. The main optimizers
  score exactly 3,200 samples in 256 batches. No additional scoring samples.
- Save an append-only per-stage event journal before GPU calls, resets, and
  individual physical commands. Counters in successful receipts are derived
  from completed calls; failed or interrupted stages cannot be called
  complete. No retries after partial failure without a new run identifier
  and a new audited receipt.
- Require host GL vendor `NVIDIA Corporation`, renderer containing `NVIDIA`,
  and record the independently probed GL version and binary hashes. Seal
  source, protocol and all input/dependency hashes; seal a standalone context
  NPZ/JSON before planning, then seal complete decisions before executing.
  Context collection itself performs 384 prefix commands, not part of
  the candidate budget. Execution must revalidate the sealed decision and
  context hashes and recheck all 13 regenerated prefix frames before each
  individual suffix. Historical RGB comparison is descriptive only.
- Offline physical endpoint replay is a separate post-execution stage.
  Report branch- and root-level position distances; threshold `<0.5` is
  wrapper endpoint success, not proof of native early stopping or robustness.
  All trajectory counts and overheads remain as stated in the parent.
