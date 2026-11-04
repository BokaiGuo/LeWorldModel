# Successor RGB source-binding development repeat

The existing prospective protocol
`PC_SUCCESSOR_RGB_SOURCE_BINDING_DEV_PROTOCOL_20261103_CODEX.md` was run on
the same four exposed roots, two arms and three RGB decisions per arm.
Beforehand the source list was repaired: hashing only the aggregate helper
module did not bind the separate files defining its imported RGB state
estimator, scoring, model loading, encoder extraction, or the scene builder.
The revised explicit set contains 35 readable files. This is a concrete
provenance coverage fix, not a claim of full dependency closure.

The run created an exclusive before receipt, 40 decision/execution/image
files, and an exclusive after receipt containing the unchanged source hashes
and hashes of all 40 files. An independent verifier recomputed these hashes;
the native physical replay checked 24 prediction traces, observation/action
receipts and eight trajectories. All 24 final selected action chains and
scores matched the earlier development run exactly on this hardware; all
eight RGB archives were pixel-identical. Decision JSON bytes need not match:
timings and receipt metadata differ.

The immutable artifacts use the prefix
`le-wm-main/results/pc_successor_rgb_bound_runner_20261103_codex`; the
`_source_before.json`, `_source_after.json`, `_verification.json`, and
`_physical.json` files have distinct roles. The original results were not
overwritten. The source-binding, policy and abstention tests passed 28 checks.

This is repeated execution on the same hardware and already exposed roots,
not an independent sample, proof of method superiority, PC2, PC3, PC4 or PC5.
The receipt records an explicit but incomplete set of files; native libraries,
all package code, loader-time in-memory state, runtime/hardware and checkpoint
training membership are not fully certified. Hashing after modules were
imported also cannot prove that imported in-memory code necessarily equals
the later file snapshot. The earlier run's missing historical hashes remain
missing; this repeat cannot retroactively certify them.
