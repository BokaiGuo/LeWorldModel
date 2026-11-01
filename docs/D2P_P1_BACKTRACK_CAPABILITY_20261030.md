# P1 backtrack common-prefix physical probes

The full fixed schedule contains 16 logical anchors × 21 suffixes = 336
branches, without physical-result selection or deduplication. The 21 suffixes
are zero and ±0.25 on each of ten time-major logit coordinates, transformed
with float32 tanh. The input preparation is
`results/p1_backtrack_capability_prepare_20261030_MAIN01.json` with SHA256
`21d117651b44ebcf2e89aa07d280bca47ab661d3b5bcc300323eb0d00e12294e`.
It is based on two previously observed physical roots and is exploratory.

Each branch independently rebuilds the source recipe and replays its prefix.
All 51 warmup/prefix off/on/off images and associated exported state,
snapshot and render identity must match the corresponding saved physical
anchor before any probe action. Simulator solver cache is not independently
serialized or byte-certified. Each anchor is written to a separate durable
shard; failures retain the attempted 21-row denominator and stop the run.
There are no neural calls. No checkpoint, source file or old result is changed.

The run started as `results/p1_backtrack_capability_runner_20261030_MAIN01`.
Anchor 00's 21 probes finished and the standalone saved consumer checked
their journal, source trace, action bytes, image hashes and endpoint responses;
the remaining anchors were still running when this note was created.
An anchor-specific success is not the outcome for all 336 rows.

The consumer computes zero-relative four-position deltas, their projection
onto the one direction fixed by original root and goal, central secants and
asymmetries. Missing endpoints remain missing. It does not by itself certify
independent publication-time report hashes, fairness, P1 admission, or hidden
solver cache equality. Final analysis must inspect every completed shard, bind
the report hashes, and retain failures and alias dependencies.

Tests: capability prepare/runner/consumer tests passed 5; the prior full probe
runner suite and related source checks passed 36. These tests are not a substitute
for the actual live run or an independent confirmation experiment.
