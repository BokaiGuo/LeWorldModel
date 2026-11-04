# Shadow RGB bridge closed-loop integration: development result

The prewritten two-root protocol ran to completion on the host, using the
existing frozen cp12 model, one-frame unroll, RGB-derived causal proprio,
projected task score and CEM. It is a development smoke on already exposed
roots, not an independent performance comparison.

| Source branch | cp12 RGB-derived score | Final RGB-derived score | Raw/shadow frames differing |
| --- | ---: | ---: | ---: |
| `r8343_a0_p2` | 1.271078 | 0.771494 | 17/18 |
| `r8345_a0_p2` | 2.151743 | 1.723802 | 17/18 |

The score is the existing model-proprio projection derived from RGB, not a
measured physical distance or success indicator. Each decision optimizes a
five-action predicted packet, executes only its first action and replans from
the next real shadow RGB. Thus the final candidate scores do not predict the
result of the subsequently changing closed-loop policy. The raw legacy stream
received the same actions but never entered the planner. Frame differences are
not physical-position differences.

Both branches passed 13/13 shadow and legacy prefix binding checks against
their respective prior sealed sources. The run recorded 10 decisions, 500
scored candidates in 40 prediction batches, one goal and 12 feedback encodes,
four bridge resets and 68 high-level physical-simulator commands (34 per
bridge, five integration ticks each). Model parameter hashes before/after
match. Total elapsed time was 6.37 s including setup, model load, execution and
serialization; the sum of the 10 synchronized planner timings including
feedback encoding was 1.025 s. CUDA used an RTX 5060 Ti, torch CUDA 13.0;
a separate same-environment GL probe reported NVIDIA 595.99.02 and
mujoco_py 2.1.2.14. This is not an energy measurement or equal-total-cost
comparison.

Each per-decision NPZ was closed and hashed before its selected command was
sent, and the journal records prediction/command attempts and returns. Each
completed trajectory NPZ contains all actions and both RGB streams. The
result and test bind snapshots to inputs and reconstruct the CEM candidates,
elite updates, seeded noise and selected action. These writes are flushed and
closed, but are not explicitly `fsync`ed, so the protocol's durable-snapshot
wording should not be interpreted as a guarantee against power loss. Also,
a failure between the shadow return and legacy return would leave the pair's
single `command_attempt` without a `command_returned`; the journal identifies
an uncertain pair but does not separately log that intermediate return.

Tests: the RGB/clock adapter and independent saved-artifact audit both pass
(5 tests). They check saved binding and CEM arithmetic; they do not rerun
the model or simulator. The real GPU execution itself supplies the smoke
evidence. No simulator state or native raw RGB is passed to the model-facing
feedback function, and no physical-position evaluator is run. Same action
delivery does not independently prove identical qpos for these adaptive
trajectories; the earlier 64-trajectory geometry audit covered different,
fixed actions. The GL probe is not inside either bridge process. Two exposed
roots, no independently optimized comparator, no endpoint truth and no fresh
holdout mean there is no basis here for a general planning or physical-success
claim.
