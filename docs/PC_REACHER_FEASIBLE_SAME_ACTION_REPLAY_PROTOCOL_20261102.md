# Reacher feasible-candidate same-action replay protocol

For each of the five predeclared arms in the two-root feasible-prefix physical
report, replay its exact saved 15 raw actions. At steps 5, 10, and 15 record
the physical qpos distance, the actual RGB encoding, an open-loop prediction
from the root, and a one-token prediction refreshed from the preceding actual
RGB/action context. Use the existing frozen checkpoint, exact normalization,
and host CUDA. Check raw/normalized action decoding, frozen model identity,
root/goal RGB, step-5 RGB and physical distances at steps 5 and 15.

Compare selected, zero, greedy and both refined candidates at each identical
time, without ranking by observed outcomes to select a method. Separate
prediction-vs-observed latent error, observed latent-vs-physical ranking,
and model-vs-physical ranking. Each arm may have a different actual optimized
tail; comparisons across arms are not same-action pairs. The three points
within each arm are matched model-versus-physical outcomes. These are known
roots and action replays, not fresh episodes, control gain, or P1 admission.
Report model/encoding costs and any failed binding; no post-hoc exclusions.
