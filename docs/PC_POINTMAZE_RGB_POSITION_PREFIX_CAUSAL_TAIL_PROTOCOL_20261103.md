# RGB-position prefixes with zero versus causal-velocity tail replanning

Development protocol fixed before execution. On four exposed roots 1884–1887,
take the two complete initial plans chosen by the earlier RGB-position
planner, one from random-CEM and one from spectral-CEM. The first packet is
fixed before observing this experiment's result and never selected by
physical outcomes. The two prefixes per root are not independent.

For each prefix, execute five high-level actions with per-step real RGB.
Require the last image to equal the corresponding sealed cp5. Replan the
remaining one-packet tail with the same frozen model, same initial task mask,
same goal RGB, and same fixed CEM innovations, 3 iterations × 16 samples.
Compare RGB-derived cp5 position with zero velocity versus the causal
cp4-to-cp5 RGB position difference divided by 0.05 seconds. Each arm gets
the same per-step RGB observations, same five-step execution horizon, and
same CEM calls; execute its own selected tail physically and save all images,
plans, scores and calls. Require all first-packet frames to match within each
paired arm. Record elapsed CUDA-synchronized feedback and planning time.

Use the sealed-source initial mask intentionally: the prior fixed-bank
diagnostic used this mask, whereas the earlier closed-loop causal-velocity
study used a newly generated cp5 mask and different initial prefixes. Do not
interpret this as an isolated mask intervention. Use simulator state only
after selections for evaluation. Independently replay the full action
sequence in the position backend and compare its endpoint position against
the legacy RGB renderer's terminal position where available. Report native
strict goal-position threshold `<0.5` as an offline endpoint check, not
certified online native success. Do not stop early or invent post-success
states. Preserve all failure outcomes.

Exposed roots, fixed prefixes, and shared adaptive innovations make this
development evidence only. It does not demonstrate reconfiguration,
spectral benefit, equal full-method compute, independent generalization or
P1 passage. Training-source provenance remains unknown. The privileged
simulator-state velocity is not used by either planner.
