# Initial proprio is a major condition in this fixed-bank diagnostic

The same frozen model, inputs, 384 two-packet actions, time index and
task-mask visual objective were used in both conditions. The only changed
model input was raw initial proprio: all-zero versus the archive's
four-dimensional initial state. Visual context tensors were exactly equal.
This state is privileged and is used **only in this diagnostic**.

| Diagnostic across eight root/arm groups | Raw-zero proprio | Archive initial state |
| --- | ---: | ---: |
| Mean task-mask visual prediction MSE | 0.65385 | 0.09074 |
| Goal-cost reversals against actual RGB | 4,960 / 9,024 | 858 / 9,024 |
| Range of centered action-response cosines | -0.0079 to 0.0776 | 0.8773 to 0.9385 |
| Mean position-replay distance of model-selected fixed-bank plan | 1.15874 | 0.17882 |

Each model independently selected from its fixed 48-candidate bank; position
labels were accessed only for post-selection evaluation. All eight choices
changed, and every selected candidate's saved position distance was smaller
in the privileged condition. These are open-loop fixed-bank outcomes already
executed in a prior development run, not eight independent tasks or new
closed-loop experiments. The 9,024 pairs are highly dependent.

This is strong *conditional* evidence that replacing the model's proprio
input with raw zero materially changes its predictive behavior on these
roots. It is not proof that the model itself is intrinsically incapable
of action prediction, nor proof that the raw-zero choice is an accidental
coding bug: it was the explicit implementation used to avoid privileged
state. It also does not show that a causal RGB-only estimator can recover
equivalent state information. Actual RGB versus physical goal rankings
still disagreed in the previous fixed-bank study.

Training source and normalizer provenance remain unverified; the shared
normalizer includes all loaded archive episodes. The original source
checkpoint expects proprio as an input, and exposing simulator state would
change the study's deployment information budget. Therefore no
privileged-state result may be reported as an RGB-only planner or P1 pass.
The actionable next step is a separately budgeted, training-free RGB/action
history state-estimation interface, with visual-only information and
uncertainty, tested against fixed-zero and privileged references before any
independent confirmation.

Source hashes, model identity, candidate selections, cost arithmetic and
feature-response calculations were verified by three tests across this and
the prior decomposition. The tests do not independently rerun model
inference or certify the underlying original checkpoint training interface.

- Protocol: `docs/PC_POINTMAZE_PROPRIO_ACTION_RESPONSE_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/a0_pointmaze_proprio_action_response_20261103.py`
- Report SHA-256:
  `ad34847ebcd6c4e6896cdda08b53838f0b16c1932a5e5816a7504dae8c9251e5`
- Array SHA-256:
  `cea199b77caca3d76cfeccd6f9607d611838d557a45248da95528f105324f9e8`
