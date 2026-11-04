# RGB state at initial planning and feedback: development protocol

Purpose: test whether the previously evaluated analytic RGB position input
can be connected to both initial planning and real-observation feedback,
rather than reusing a first action selected with zero proprio. This is an
exploratory interface and control experiment, not an independent P1 gate.

Use the four previously exposed roots 1884--1887, frozen checkpoint, archived
RGB goal at frame 10, source dataset and previously sealed initial/tail CEM
innovations from `a1_pointmaze_task_spectral_cem_dev_20261102_DEV01`.
On each root run four methods from the identical native reset:

* raw-zero proprio with random CEM;
* raw-zero proprio with task-Jacobian spectral-initialized CEM;
* RGB analytic position with zero velocity, random CEM;
* RGB analytic position with zero velocity, spectral-initialized CEM.

At initial RGB, encode the specified state input. For the spectral arms,
compute the task-masked visual Jacobian under that same context and use its
leading right singular direction to inject the existing four bounded
proposals. All arms use the same three generations, sixteen candidates per
generation, initial horizon two action packets, and sealed innovations.
Execute the first selected packet for five high-level steps, observe real
RGB, encode the same arm's state convention again, and optimize one packet
with the same tail CEM noise and budget. Execute that packet. Persist images,
input states, actions, proposals, scores, costs and identities. Reset state
is used only to initialize the environment; target physical position and
replay backend are evaluation-only.

Position is inferred only from RGB and pre-existing fixed scene/camera
metadata; unknown velocity is explicitly zero, not measured. Marker or
metadata failure aborts. Spectral structure incurs two JVP calls and one SVD;
it is extra relative to random CEM, so matching prediction counts alone is
not a full equal-compute claim. Time each planning and structure component.
Model and source hashes must match; never overwrite outputs. Compare the
zero-state arms against the saved earlier source to catch interface drift.
The same four exposed roots are development data, not independent outcomes.
Native physical success, performance significance, RGB generalization,
non-greedy reconfiguration and P1 passage are not implied by this protocol.
