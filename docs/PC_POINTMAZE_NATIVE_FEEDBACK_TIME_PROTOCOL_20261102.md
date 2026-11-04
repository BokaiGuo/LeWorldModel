# Development protocol: native RGB feedback and time alignment

Use exposed archive roots 1880–1883, archive frame 0 as initial environment
state and frame 10 as RGB goal, the previously fixed eight physical-action
directions, and the development-selected `(3,3)` legacy camera. Each packet
repeats one direction for five archive actions.

At the initial RGB, both arms evaluate the same 64 two-packet combinations
with the frozen model and choose the same lowest predicted terminal-latent
cost. Execute the first packet in separate identical native RGB environments,
then re-encode the actual post-step RGB at step 5. At that state:

- `remaining` evaluates each of the eight possible one-packet tails.
- `extended` evaluates all 64 two-packet tails but executes only the first.

Both are physically evaluated at step 10, so the extended arm scores a
step-15 prediction for a step-10 execution. Each arm uses an independent
environment instance, the same feedback point, action bank, goal and first
packet. Report actual encoder counts, prediction samples, modeled transition
counts and elapsed times; the two arms intentionally do **not** have equal
search or computational budgets. They do not represent a fair D2P versus
strong-baseline result. No root or action may be selected after observing
the result.

The checkpoint expects proprio. Supply a fixed raw zero vector for every
model context and goal, rather than simulator state; this is a missing-modality
ablation, not an authenticated training-distribution input. The existing
normalizer is computed from the full archive, including these exposed roots;
its training provenance remains unknown. Consequently this diagnostic is not
independent confirmation, a full RGB-only model certification, or P1.
Physical position scoring is offline evaluation only and must not feed back
into either model choice. Do not train or alter the model or gate.
