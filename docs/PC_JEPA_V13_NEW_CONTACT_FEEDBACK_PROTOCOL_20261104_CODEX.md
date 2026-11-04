# v13 PushT feedback: fixed new-root development protocol

If and only if the separate `PC_JEPA_V13_NEW_CONTACT_WITNESS_PROTOCOL`
finishes with an exact, nontrivial witness, run one paired three-decision
feedback experiment at that same new seed. Otherwise record the witness
failure and stop. No root replacement, retries, adaptation or goal search.

Use isolated native JEPA planning with RGB plus legal proprioceptive history,
the final witness RGB as goal, and no witness actions or evaluator physical
state. Run v13 and a matched global-bank full-action CEM control. At each
decision, use 15/10/5 remaining actions, execute the first five, refresh the
physical observation and carry the unused tail. Each arm gets the same
1920 prediction-token cap. The global-CEM control receives the identical
unprivileged global proposal family (including zero and, after the first
decision, its own carry), selects its best bank chain by current model
score, spends the remaining budget on per-step CEM, and reserves a final
incumbent/zero-first-packet pair exactly as v13 does. Bank seeds are common
per decision; each arm has its own physical context and carry. Report actual
token, SVD, GPU predictor, encoding and wall costs separately.

Seal source/model/input hashes and each selected chain before dispatch.
Save all predictions, actions, scores, carry, observations, and physical
trajectories. Audit packet lengths, bank composition, selected complete
incumbent, paired decisions, carry reevaluation and action dispatch. Replay
both physical trajectories from reset using the same backend. Incomplete
arms remain failures; no repeat. If successful this is still one constructed
development target, not a PC2 mechanism, PC3 strong-baseline advantage,
PC4 long-horizon success, or PC5 independent generalization.
