# Contact-goal prediction-versus-physics fixed-bank diagnostic

The v12 contact-goal planning outcome and evaluation-only witness are already
known. This is a **post-hoc diagnostic** on the stopped root, not new planning
or evidence of method performance. The purpose is to distinguish poor
candidate coverage from failure of the model/latent objective to rank an
actually reachable contact outcome.

Construct a 64-chain, 15-step float32 bank:

1. Exactly 56 chains at integer indices from `np.linspace(0, N-1, 56,
   dtype=int)` in chronological order across every saved full-horizon
   predictor batch of the first v12 decision. Preserve their original
   recorded scores and identities; repeated chains remain repeated.
2. Exactly eight privileged diagnostics derived from the fixed witness:
   scaled by 0, 0.25, 0.5, 0.75, and 1; first five steps then zeros; first
   ten then zeros; and the reverse-time witness sequence. These are not
   planner candidates or a comparison under matched search budgets.

Re-evaluate the entire bank against the frozen predictor and same initial
RGB/proprio/history using a dedicated isolated **evaluation-only** worker.
The worker can see these action chains for scoring but cannot submit decisions
or execute them. Separately reset and physically replay all 64 chains from
the exact verified warm boundary; save terminal images, state and physical
success metrics. Encode the actual terminal RGBs with the frozen encoder in a
separate isolated worker. Report all 64 rows, original and rescored model
scores where applicable, actual-encoded scores, absolute physical metrics,
success, number of distinct object poses and pairwise order agreement with
ties explicitly counted. Report planner-slice and privileged-slice summaries
separately; pairwise comparisons are correlated, not independent samples.
Do not change the planner or use any result to select an action.

Bind source hashes, verify context/history and action units, replay saved
states and images, and retain errors rather than replacing samples. A witness
ranking discordance is a diagnostic of this specific model, objective and
root, not a proof of why the planner failed, a physical-policy achievement,
or a PC0–PC5 pass. Stop at the fixed bank.
