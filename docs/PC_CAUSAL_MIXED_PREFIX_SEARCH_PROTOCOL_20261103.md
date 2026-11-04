# Causal mixed-prefix search: development protocol

Use the same four exposed roots, two zero observation steps, checkpoint,
normalizers, task score and causal RGB state as the saved full-bank history
experiment. Reuse its 2-by-10 causal action-response Jacobian and charge its
10 JVP calls per root to both arms for this controlled diagnostic; source
finite-difference checks are separately reported, not considered free.

For each root use four uniform unit-ball draws in ten dimensions from seed
`2026110390 + root`. Scale the raw draws by 0.5. The spectral arm uses the
same coefficients in the full right-singular basis of the saved Jacobian,
with the first two coefficients multiplied by 0.25 and the remaining eight
unchanged; it is therefore deliberately null-direction-emphasizing rather
than distribution-matched. Do not normalize individual candidates after
transformation. All actions must be in [-0.5, 0.5].

Both arms include the same zero prefix and the original causal-bank greedy
prefix, defined before new outcomes as the minimum first-packet model cost in
the source run. For each of their six prefixes run the same 50-sample
three-packet lookahead: one fixed prefix plus two CEM-optimized packets.
Save predicted prefix and full-chain scores, CEM scores and plans. Within
each arm report (a) the terminal-cost winner without a nonprogress
constraint and (b) the best nonzero model-nonprogress prefix only if it
strictly improves predicted terminal cost over both zero and greedy. If no
candidate qualifies, report `null` and do not reinterpret the winner as a
mechanism candidate.

Before physical execution freeze both arms' terminal winners and candidate
decisions. Physically execute the four predeclared branches zero, greedy,
raw winner and spectral winner even when arms alias. After each five-step
prefix, use the two most recent real RGB frames to estimate the current
position and velocity, encode the latest RGB and estimated state, then replan
and execute ten actions with the same 50-sample two-packet CEM. The actual
action history is recorded but is not an explicit input to this one-frame
encoded context. Store actions, RGB histories and posterior plans, plus cp2, cp7 and cp17
physical position distance from a separate position-backend replay. Simulated
position is never used to generate or select a candidate.

Report nonprogress separately in model and physical metrics, compare winners
to shared baselines, and retain all failures. The prefix selection uses a
proprio-encoded RGB estimate rather than visual-token latent distance. Record
individual model sample counts, encodings, JVP provenance and environment
steps. Same candidate count and suffix optimization are not a full cost
certificate. This is not a common-suffix intervention, independent-root
confirmation, native success experiment, or scientific gate.
