# Fixed-bank two-packet optimized lookahead development protocol

Use the nine sealed spectral-bank prefixes, the fixed root list, source
context and frozen checkpoint. Give each prefix the same normalized
zero-raw-action prior and the same seeded AP-compatible CEM budget: three
iterations of 16 and two final scores, for 50 two-packet samples per
prefix. The first packet is the fixed prefix and the second is the CEM
candidate; normalize actions exactly once. Assert that every first-packet
model score agrees with its previously sealed one-packet score. Retain
each optimized suffix and the final prior-versus-refined terminal scores.

Select the eligible nonzero prefix with lowest optimized lookahead score
only if it is strictly lower than both the zero-prefix and the same-bank
greedy-prefix optimized lookahead scores; otherwise do not select.
Eligibility is exactly the previously sealed predicted non-progress
condition, and ties resolve by prefix index. If none exists, do not select.
Use no physical outcomes in selection. Record all scores, winning suffixes,
the old immediate-only choice, and compute costs. Execute chosen, old
choice, same-bank greedy and zero for five high-level actions per root.
Verify the actual final RGB against the sealed bank and use the independent
position evaluator for physical goal distances. No suffix is executed here.

The suffix optimizer and two-packet horizon are an exploratory fixed design
on already outcome-exposed roots. This is model-simulated open-loop
lookahead, not a promise of real feedback behavior or independent
confirmation. Report the 450 two-packet samples and the source bank's 2 JVP,
1 SVD, and 9 one-packet samples; compare total costs before any advantage
claim. Do not weaken P1 or conflate latent non-progress with physical
non-progress.
