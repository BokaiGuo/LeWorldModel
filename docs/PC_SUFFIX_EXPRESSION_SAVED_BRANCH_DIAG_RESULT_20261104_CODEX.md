# Saved suffix-expression collision-branch diagnosis

The alternate line-segment implementation reconstructed all 115,200
previously saved suffix-search proposals from the stopped four-root panel.
This analysis did not generate actions or run an optimizer.

| Suffix expression | Proposals | At least one blocked step | Ever `x > 0` | Roots with a crossing | Best final distance among crossings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four knots | 57,600 | 43,230 | 23 | 4/4 | 0.7221 |
| Ten knots | 57,600 | 40,832 | 2 | 1/4 | 1.0775 |

The counts are proposals, not independent tasks; the same candidate was
searched repeatedly. The respective crossing counts cover 11 and 2 of
40 root/candidate pairs. “Ever `x > 0`” does not require the final state
to remain on that side. The best crossing-chain distances are still
above the existing 0.05 success threshold; neither arm had a success.

The goal is at `x=0.5`; wall-blocked endpoints near `x=0` therefore
explain why many best recorded distances cluster near 0.5. This
interpretation is consistent with the measured collision counts, not a
proof that every unsuccessful trajectory hit the wall or that 15-step
success is impossible. The collision semantics and scores agree with
the alternate implementation on these saved actions; that is not
independent physical validation or proof that the benchmark is
well-designed. No specific dynamics bug was established.

The source audit and saved trace hashes were verified, every suffix batch
was assigned once to its recorded candidate/run, and zero-tail initial
evaluations were kept out of the proposal counts. This is a post-hoc
development diagnosis and does not constitute PC2, PC3, feasibility, or
independent confirmation evidence.

Protocol: `PC_SUFFIX_EXPRESSION_SAVED_BRANCH_DIAG_PROTOCOL_20261104_CODEX.md`.
Raw per-run counts and dependency hashes:
`le-wm-main/results/pc_suffix_expression_saved_branch_diag_20261104_codex.json`.
