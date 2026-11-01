# Spectral8 cp10: three-prefix development feedback

Three separately sealed runs use the same known root0, cp10 goal and frozen
model. The selected and zero arms use the same spectral8 optimizer; zero
replaces the first packet with zero but retains the selected model plan's
tail as a candidate for later rescoring. The greedy arm uses the minimum
one-packet predicted distance among the same eight optimized plans; ties use
the first index. Its eight extra model evaluations are charged in its node
ledger. After the first packet, each run replans from its own actual RGB and
executed action history.

| Arm | First native success | Pusher XY at 0 → 5 | Object XY at 0 → 5 | Wrapped angle at 0 → 5 | Joint XY at 0 → 5 |
|---|---:|---:|---:|---:|---:|
| selected | 13 | 141.8206 → 113.5065 | 101.6764 → 100.7559 | 0.6222 → 0.6146 | 174.5027 → 151.7744 |
| zero | none through 20 | 141.8206 → 142.6493 | 101.6764 → 101.6764 | 0.6222 → 0.6222 | 174.5027 → 175.1768 |
| same-bank greedy | 16 | 141.8206 → 72.8841 | 101.6764 → 63.2054 | 0.6222 → 0.3148 | 174.5027 → 96.4729 |

The selected arm's immediate physical improvement precludes presenting this
particular prefix as a no-immediate-improvement witness under these metrics.
Different early-stop times are not padded. The node ledgers are
`closed_unverified` and total-budget equality is not established; the greedy
and zero interventions were designed after previous outcomes were visible.
This is one development root, not independent confirmation, and no P1 gate
is changed.

The separate common-tail physical extension uses the same root and the
structured tail, but its greedy prefix is from the historical DLS-mix bank,
not this spectral8 bank. It therefore cannot be substituted for a same-bank
mechanism comparison. Its DLS-greedy/common-tail branch stopped at step 11
with native success; it has no observed step-15 endpoint. The zero/common
branch completed step 15 with joint XY 87.9792. Both are post-outcome
development probes; see `docs/D2P_P1_CP10_ZERO_GREEDY_TAIL_RESULT_20261101_CODEX.md`.

Reports under `le-wm-main/results/`:

- `codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01/report.json`
- `codex_p1_cp10_zero_carry_20261101_exclusive_RUN_CODEXZERO01/report.json`
- `codex_p1_cp10_greedy_carry_20261101_exclusive_RUN_CODEXGREEDY01/report.json`
