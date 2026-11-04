# PointMaze archive-start fixed-bank replanning diagnostic

The fixed four archive roots, eight-action bank, RGB goal timestamp and
position-only physical endpoint source from
`D2P_PC_POINTMAZE_ARCHIVE_BANK_ALIGNMENT_20261102.md` were reused. Each
packet repeats one candidate for five archive actions; all policies are
evaluated after ten archive actions. The experiment uses privileged initial
state, physical successor and positional goal, and does **not** invoke a
neural model.

| Archive root | Joint 10-step open | Joint receding, 10-step lookahead | One-packet greedy replan | Zero | Previously saved model-chosen open plan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1880 | 0.057574 | 0.076308 | 0.216248 | 0.635901 | 0.214948 |
| 1881 | 0.016933 | 0.206812 | 0.024917 | 0.248435 | 0.359256 |
| 1882 | 0.008772 | 0.270537 | 0.008772 | 0.355503 | 0.150511 |
| 1883 | 0.006118 | 0.124712 | 0.167546 | 0.202725 | 0.044546 |

Values are distances to the archive-frame-10 position; lower is better.
`joint_replan` optimizes another two-packet horizon at step 5, but only
executes its first packet before the common step-10 evaluation. Thus its
second decision optimizes a step-15 endpoint, unlike the other step-10
criteria. It uses 128 conceptual search rollouts versus 64 for `joint_open`
and 16 for `greedy_replan`, plus the respective executions; this is **not**
an equal-budget comparison. The full implementation shares initial batches
and physically evaluates 153 candidate/verification sequences per root,
not the sum of the conceptual policy budgets.

After outcomes were opened, the saved one-packet successor arrays were also
used to compute the remaining-five-step best tail. On all four roots this
tail equals the original joint-open tail, and its step-10 distance equals
the joint-open distance. In contrast, the extended-lookahead replan selects a
different tail on all four roots. This supports a **specific finite-bank,
known-root time-index mismatch**, not a general claim that feedback is bad.

The saved consumer recomputed all candidate costs and selections, checked
the initial endpoints against the prior source, and re-executed 16 selected
two-packet paths on the position backend. The same selected open-loop chain
and its split-at-five execution returned exactly identical four-component
states on these four paths. This does not establish full-state equivalence
for all counterfactuals or prove that resetting velocity is generally valid.
The backend does not provide native success/termination labels, and the
visual model's original scores also used archive proprio and normalization
whose training-only provenance is not established. The model column is a
previously saved open-loop result, not a new closed-loop model comparison.

These are four exposed roots and a finite candidate library. The result is
not a perfect-dynamics certificate, RGB-only planner, fair strong-baseline
comparison, independent P1 outcome or evidence that the mechanism passes.
Three targeted packet/validation tests passed.

- Physical report SHA-256:
  `d41a0513215237e6a9e63ec188b3544eef16be898c0dc618685c2a382378724d`
- Saved verification SHA-256:
  `ecc2cb26b2426639531c70fa494299647ee50775775b580d968dff0f9fa39d46`
