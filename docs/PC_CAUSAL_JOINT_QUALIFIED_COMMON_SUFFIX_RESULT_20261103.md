# Joint-search qualified prefix: fixed common-suffix intervention

On exposed root 8349 the same five fixed ten-action suffixes were applied
after zero, greedy and the preselected `raw_qualified` prefix. The qualified
prefix's physical distance rose from 1.6554 at cp2 to 1.6674 at cp7; zero
and greedy were already at 1.6136 and 1.4952 at cp7. Values below are
physical position distances after the same suffix; lower is better.

| Shared suffix | Zero prefix | Greedy prefix | Qualified prefix |
| --- | ---: | ---: | ---: |
| Zero | 1.4441 | 1.3611 | 1.5406 |
| Qualified model plan | 0.3621 | 0.5129 | 0.2901 |
| Greedy model plan | 0.9840 | 0.3087 | 0.9521 |
| Random 0 | 1.2654 | 1.2862 | 1.3772 |
| Random 1 | 1.4912 | 1.4505 | 1.4772 |

For the qualified model suffix, the qualified prefix wins on absolute terminal
distance and improves by 1.2505 from its own zero-suffix outcome, versus
1.0820 for zero and 0.8482 for greedy. The response is suffix-dependent:
the qualified prefix does not dominate the two controls across the other
four fixed suffixes. Saved position-vector changes show the suffix response
also differs, not just the starting distance.

The prefix was selected by the model before its first physical evaluation,
but this root and intervention were chosen after development outcomes were
visible. Its own real-RGB suffix replan in the joint-search experiment was
worse than both controls. Thus this result supports a **conditional
same-action response difference**, not a general improvement in independently
plannable capability, independent generalization, or a P1/PC pass. The
qualified model suffix was produced for that prefix; common-suffix outcomes
cannot replace separately optimized suffix comparisons.

The panel executed 15 trajectories, 255 RGB-backend steps and 390 separate
position-backend replay steps; no new model inference occurred. It inherits
joint-search and Jacobian costs. The saved audit verified suffix construction,
15 action/prefix/RGB bindings, scores and position differences, but did not
rerun the model or independently replay all saved frames.

- Protocol: `docs/PC_CAUSAL_JOINT_QUALIFIED_COMMON_SUFFIX_PROTOCOL_20261103.md`.
- Runner: `le-wm-main/experiments/pc_causal_joint_qualified_common_suffix_20261103.py`.
- Report: `le-wm-main/results/pc_causal_joint_qualified_common_suffix_20261103_DEV01.json`.
- Saved audit: `le-wm-main/results/pc_causal_joint_common_suffix_saved_audit_20261103_DEV01.json`.
