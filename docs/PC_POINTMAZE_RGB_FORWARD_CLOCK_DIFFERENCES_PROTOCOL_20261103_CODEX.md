# Forward trajectory discrepancy: diagnostic replay

The serial audit passed sealed-image and immediate boundary-state gates, then
failed exact equality of the two full trajectories. Before this replay, freeze
the identical 64 trajectories, worker and inputs. Save its complete raw
arrays, sealed-frame match count, exact state mismatch count, first differing
tick, and maximum absolute qpos/qvel/time differences. Report geometry residuals
against each arm's own direct qpos, and RGB differences. No re-selection, model
calls or physical-state input to a planner. This is a post hoc numerical
diagnostic, not an independent confirmation or proof that small errors cannot
grow in other trajectories. Preserve the failed attempt and do not deploy a
bridge under the original exact-trajectory condition.
