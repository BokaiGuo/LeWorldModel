# Shared-prefix tail rescore with three real frames

On the four already-exposed PushT development tasks and exactly the 301
previously frozen shared-prefix candidates per task, compare single-frame
rescoring with a three-frame context. Replay the selected first ten environment
steps from the original initial state, encode actual rendered images at steps
0, 5 and 10, and use the two already-executed action packets as history.
Predict the same three remaining packets and select the minimum model goal
cost, before reading any physical outcome labels. Verify the replay's initial
snapshot, actions, and every recorded state against the original trace, and
verify the step 5/10 encodings against the earlier alignment artifacts.

Freeze the full score arrays and action identifiers, then match the chosen
chains to previously recorded physical outcomes. Report ever-success,
terminal-success, success ranks, and incremental candidate-packet evaluations
and synchronized predictor time. This is a post-hoc conditional comparison,
not a fresh feedback-policy trial or an independent task sample. The original
planner actually dispatched a longer prefix. A successful offline reranking
would not establish action-effect descriptor contribution or PC4.
