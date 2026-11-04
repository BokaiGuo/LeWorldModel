# Physical test of the fixed CP5 history selector

Take all 16 exposed-root branches from the sealed history diagnostic, with no
filtering by outcome. For each, execute the original saved five-action prefix
and then the preselected history-context tail from the same eight-action
bank. Native RGB is captured at steps 5 and 10. Require step-5 RGB to match
the source and, when the history tail equals the source tail, require step-10
RGB to match too. Replay the complete ten-action chains in the existing
position evaluator after all decisions are fixed, and compare its endpoint
distance to the sealed single-frame source. The new selector does not use
physical outcome for planning. All unchanged choices remain in the panel.

The history scores were computed on host CUDA before this execution. A
position-only offline evaluator is not the online simulator state, and no
new training or neural CPU fallback is permitted. The four exposed roots,
zero proprio, full-archive normalization, unknown training membership,
shared candidates, and lack of equal total cost certification prevent
independent generalization, native success or P1 claims. Score MSE and
physical distance answer different questions and must remain separate.
