# Three-prefix, matched ten-step capability preparation

This is a post-outcome known-root exploratory design. It fixes all eight
root/condition/arm logical pairs, each with selected, zero, and same-bank
stage-greedy prefixes; duplicate prefixes are labeled, not discarded.
The source incumbent is candidate zero, the original mixedmode winner prefix.
The separately saved mixedmode analysis identifies that winner as the minimum
stage-cost candidate in the complete 222-prefix bank in all four slots, with
stable index tie-breaking; the preparation checks source identity, the saved
consumer's task identity and literal incumbent prefix bytes. This is not a
claim that an arbitrary incumbent is greedy, nor that the two-packet winner
objective generally equals greedy.

The two already-fixed selected/zero anchors come from accepted prefix-null
MAIN01; greedy anchors come from accepted original MAIN02 incumbent rows.
They share the same native root, task, arm and warmup source, but have separate
51-frame anchor traces and checkpoints. Runtime must independently reproduce
each trace and preserve solver-cache limitations; it may not substitute a
selected anchor for a greedy anchor or reuse physical responses as new runs.

Each three-way group shares exactly the same 42 ten-step suffix byte strings:
zero, +/-0.25 logits on each of 20 coordinates (tanh to raw float32), and
that logical selected plan's original ten-step suffix. The latter is explicitly
post-outcome exploratory. Full planned denominator is 8 x 3 x 42 = 1008.
No prefix, root, probe, stop, failure or censor is discarded; missing endpoints
remain null, and native terminal must not be padded. Out-of-domain censoring
requires the original strict incident evidence; all other failures remain
engineering failures. No outcomes from the running 16 x 21 probe are inputs.

Under the existing native adapter assumptions and no early stops, design
arithmetic is 1008 constructors and closes, 2016 resets, 25200 public
steps, 256032 internal physics steps, and 81648 explicit renders. These are
upper schedule counts, not measured costs, and exclude additional failure or
retry operations (retries are forbidden). Complete operation attempt/return
accounting, storage/wall bounds and failure persistence must be implemented
and validated by a future executor. Historical proposal and shared costs remain
unknown, not zero; this is not equal-budget planning.

This artifact prepares inputs only. There is no ten-step executor or saved
consumer here; the existing five-step consumer cannot validate ten-step
probes. No physical, GPU or neural operation is started. No external signature
or new approval gate is introduced. Neither these roots nor duplicated arms
are independent confirmations, and P1 admission, freshness and equal-budget
certification remain false.
