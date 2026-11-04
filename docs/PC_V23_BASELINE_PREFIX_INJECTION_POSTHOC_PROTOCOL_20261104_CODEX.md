# v23 baseline-prefix coverage diagnostic (post hoc)

Use only the eight already-audited v23/shared-CEM development pairs. Do not
rerun either planner or alter the original stopping rule. For each root,
use the CEM selected chain as a diagnostic reference, irrespective of
success. Independently replay the original recorded candidate prefixes at
depths one and two, joined to the corresponding tail of that CEM chain.
Compare all generated prefixes with the originally retained subset.

Report the reference score, minimum injected full-chain distance for all
and retained prefixes, number of injected chains below the original 0.05
threshold, and nearest prefix in action RMS and in physical successor
position. Report every root, including failed and unsuccessful baseline
references. Recompute reference and injected outcomes using the independent
line-intersection implementation and bind the original manifest, plan,
audit and per-root trace hashes.

This is a privileged, post hoc coverage/retention diagnostic: the CEM tail
is not available to the original compiler for free. A successful injection
does not imply that the compiler could find that tail; a failure does not
prove no other tail works. State proximity is diagnostic and is not asserted
to preserve heading or controllability. Do not promote these records to PC2,
PC3 or independent confirmation.
