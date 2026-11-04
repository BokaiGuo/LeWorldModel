# CEM32 model-scored action chains: physical development replay

On the two previously exposed roots, replay the selected, neutral and
same-bank-greedy prefixes and the exact corresponding CEM32 suffix parameters
from the model-only selection report. All six chains are included, with no
outcome-based selection. Each chain uses the original saved five-step raw
prefix and a CPU-decoded ten-step tail with the sealed action statistics.
Record goal/root identity, every executed step, termination and physical
distance. Compare saved predicted terminal costs and final physical joint
distance only as rankings for approximately the same bounded actions.

The CPU decoding of saved float parameters is not certified bitwise identical
to the CUDA transformation used during model evaluation. This is an
open-loop physical replay on known roots, not a new RGB-feedback planning
run, a fair baseline benchmark, or P1 confirmation. A separate same-time
actual RGB encoding is needed to distinguish model prediction error from
metric/physical mismatch. Preserve all errors and stopped arms; do not fill
missing outcomes.
