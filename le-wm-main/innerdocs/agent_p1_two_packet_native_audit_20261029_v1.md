# Two-packet native ledger audit component

Each arm's saved complete 36-forward/16-reverse real native ledger is checked
using the original complete IO/trajectory auditor. Gradient labels are adapted
at the backend boundary so the original auditor recognizes actual AD phases;
the method and original module globals are unchanged. Saved raw/gradient/loss
and backward callback buffers are matched bitwise, then the entire method is
re-executed using only saved callback values and compared to every saved control
event and result. This does not run a model or physics.

This is a single-arm component, not the requested four-slot host or complete
saved consumer. The original winner binding, all eight durable arm journals,
partial failures, source/model/precision closure and complete output census
still need production orchestration and integration tests before MAIN can run.
