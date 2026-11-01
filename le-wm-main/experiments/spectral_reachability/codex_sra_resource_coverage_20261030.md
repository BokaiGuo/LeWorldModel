# SRA resource-coverage precondition

The saved historical SRA report contains a planning sample ledger, not a
complete end-to-end budget. The saved CPU checkpoint identity proof establishes
the recorded identity check, not its load time or hardware I/O. The companion
audit checks both artifact hashes and reports present checkpoint file sizes;
sizes are not measured I/O and current checkpoint contents are not rehashed.
Other runs' partial timings and tensor counts cannot fill these gaps.

`codex_sra_io_transfer_meter_20261030.py` is an opt-in, single-process
explicit-operation interface. A caller supplies a binding hash, a verified
source hash and a cap or `None` for every resource axis. Missing or exhausted
caps reject the relevant operation before its payload work. Each attempt is
durably journaled before the callback; failed or over-limit work remains
pending, with no refund or restart. `finish` seals a hash-chained ledger and
returns its digest; `consume` needs that externally retained digest and binding.
A self-generated digest alone does not authenticate a forged ledger.

Reads count logical bytes returned by `read_asset`, transfers count logical
tensor payload, and loading counts callback entries and monotonic elapsed time.
The loading time cap is checked afterwards, not preemptively. File open and
metadata operations before reservation, ledger writes/fsync, callback-internal
operations and uninstrumented calls are not counted. Tensor payload is not
physical bus traffic, and a fixture callback is explicitly marked. Even a
completed operation does not prove total process I/O, GPU execution, or a
complete dependency closure. Unavailable totals stay null.

This module is not automatically wired into the existing checkpoint guard,
`torch.load`, model device moves, or run_bank. It does not prove a fair budget.
Future evaluation needs mandatory instrumentation across all paths, treatment
of shared model loading and accounting overhead defined in advance, and
comparable measured budgets. This change does not alter scientific gates.
