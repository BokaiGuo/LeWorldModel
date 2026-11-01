# K17 physical gate seam, not a runnable adapter

`authorities(slot)` loads the pinned K17 receipt and uses the frozen replay
bridge to extract actual paid whole and segmented tokens and the original
reset/history/goal reference. It does not rerun upstream fullstrict.
`capture_check` hashes 33 warm PNG files and the plan0 PNG before invoking the
frozen live/replay guard. It accepts a caller-provided capture; it does not
authenticate that capture as native or perform a reset. `record_preflight`
durably records guard acceptance/rejection but cannot issue a physical intent.
`consume_preflight` verifies only its journal chain and no-dispatch flags.

The 64 MiB bound leaves 12,259,052 bytes for all journal/report/completion
metadata after 306 worst-case PNGs and 8 MiB reserve. This is not a proof that
native snapshots, operation receipts and failure records fit. Their serialized
sizes and duplication must be bounded and enforced, including partial writes
and final publication, before a real adapter may dispatch. A quota alone would
fail closed but would not guarantee that all six complete records fit. No
capacity certificate, actual native receipt, equal-budget claim, or production
entry point is provided by this seam.
