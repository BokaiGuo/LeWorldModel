# Pinned AP source and CEM kernel parity

The authors' project identifies `daybraeklaxry/Aim-Short-to-Reach-Far` as
code and data for arXiv 2609.30036. The downloaded source is pinned to
commit `984327424ca7a2e52c75a8279197750d9d0b7c8e`, tree
`95f9f45db171af9bb56a3a7e2a71dd97f0479182`; the GitHub API reports the
commit as unsigned. The archive SHA-256 is
`f22083b2aacc9abd4c8b4a230a24e4c0e227623ab2e54372df4538337ce295a4`.
README, IMPLEMENTATION, LICENSE and REPRODUCING local Git blob IDs were
matched to the API tree; the entire extracted tree has not been independently
blob-verified.

The earlier `pc_ap_cem.py` was a paper-based implementation, not a faithful
copy: it uses NumPy float64 sampling and a population standard deviation,
whereas the pinned `code/ap/anchored_planning/gaussian_cem.py` uses torch
float32 sampling and the default sample standard deviation. Upstream also
accepts an explicit, in-bounds per-decision prior; the earlier implementation
hard-codes a zero prior. These differences can change the action proposals,
and cannot be dismissed as bitwise noise or retroactively corrected in old
receipts.

The new separately named `pc_ap_official_cem_kernel.py` implements the pinned
torch update, preserving a caller-supplied in-bounds prior, clipped scoring,
unclipped elite update and strict final-mean comparison. A CPU **analytic
fixture**, not a neural experiment, compares every one of 31 scored batches,
the final selected block and costs against the extracted upstream CEM with
the same seed, prior, bounds and callback. A separate analytic retrieval fixture
also matches the upstream source index and five-step target on six unique,
exact-source queries spanning three horizons, with unit feature weights.
The retrieval implementations differ in floating-point dtype and distance
calculation; ties and near-ties are not certified. The two parity tests,
prior validation, and existing paper-implementation tests pass: 12 total.

This establishes a narrow numerical kernel comparison only. It does not
establish general retrieval-index parity, official data/checkpoint identity, real
physical controller parity, full cost accounting, or official paper-result
reproduction. Existing NumPy CUDA and physical results keep their original
provenance and must not be relabeled as official AP. Production model calls
must use host CUDA and a separately verified real RGB/action-history adapter.
