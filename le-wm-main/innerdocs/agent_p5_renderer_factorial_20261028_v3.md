# Static rendering v3: exact inherited pygame origin

MAIN actual v2 execution failed before any environment construction at the origin check.
Failure report SHA: a0ad0280f11c8754018da4496115e0852c7b038c112a334771b7a2068e1bb751.
The frozen runtime explicitly inherits pygame 2.6.1 from the project base_site;
pymunk remains in .jepa-physics-site. Requiring both in overlay was incorrect.

This new wrapper binds the exact module paths, original distribution receipt and
actual imported-file hashes. Arbitrary base/overlay origins still fail. It reuses
the frozen v2 execute/consume kernels without modifying globals or original files.
All five baseline RGB byte gates, all ten conditions, exact state comparisons,
zero Space.step/reset/neural calls, cleanup and per-operation meters are unchanged.
The read-only timestep exception remains explicit; no dynamic restore is claimed.

Prepare includes the original v2 plan, failed report, native source/pygame/pymunk
init hashes and own source/tests/doc. A separate exclusive launch binding ties
the new manifest to the result directory. Native report retains its original v2
kernel schema; the v3 strict receipt binds both layers and reruns v2 full consume.
Never treat the failed run as a physical negative or RAM tests as rendered evidence.

From the inner root, use the project .venv/bin/python -B -m with module
experiments.spectral_reachability.agent_p5_renderer_factorial_20261028_v3.
Commands: prepare --manifest NEW_JSON; verify --manifest JSON --manifest-sha256 SHA;
run --manifest JSON --manifest-sha256 SHA --output NEW_DIR --main-authorized;
consume --manifest JSON --manifest-sha256 SHA --report REPORT --report-sha256 SHA --output NEW_RECEIPT.
The run requires PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site,
PYTHONNOUSERSITE=1 and PYTHONDONTWRITEBYTECODE=1, same as v2.
No neural process is launched by this renderer. Any subsequent encoder diagnostic
must be separately reviewed and use host CUDA only; no new planner score is adopted.
