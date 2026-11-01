# Fixed-prefix finite-difference native-ledger component

RAM-only callback orchestration of 64 ordered groups with per-group native
resource caps and fail-closed roster, plus saved single-group native IO and
control reconstruction. Valid pair counts are determined before dispatch;
invalid pairs remain in the fixed 40-pair denominator and are not silently
clipped. Actual forward attempts and completions use the native ledger.

This is **not** the requested complete production host: it does not rebuild
source strict, persist a journal, seal CUDA model/class/precision, enforce
disk/wall/GPU budgets, or supply prepare/run/consume. The caller must not use
the component to start CUDA. Exceptions before ledger creation have unknown
cost, not a zero-cost certified receipt. The callback interface and fixture
native transcript are not production evidence. No model, physics or CUDA was
run. MAIN must review a separate completed host before any production.

Test: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_fd_host_component_20261029_v1.py tests/test_agent_p1_fixed_prefix_fd_20261029_v1.py`.
