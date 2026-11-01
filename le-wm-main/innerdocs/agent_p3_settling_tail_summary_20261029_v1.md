# P3 settling tail: saved scalar summary

This post-hoc time-extension diagnostic inherits the complete ten-branch saved receipt by exact SHA; it does not re-run its fullstrict, physics or neural model. Five fixed endpoints each have U and Z2; cp15 and cp20 give 20 branch slots and 10 same-time pair slots. Missing observations remain null. Original termination/success and engineering status remain distinct from distance. Neither known root is an independent new root.

For each present checkpoint the consumer checks the native returned state and complete snapshot, original full goal7 and reward; it independently computes all seven residuals, squared components, native7 distance, agent/object/joint XY, raw and diagnostic wrapped angle, and agent-velocity error relative to the original goal velocity. The original native success is preserved; its historic angle expression is separately diagnostic. Pair gain is Z2 error minus U error: positive means lower U error, zero is a tie. No selection, threshold change or gate admission follows from these numbers. A zero raw tail does not imply a static state, smaller velocity error or improved reward.

All seven squared-coordinate gains are retained alongside aggregate gains; each branch also reports cp20 minus cp15 for every scalar axis, or null if either checkpoint is absent.

MAIN, after review, from `/home/gbk/code/LeWorldModel/le-wm-main` (new exclusive canonical absolute output):

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_settling_tail_summary_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_settling_tail_summary_20261029_v1 build --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_consume_MAIN01.json --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_summary_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_settling_tail_summary_20261029_v1 verify --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_physical_20261029_v1_consume_MAIN01.json --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_settling_tail_summary_20261029_v1_MAIN01.json
```

Verify reconstructs the entire deterministic summary and compares its bytes, including source/test/doc hashes. It inherits the upstream full receipt, not a fresh upstream verification. Fixture tests are not experimental evidence.
