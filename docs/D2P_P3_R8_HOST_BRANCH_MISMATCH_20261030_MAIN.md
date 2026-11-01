# P3 R8 host run: first-decision branch mismatch

The independently prepared R8 host CUDA job
`p3-r8-host-20261030-codex-main01` exited with status 1. Its output is
`le-wm-main/results/p3_r8_host_driver_20261030_codex_v1_RUN_MAIN01/`.
The runtime loaded the frozen model on the assigned GPU and produced a
six-row terminal record, but all rows failed and `candidate_complete` is
false. `failure.json` reports `incomplete fixed roster`. These are not
scientific or performance outcomes.

The decision journal records the actual scheduler output before the error:
all six slots, at stage 0, request `progress`, four candidates and horizon 3.
The first root has unreachable fraction 0.3618971377654383 and horizon
gain 0.11210792064295808; the second has 0.45664032081610195 and
0.0870246901654077. The three policies share each root's stage-0 signal;
they are not six independent signals.

The new `validate_decision` in `p3_decision_equal_r8_20261030_codex_v1.py`
only accepts `reconfiguration` and eight candidates. It therefore rejects
the valid progress decisions before generating R panels or dispatching
control packets. The earlier progress-only experiment encountered an R
decision after two or three packets; this run instead fails at the first
progress decision, so those observations do not contradict one another.
Warmup steps do not count as formal control actions.

The appropriate next implementation must support both actual scheduler
branches, preserve the original decision and fail closed for unsupported
values, account for all branch-specific work, and use a newly fixed contract
and output. Forcing progress into R or changing the sealed old contract
would confound the comparison.

SHA-256: decision journal
`7eed6f6b1e55b07a0e509f6dc721c8b2d949421f91f198fb26686053bfd5319b`;
end `89a50d73d117ab87dc6e478bbca39b6744b39961ccd59af85b4157b874ea99e7`;
failure `e6380c7fa9d0aac542c7070f252bc41d5a26b46a0d50d83afd62b7d1f7262501`.
These saved observations do not constitute complete cross-log certification.
