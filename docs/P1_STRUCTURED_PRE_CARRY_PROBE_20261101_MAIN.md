# Structured pre-carry retrospective physical probe

The existing root0 structured run's sealed first packet was replayed, followed
by its cp5 pre-carry H3 candidate 0. This is not the original feedback policy:
the original cp5 selection chose carry candidate 8 after rescoring. The
retrospective branch executed 20 control steps without native success.

At control steps 0 and 5 respectively, object-position error was
101.6764 and 100.7559, object-angle error was 0.62219 and 0.61458,
and pusher-position error was 141.8206 and 113.5065. All three reported
errors decreased over this prefix. Therefore this particular single branch
does not establish a prefix with no immediate progress under any of those
individual measurements. Nor does it compare against zero or greedy or show
that the prefix changes subsequent control capability.

The manifest and report are at
`le-wm-main/results/codex_p1_s_own_probe_20261101_exclusive_RUN_CODEX01/`.
The run used sealed earlier CUDA-generated actions, not new neural calls.
Its host physical replay and seven passing regression tests do not constitute
independent confirmation or P1 admission.
