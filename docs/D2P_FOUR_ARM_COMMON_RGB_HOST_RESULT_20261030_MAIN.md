# Common suffix four-arm host CUDA exploration

The fixed known-root four-arm run completed on host CUDA with actual RGB
feedback. The shared suffix uses eight H3 initializations, two fixed gradient
steps and final nonlinear replay. Four arms completed 20 control steps each
after warmup; all four native successes were false. There were 16 encoder and
16 projector calls completed and no reported run errors.

Final object position errors for structured, native-style, random and
immediate-progress were respectively 20.0270, 36.7523, 70.7867 and 32.4355.
These are single-known-root descriptive results, not independent evidence of
advantage. The native-style initializer is not an authenticated strong
baseline; all arms share a cap but have different actual charged work. Full
end-to-end equal-cost fairness is not established.

For all 16 nodes a separate saved-data check verified each candidate-trace
SHA, and matched the selected optimized H3 first packet to the sealed commands
and the corresponding five recorded executed raw actions. This is not a full
physical event audit or independent verification of total resource coverage.
The 15 targeted launcher and node tests passed.

Exploratory checkpoint decomposition from the saved state and goal arrays:
the structured arm's pusher-position error was 165.21, 143.56, 116.18,
90.28 and 65.96 at steps 0, 5, 10, 15 and 20; the corresponding
object-position error was 70.79, 70.79, 43.83, 20.40 and 20.03. The
object's wrapped angle error was 0.476, 0.476, 0.255, 0.066 and 0.063
radians. Thus object pose was unchanged over the first packet and improved
later, while the pusher also moved. This is a post hoc hypothesis, not a
capability intervention: subsequent actions and actual feedback also changed,
and the native success condition includes pusher position.

For the final saved checkpoint, independently recomputing the existing
native criterion `norm(state[:4]-goal[:4]) < 20` and the native angle
condition agrees with the reported false termination for every arm. The
joint position errors are 68.93, 164.98, 83.76 and 97.86 in arm order;
the angle condition is met by all except random. In particular, the
structured arm is not merely 0.03 beyond an object-only threshold: the
native joint-position error is also above 20. This arithmetic checks saved
final states and the current source semantics; it does not by itself
authenticate the complete physical trajectory or establish why planning
failed to achieve the joint target.

Saved candidate arrays show nonzero optimization updates at all 16 nodes:
mean absolute raw-coordinate changes range approximately 0.0043–0.0145 per
node. This excludes a universally frozen action update in this run; it does
not establish that the updates improved the model objective or real task.

Report: `le-wm-main/results/codex_four_arm_common_rgb_launcher_20261030_RUN01/report.json`,
SHA-256 `a035a1d6558ad88deebc8fbb6ce4384d012d7127a7cc615419b56948664ef2a3`.
This does not establish the common-suffix physical intervention, real
capability change, P1 admission, or independent generalization.
