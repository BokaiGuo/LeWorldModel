# Goal-image translation subspace: fixed-pair development probe

On the two already exposed Reacher roots, construct a goal-only latent
subspace without training or simulator state. Shift the processed goal image
by exactly eight pixels in each of the four cardinal directions, extending
edge pixels rather than wrapping. Encode the unshifted goal and all four
shifted goals with the frozen encoder, map each through the goal planning
coordinate, and form the four unweighted difference rows. Use their four
right-singular directions if the smallest singular value is at least
`1e-6` times the largest; otherwise fail closed. Score a predicted or
observed latent by its mean squared projection of goal residual onto that
four-dimensional subspace. No post-outcome metric selection, fitting,
retry, action selection or physical execution is part of this probe.

Apply the fixed rule to all 26 saved Q-attribution action pairs. Check the
unshifted goal encoding against the previously saved target and bind the
same predicted/actual vectors and physical outcomes. Report each root
separately, with exact ties, and retain failures. The shifted full image
includes the static background: the resulting directions are not guaranteed
to represent joints, end-effector or any task quantity. A favorable
retrospective ranking would justify only a separately frozen closed-loop
development test, not a planning advantage or P1 admission.
