# P3 common deadline, expanded baseline, exploratory protocol

Fixed after ratio34 outcomes: two known roots, points 20/30, three arms,
same frozen model and 30-candidate-per-root bank. Each point/arm is a fresh
separate host CUDA process with common 30-second launch-to-exit deadline,
including cooperative lock wait, initialization, loading, signal, scheduling,
all model search and durable events. Consumer is outside deadline. Timeout or
failure stays in the denominator, unknown unobserved work is not zero, no retry.

Uniform uses no signal and evaluates 20 or 30 actual candidates on each root;
signal and reverse each calculate their own root signals and use the ratio34
rank rule for 20 or 30 candidates total. The baseline thereby can use the time
not spent on signal/VJP for useful additional candidates, not dummy calls.
This is a fixed search schedule, not a claim it exhausts all available time.
Each real action is charged. The common per-arm safety vector at points 20/30:
four forward axes each 80/120, metric 40/60, Jacobian 2, VJP/grad/backward-hook
each 384, SVD 2, scheduler 1. Distinct nested axes are never added or converted.
All arms also retain the same memory and disk limits. Wall time is the only
explicit shared cross-operation budget; 30 seconds is an upper bound, not
equal measured time, GPU exclusivity, FLOPs or energy.

After complete independent saved consumption, a separate downstream physical
protocol may replay all twelve fixed winners, one fresh physical branch per
point/root/arm including repeats. Native stops and failures remain in the
denominator; this is no independence or scientific admission. No root,
strategy, or budget point is selected using the new outcomes. Known roots and
historical outcomes were observed before this exploratory protocol; source
closure is not complete and cooperative locking is not global isolation.
