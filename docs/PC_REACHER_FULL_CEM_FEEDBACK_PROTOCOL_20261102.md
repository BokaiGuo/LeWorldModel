# Reacher full-horizon CEM feedback development protocol

This exploratory comparison uses the two already-observed guarded-spectral
development roots, not independent confirmation. Before execution, run a
three-packet CEM with 16 samples, 12 iterations, 4 elites and seed root+9000.
Save the complete selected parameter sequence, execute only its first packet,
then replan two packets from actual successor RGB and action history using CEM
16 samples, 2 iterations, 4 elites and seed root+5000. The existing physical
runner supplies the same successor feedback as the guarded-spectral run.

Bind the source reports, checkpoint, action statistics, root and goal RGB,
and pre/post model parameters by hash. Preserve the original and replanned
actions separately. Report forward rollout candidates and horizon-weighted
predict sample counts, VJP rows and wall time separately. Nominal initial and
suffix counts are 192 and 32 candidate rollouts, or 576 and 64 horizon-weighted
samples respectively. These counts do not certify full computational fairness:
encoder, counterfactual physical probe, GPU synchronization, and spectral
Jacobian costs are not equivalently metered. No results or gate criteria may
be selected or altered after execution.
