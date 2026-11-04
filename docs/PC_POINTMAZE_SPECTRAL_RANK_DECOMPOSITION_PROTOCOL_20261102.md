# Fixed-candidate spectral ranking decomposition

Use the sealed four-root A1 development result and all five candidates from
each of its spectral, random and axis sets: 60 candidate slots total,
including repeated zero actions. Do not regenerate or select candidates.
For each slot reset the native RGB environment to the original float32-cast
archive state, execute its saved five-action packet, record the actual RGB,
and encode it using the same frozen model, fixed-zero raw proprio and
preprocessing as the A1 run. Obtain the physical endpoint separately with
the documented position-only replay backend. The archive frame-10 RGB and
position remain the common goal.

For each root and candidate set, save:

1. the original model-predicted latent goal cost;
2. the actual RGB-encoded latent goal cost at step 5;
3. the physical position distance at step 5;
4. the model-selected index, actual-latent-minimizing index and
   physical-minimizing index;
5. all ten within-set pair margins and signs for all three measures.

Use exact sign, preserving and counting ties separately. These candidate
pairs share trajectories, roots and duplicate actions and are not
independent statistical units. Never use outcome results to rerank the
original A1 method, modify the candidate library, gate or protocol. Bind
the old selected candidate's replayed RGB hash and position to its saved
result; fail rather than silently substitute if either differs. Save hashes,
parameter identity and compute counts. The source score and physical
outcomes were already exposed; this is an exploratory attribution, not a
fresh test, a strong-baseline comparison or P1 confirmation.
