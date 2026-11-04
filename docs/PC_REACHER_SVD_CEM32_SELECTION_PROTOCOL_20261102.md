# Response-SVD full-bank CEM32 selection: development protocol

Evaluate the same 13 previously generated and physically exposed prefixes on
the same two known Reacher roots with the frozen model. For each prefix,
predict its successor and optimize a two-block tail with 32 candidates,
three CEM iterations, four elites and the same `root + 5000` seed.
Keep the existing model-only non-greedy eligibility margins of `1e-6` and
select the lowest predicted future cost among eligible candidates.
Also report the same-bank immediate greedy and the neutral-prefix index 0.

All candidates and physical outcomes on these roots were previously opened.
This is an exploratory change in compute and cannot confirm P1, select a
confirmatory root, or justify a favorable retrospective comparison. Report
every candidate's immediate and future scores, suffix parameters, all model
sample calls, unchanged model hash and errors. The selector is not allowed
physical state or outcome inputs. The 13 candidate suffix searches alone
cost `13 × 32 × 3 = 1,248` rollout evaluations per root; full encoder and
candidate-generation costs remain outside that figure. Subsequent real-RGB
physical execution must include selected, neutral and greedy without
outcome-based filtering, with all initial and reranking costs disclosed.
