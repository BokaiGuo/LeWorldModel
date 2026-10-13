# Revision Log

## Pass 44: Cross-horizon normalization and density robustness

- Rechecked the runtime boundary: the controlled project PyTorch probe runs
  with `torch.cuda.is_available() == True`, one NVIDIA GeForce RTX 5060 Ti,
  and a successful CUDA tensor operation. The earlier `False` result came from
  the restricted execution context, which could not see `/dev/nvidia*`; it is
  not the current experiment-runtime result.
- Completed the nine-cell frozen cross-horizon robustness matrix: public,
  validation-split z-score, and raw/100 action normalization crossed with two,
  four, and eight windows per episode. Every cell has 21 held-out episodes,
  CUDA inference, leave-one-episode-out fitting, rank-8 controls, and 2,000
  episode-cluster bootstrap replicates.
- Added the machine-readable robustness summary and Markdown table under
  `results/pusht/lewm/external_cross_horizon_readout_20261004/`, together with
  a completeness regression test and a reproducible summarization script.
- Updated the main text and Supplement Table 55 to report positive forward
  dynamics-minus-random contrasts across the declared matrix, while retaining
  the boundary against physical causality, causal action effects, planning,
  task success, retraining gain, and universal world-model claims.
- Recompiled the main paper (14 pages) and supplement (32 pages); the first
  supplement build exposed and the second build fixed a seven-column table
  declaration. The final LaTeX build has no fatal errors or unresolved refs.

## Pass 45: Cross-horizon source-fit robustness

- Ran 12 additional frozen cross-horizon readouts using independently fitted
  source basis pairs from splits 20261006 and 20261007, crossed with public,
  validation-split z-score, and raw/100 normalization and two/eight windows
  per episode.
- Every forward dynamics-minus-random interval and every $4\to8$
  residual-minus-dynamics interval has a positive lower endpoint in all 12
  cells. The complete JSON/Markdown summary and CLI completeness test are
  stored under the source-split robustness result directory.
- Added Supplement Table 56 and a bounded main-text statement. The result is
  treated as source-fit robustness of a frozen encoded-latent readout, not as
  physical causality, action causality, planning success, or a universal
  world-model claim.

## Pass 43: Runtime Environment Wording Audit

- Rechecked the host environment on 2026-10-03. Host-level `nvidia-smi` was
  transiently inconsistent across checks, while the project PyTorch environment
  (`2.12.0+cu130`) consistently reports `torch.cuda.is_available() == False`,
  zero visible CUDA devices, and cannot initialize CUDA/NVML.
- Clarified the current reproducibility documentation and experiment notes:
  the newly added synthetic results remain CPU-only and carry no CUDA, GPU
  efficiency, PushT, or nonlinear-LeWorldModel performance claim. Historical
  revision entries are retained as historical records.

## Pass 42: Latent/ Predictor Equivariance Ablation Closure

- Added a three-seed CPU-only synthetic vector-world benchmark that separates
  latent equivariance, predictor equivariance, and their combined intervention
  under an exact additive action-conditioned symmetry.
- Audited and removed an initially confounding SIGReg term after measuring its
  scale dominance and premature early stopping; the canonical benchmark now
  contains only prediction loss plus the declared intervention and retains
  stable rank as a collapse diagnostic.
- Added the complete 21-record JSON, reader-facing Markdown table, protocol
  notes, visual audit, source snapshot, claim ledger entry, display-plan row,
  and supplement Table 23 with continuous equations (S7)--(S8).
- Repeated the canonical sweep exactly, verified finite values and transform
  conventions, and confirmed the CPU-only boundary because the host has no
  usable NVIDIA driver; no GPU or LeWM/PushT performance claim is promoted.

## Pass 41: Conditioned-Readout Display and Numbering Closure

- Repaired the 23-row display plan after the new conditioned-readout row was
  parsed incorrectly by an unquoted comma; the package plan and the canonical
  worktree audit now both report zero blocking findings.
- Added the conditioned-readout visual audit and corrected its rendered location
  to supplement page 7, Table 19. Corrected the neighboring intervention audit
  to reference adapter equation (S6), while the held-out score remains (S5).
- Recompiled the main and supplement entry points, confirmed 12 and 13 pages,
  and reran manuscript and reader-facing table audits with zero findings.
- Reconfirmed that the current host exposes an NVIDIA device but the project
  PyTorch process cannot initialize CUDA/NVML; the reported pilot remains
  CPU-only and no new GPU experiment is promoted.

## Pass 40: Abstract Terminology and Scientific Register

- Replaced implementation-facing references to released checkpoints, cross-fitted operators, shuffled pairs, and reconstructed splits with a scientific summary of the learned-representation evidence.
- Reorganized the abstract around the identification problem, exact operator-pair result, mechanism-separating controls, learned-system observations, and methodological consequence.
- Retained the distinction between the finite-dimensional theorem and measurements in LeWorldModel, I-JEPA, JEPA-WM, and DINO-WM without presenting experimental workflow as a contribution.
- Recompiled and inspected the first page. The main paper remains 12 pages, References begin on page 10, and the LaTeX log contains no unresolved references, overfull boxes, or fatal errors.

## Pass 39: Equation Numbering and Cross-Reference Audit

- Replaced every unnumbered main-paper display that defines an operator, diagnostic, theorem identity, proof block, or transfer readout with a labeled equation environment.
- Established the continuous main-paper sequence `(1)--(10)`, grouping the trace readouts as `(6a)--(6b)` and the coordinate-energy definition and expansion as `(10a)--(10b)`.
- Added explicit in-text references from the method, theorem interpretation, proof sketch, and transfer proposition to the corresponding equations.
- Assigned the standalone supplement its own `(S1)--(S4)` derivation sequence and added local cross-references.
- Recompiled and visually inspected the mathematical pages. Main and supplement remain 12 pages; the main-paper References begin on page 10, all references resolve, and both manuscript-language audits pass with zero findings.

## Pass 30: Oral-Level Argument Spine and Positive Scope Writing

- Reorganized the first-screen argument around four linked messages: identification failure, eigenspace-alignment mechanism, protocol-specific boundary evidence, and the measurement consequence.
- Rewrote the Abstract, Introduction, Results opening, learned-representation paragraphs, Discussion, and Conclusion so evidence is introduced through the scientific object it measures rather than through project-status or defensive disclaimers.
- Brought the completed JEPA-WM, DINO-WM, and matched PointMaze evidence into the main Results as a compact cross-family boundary check; retained the horizon-dependent PushT sign change as informative regime evidence.
- Replaced repeated `not a`, `does not establish`, and `unsupported` constructions with positive scope statements, while retaining one concentrated Discussion sentence that distinguishes the finite-dimensional theorem from learned-model generality.
- Recompiled the ICLR main PDF and reran manuscript morphology and reader-facing table audits.

## Pass 31: Abstract and Report-Morphology Polish

- Rewrote the abstract around the four-part scientific arc: spectrum-only identification failure, eigenspace-alignment mechanism, cross-protocol learned measurements, and the operator-pair reporting rule.
- Removed duplicated I-JEPA result prose and converted remaining `anchor`, `artifact`, and status-like phrasing in the manuscript and supplement into formal descriptions of measurements, stored records, and scope.
- Reworded supplement captions and temporal sections so tables report estimands and outcomes rather than validation workflow or evidence routing.
- Recompiled main and supplement PDFs; manuscript morphology and reader-facing table audits remain clean.

## Pass 1: Argument Spine

- Reframed the manuscript around the expectation-changing update: eigenvalues
  alone do not identify transfer geometry.
- Removed controller language from the thesis and kept empirical learned-latent
  claims outside the theory paper.
- Promoted the single-plane commutator law to the paper's central theorem.

## Pass 2: Evidence and Claim Closure

- Added the stronger null package as the main falsifier display.
- Added the spectrum-vs-alignment decoupling table to separate alignment
  variation from spectrum-shape variation.
- Updated the claim ledger so each claim has a rival, boundary, display, and
  status.

## Pass 3: Human-Paper and Visual Polish

- Downloaded and applied the official ICLR 2026 style and bibliography files.
- Rebuilt the manuscript as an anonymous ICLR-format PDF.
- Repaired float order so both Results tables appear before Discussion.
- Hid link borders and checked rendered pages at normal reading scale.

- Pass 36: Added the three-seed learned pixel-to-latent intervention to the
  supplementary evidence package. The new table reports held-out one-step and
  random-sequence rollout error, ordered action composition, global and
  conditional commutators, and stable rank for an encoder trained jointly with
  the predictor. The text keeps this as a synthetic boundary experiment and
  explicitly excludes nonlinear LeWorldModel, PushT, planning, controllability,
  and GPU claims. The source, 21-record JSON, Markdown summary, notes, claim
  ledger, display plan, and crosswalk were synchronized; the supplement still
  requires recompilation and rendered-page inspection before the package hash
  is refreshed.

## Pass 35: ICLR 2027 Official Style Migration

- Downloaded the official ICLR 2027 style archive specified by the ICLR 2027
  Author Guidelines and added `iclr2027_conference.sty`, `iclr2027_conference.bst`,
  and the accompanying template dependencies to the collaboration package.
- Migrated both `main.tex` and `supplement.tex` from the local ICLR 2026 style
  to `iclr2027_conference`; the bibliography style now matches the same release.
- Removed the manual appendix-only header override. The official style now
  supplies the审稿态页眉 `Under review as a conference paper at ICLR 2027`.
- Recompiled both entry points independently in the source tree and the
  Overleaf package: main 11 pages, supplement 12 pages, with no LaTeX errors
  or overfull boxes.

## Pass 36: Alignment Intervention Boundary and Delivery Audit

- Added a three-seed learned synthetic intervention comparing no alignment,
  global pooled alignment, and action-conditional alignment under heterogeneous
  action-local geometry.
- Added a matched frozen PushT latent-adapter intervention with a deterministic
  100,000-row split and paired one-step evaluation.
- Integrated both bounded results into the supplementary section “Alignment
  Interventions and Prediction Boundaries”; the adapter equation is numbered
  within the supplement and the text explicitly excludes end-to-end, rollout,
  online-control, and planning claims.
- Synchronized the collaboration package README, reproducibility README,
  result directory, source snapshots, claim ledger, display plan, and hashes.
- Recompiled and re-audited the 12-page main paper and 13-page supplement;
  stale package metadata from the previous 12/14-page state was removed.

## Pass 38: Formal Introduction Reconstruction

- Rebuilt the Introduction as a 656-word scientific argument covering problem
  importance, the hidden spectral-sufficiency assumption, the operator-pair
  object, the exact non-identifiability theorem, mechanism-identifying controls,
  learned-system applicability, and three explicit contributions.
- Repositioned Figure 1 after the operator-pair formulation and compacted the
  contribution list so that the figure and all three contributions render as
  coherent units rather than splitting unfinished prose across pages.
- Preserved the theorem/measurement boundary and the symmetry-agnostic,
  complementary-to-equivariant-JEPA positioning established elsewhere.
- Removed one obsolete uncited bibliography entry. Citation parity closes at
  37 cited keys and 37 bibliography entries.
- Recompiled the main paper to 12 pages. The scientific main text remains
  within nine pages and References begin on page 10.

## Pass 37: Formal Abstract Reconstruction

- Replaced the compact result inventory with a 311-word, single-paragraph formal abstract
  organized around the identification problem, operator-pair formulation,
  closed-form commutator theorem, decisive controls, learned-system evidence,
  and measurement consequence.
- Included the exact single-plane commutator identity while synthesizing the
  isospectral, frozen-representation, and held-out temporal evidence in
  argument form rather than as a list of run-level numbers.
- Preserved the claim boundary between the finite-dimensional theorem and the
  learned-model measurements: the theorem proves non-identifiability, while
  the empirical results establish that the proposed quantities are measurable
  beyond the analytical construction.
- Recompiled and visually inspected the first page. The main paper remains 12
  pages; References now begin on page 10.

## Pass 36: seq-JEPA Positioning and Symmetry-Agnostic Scope

- Added seq-JEPA to the main-paper Related Work with an explicit comparison:
  seq-JEPA induces invariant/equivariant representations through known
  transformations, actions, and architectural inductive bias, whereas this
  paper diagnoses covariance--predictor alignment without assuming a group
  action.
- Added the reader-facing scope statements "symmetry-agnostic diagnostic" and
  "complementary to equivariant JEPA" to Related Work and Discussion.
- Added the NeurIPS 2025 seq-JEPA bibliography entry and rebuilt the main PDF.
- Main-paper citation parity remains closed with no missing cited keys; the
  new main PDF is 12 pages, with References beginning on page 9.

## Pass 4: Full Paper Expansion

- Expanded the manuscript from a 4-page skeleton into a 12-page main-plus-appendix theory paper.
- Added the problem-setting equations, normalized commutator estimand, top-band transfer functional, theorem, under-identification proposition, and closed-form derivation.
- Added a reader-facing comparison/ablation design table to separate rival-removal comparisons from factor-isolating ablations.
- Added appendix tables for the exact commutator curve, complete rotation sweep, metric-robustness ablation, anisotropy boundary, cross-dimension replication, encoder/predictor mechanism sweep, heat/oblique diagnostics, seed/noise robustness, empirical bridge, and claim ladder.
- Copied appendix figures for mechanism, robustness, cross-dimension, and empirical-bridge artifacts into the ICLR bundle.
- Replaced internal status labels in reader-facing tables with semantic manuscript-use language.
- Re-rendered pages with `pdftocairo` after `pdftoppm` produced blank late-page PNGs on this PDF.

## Pass 5: Standalone Supplement Expansion

- Added `supplement.tex` as an 11-page standalone ICLR-style supplementary document.
- Included full mathematical derivation, complete same-spectrum rotation sweep, null comparisons, factor-isolating ablations, anisotropy/dimension boundaries, mechanism sweep, robustness sweeps, earlier theory sweep, rich spectral diagnostics, demoted controllability diagnostic, empirical PushT context, blocked evidence, and supplement claim map.
- Compiled `supplement.pdf`, rendered supplement pages with `pdftocairo`, and inspected `supplement_rendered_cairo/contact_sheet.png`.
- Updated `main.tex` to point to the standalone supplement from the provenance appendix.

## Pass 6: ICLR Main-Page Compression

- Removed the in-main appendix and preserved the full evidence package in the standalone 11-page supplement.
- Kept the main paper dense rather than skeletal: theorem, proof sketch, residual-growth formula, proposition, corollary, main figure, null table, ablation table, and boundary/robustness summary all remain in main text.
- Recompiled `main.pdf` to 8 pages including references at this pass, within the formal ICLR main-paper page budget.

## Pass 7: Formal ICLR Figure/Table and Terminology Audit

- Inspected ICLR proceedings examples and recorded the calibration in `domain_table_calibration.md`.
- Added `terminology_calibration.md` to replace coined title-level language with accepted vocabulary: commutator, commuting operators, simultaneous diagonalization, eigenspace alignment, principal angles, spectral projectors, and isospectral family.
- Changed the title to "Same Spectrum Is Not Enough: Commutators Reveal Eigenspace Alignment in Spectral Transfer".
- Regenerated the main figure from `run_operator_alignment_sweep.py` with larger fonts and clearer panel labels.
- Rewrote main and supplement table headers so they name scientific variables rather than internal audit roles.
- Moved old appendix display-plan entries from `main.pdf` pages 7--12 to `supplement.pdf` pages 2--11.
- Added `figure_table_audit_report.md` as the formal display-audit record.

## Pass 8: Formal Paper Organization

- Reorganized the main manuscript into a standard research-paper structure: Introduction, Related Work, Method, Theoretical Analysis, Experiments, Results and Analysis, Discussion and Limitations, Conclusion, Supplementary Material Roadmap, and Reproducibility Statement.
- Expanded the Method section into reader-facing components: finite-dimensional transfer model, same-spectrum construction, eigenspace-alignment measurements, and transfer readouts.
- Added the spectral-projector/eigenspace-overlap measurement as standard vocabulary alongside the commutator, so the paper does not rely on a coined concept.
- Removed internal workflow comments and replaced internal phrases such as proof duties, claim ladder, claim ledger, artifact dump, control paper, and random control with formal scientific language.
- Recompiled `main.pdf` to 8 pages at this pass and `supplement.pdf` to 11 pages; both static table audits and the oral paper writing contract pass after this pass.

## Pass 9: Venue and Field Section Calibration

- Checked ICLR-style proceedings papers and same-field spectral-analysis papers before finalizing section names.
- Recorded the structure audit in `section_structure_calibration.md`.
- Replaced `Methodology` with `Method`.
- Split `Experimental Validation` into `Experiments` and `Results and Analysis`, matching common ICLR organization where setup and interpretation are separated.
- Merged `Discussion` and `Limitations` into `Discussion and Limitations`, a recognizable academic structure.
- Kept `Theoretical Analysis` because same-field spectral-bias theory papers commonly include a theory/main-results section between setup and experiments.
- Verified source-level headings and rendered-PDF headings after recompilation; `main.pdf` remained 8 pages before the later Related Work expansion.

## Pass 10: Related Work Densification

- Expanded Related Work from a compact two-paragraph positioning note into three formal functions: spectral bias and operator spectra, matrix-analysis vocabulary, and commutators or basis-sensitive transfer.
- Added standard matrix-analysis and matrix-perturbation references so the paper uses recognized field vocabulary rather than coined terminology.
- Recompiled `main.pdf`; the paper is now 9 pages including references, with the scientific main text still within the formal ICLR main-paper page budget.

## Pass 11: Scientific Punchline Revision

- Rewrote the Abstract around the isospectral counterexample: identical predictor eigenvalues can still yield different transfer geometry.
- Sharpened the Introduction from a generic spectral-summary motivation into a field-level question about whether eigenvalue compression remains valid for latent transfer.
- Reframed the main Results opening as a counterexample to spectrum-only transfer rather than a descriptive report of the figure.
- Kept the learned-LeWM and online-planning claims outside the theorem while making the measurement consequence more explicit.

## Pass 12: Transfer Functional Theory Link

- Added Proposition 2, a closed-form horizon-one coordinate-energy expression showing that a transfer readout can vary under fixed predictor eigenvalues.
- Mirrored the derivation in the supplement so the main-paper proposition has an appendix-level calculation.
- Strengthened the bridge from the commutator identity to the measured top-band transfer readout without broadening the learned-latent claim.

## Pass 13: Importance-Risk Revision

- Added an Introduction paragraph explaining why the finite-dimensional same-spectrum construction matters as an identification counterexample, not as a toy benchmark result.
- Rewrote the Discussion around the audit requirement: eigenvalue decay is incomplete for transfer unless paired with commutator, principal-angle, or spectral-projector overlap statistics.
- Replaced defensive language about the separate empirical paper with a sharper distinction between the theory paper's measurement principle and the empirical paper's learned-latent generality question.
- Removed self-weakening toy-model wording from the main-paper importance paragraph.

## Pass 14: Identification-Language Revision

- Replaced the remaining paper-facing `certificate failure` framing with standard identification language: eigenvalues are not sufficient statistics for transfer under the isospectral construction.
- Rewrote the Abstract, Introduction, Results opening, and Discussion so the practical consequence is false equivalence under same-spectrum reporting, not a generic claim that spectra are useless.
- Converted the Related Work world-model boundary from defensive `not a new planner` wording into a positive measurement-question contrast.
- Recompiled `main.pdf`; citation parity now closes at 37 cited keys and 37 bibliography entries after removing the obsolete scikit-learn smoke-test citation, with no missing or uncited keys.

## Pass 15: Introduction Contribution-List Removal

- Removed the generic `The paper makes four contributions` bullet list from the Introduction.
- Replaced it with prose that links the experimental comparisons to the theorem's prediction and then states the field-level consequence: spectral-transfer analysis should treat transfer as an operator-pair measurement, not as a property of a predictor spectrum alone.
- Replaced the nearby supplement reference wording `retained in the supplement` with neutral scholarly wording, and removed remaining reader-facing `audit`/`display plan` wording from the main manuscript.
- Strengthened the Discussion and Conclusion so the value claim is not a slogan: the paper changes what counts as evidence for spectral transfer by requiring either a shared-eigenbasis assumption or an observed alignment measurement.
- Recompiled `main.pdf`; the paper remains 10 pages including references.
- Re-ran the reader-facing table audit and the manuscript-not-report audit on `main.tex` and `supplement.tex`; both pass with zero blocking and zero recommended findings.

## Pass 16: Eight-Page Main-Text Fill

- Extended Discussion with a non-redundant interpretation paragraph separating predictor spectra, latent covariance spectra, and covariance-predictor eigenspace relations.
- Extended the Conclusion with a concrete empirical follow-up paragraph: learned-world-model studies should test whether predictor eigenspaces align with task-relevant covariance directions across horizons and checkpoints.
- Recompiled `main.pdf`; the main scientific text now fills page 8 and `References` begins on page 9.
- Rechecked the static table and manuscript-not-report audits; both remain clean.

## Pass 17: LeWorldModel-Facing Abstract Polish

- Rewrote the abstract opening so the research question visibly starts from LeWorldModel spectral transfer rather than from a generic matrix-analysis problem.
- Recast the core contribution as an operator-pair statement: predictor eigenvalues do not determine covariance-relative latent transfer without an eigenspace-alignment assumption or measurement.
- Preserved the evidence boundary by separating the finite-dimensional isospectral theorem from frozen LeWorldModel PushT latent measurements.

## Pass 18: Academic Abstract Revision

- Rewrote the abstract in a more formal conference-paper register: problem context, under-identification statement, finite-dimensional construction, closed-form law, experimental separation, learned-representation comparison, and methodological consequence.
- Shortened the prose enough to keep the Reproducibility Statement on page 8 and References beginning on page 9.
- Replaced over-strong abstract wording with scoped language for the fixed-spectrum construction.
- Recompiled `main.pdf`; the reader-facing table audit and manuscript-not-report audit both pass with zero findings, and the LaTeX log has no citation, undefined-reference, fatal-error, or overfull warnings.

## Pass 19: Held-Out JEPA Operator Boundary

- Added `experiments/theory_spectral_transfer/run_jepa_operator_anchor.py`, a reproducible low-rank ridge view-operator fit in frozen I-JEPA feature space with held-out prediction, covariance--operator commutator, and spectrum-preserving random-basis nulls.
- Ran the CPU stress test on 20 deterministic scikit-learn-digits pairs (14/6 split, rank 8): held-out $R^2=0.91275$, observed commutator $0.09040$, random-basis mean $0.12580$, and predicted-displacement top-4 lift $0.44826$ versus pair-shuffled $1.53074$.
- Kept the negative operator-fit result in the supplement and added it to the claim/display ledgers as an estimator-identification boundary; the main text now explicitly separates positive view-transfer structure from a stronger learned-operator claim.
- Added a regression test for spectrum preservation under orthogonal conjugation and documented the command and evidence boundary in the JEPA theory README.

## Pass 20: JEPA Rank-Sweep Evidence Closure

- Extended the frozen I-JEPA operator-fit stress test from one small digits run to five held-out fits: CIFAR-10 at ranks 8 and 16, and scikit-learn-digits at ranks 8, 16, and 32.
- Across the five fits, held-out $R^2$ is $0.97141$--$0.99840$, while the observed covariance--operator commutator is below the spectrum-preserving random-basis null mean in every fit (differences $-0.00417$ to $-0.09323$); predicted-displacement lift remains below the pair-shuffled contrast.
- Added the rank-sweep summary artifacts, synchronized the main text, supplement Table 19, claim ledger, display plan, and oral-readiness metadata, and retained the result as a negative estimator-identification boundary rather than a JEPA-dynamics claim.

## Pass 21: JEPA-WM Temporal Gate Audit

- Added `experiments/theory_spectral_transfer/audit_jepa_wm_temporal_gate.py` to make the independent temporal/action-conditioned promotion gate reproducible.
- Verified a concrete Push-T JEPA-WM checkpoint (epoch 50; SHA-256 recorded in the gate artifact) and recorded the missing trajectory protocol separately.
- The current state is `checkpoint_ready_data_blocked`: the local Hugging Face trajectory request requires access approval, and a residual download directory is not counted as data. No multi-step learned-dynamics claim is admitted until the manifest, split, and held-out metrics exist.

## Pass 22: Public DINO-WM Trajectory Closure

- Followed the public GitHub release pointer to the DINO-WM OSF mirror and
  downloaded `datasets/pusht_noise.zip` (2,785,304,515 bytes).
- Verified the official SHA-256
  `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08` and
  extracted a concrete train/validation trajectory bundle (18,685 train and
  21 validation episodes, with states and action tensors).
- Re-ran the temporal gate audit: the data protocol is now closed. The gate is
  narrowed to the JEPA-WM predictor/API path and held-out multi-step metrics;
  no action-conditioned temporal claim is promoted before those metrics run.

## Pass 23: Claim-Bounded JEPA-WM Temporal Diagnostic

- Added a reproducible CPU evaluation over all 21 held-out PushT episodes using
  the epoch-50 checkpoint and a two-frame context.
- After correcting the wrapper's historical action-token alignment with zero
  normalized-mean padding, the rollout runs at h=1, 2, and 4. Latent MSE is
  0.07257, 0.06465, and 0.09167; commutators are 1.17128, 1.59889, and
  2.09273, compared with spectrum-preserving null means 1.39180, 1.40656, and
  1.41442.
- These are diagnostic temporal measurements for one released checkpoint and
  protocol, not a universal learned-JEPA or learned-world-model theorem.

## Pass 24: Context-Two Interface Closure

- The previous `512 vs 256` AdaLN failure was traced to a missing action token
  for historical context frames rather than to data, checkpoint, or predictor
  weights.
- Patched the wrapper to align action and visual time dimensions by padding
  missing history with the normalized mean action; context-two multimodal
  rollout now completes on the public validation trajectories.
- Recompiled the 12-page main paper and 8-page supplement. Manuscript-report
  and reader-facing-table audits both pass with zero blocking findings.

## Pass 25: Independent DINO-WM Family Replay

- Downloaded and verified the public DINO-WM PointMaze archive and its
  epoch-10 pretrained checkpoint from the same public release family.
- Added a reproducible PointMaze evaluator using three-frame context,
  five-action blocks, and horizons 1, 2, and 4. On a deterministic tail replay
  of 50 episodes (IDs 1800--1849), latent MSE is 0.00203, 0.00543, and 0.00780;
  commutators are 4.99992, 5.85025, and 9.10460 versus spectrum-preserving null
  means 1.45657, 1.42159, and 1.47253.
- The public checkpoint does not include its original episode-membership split;
  the result is therefore recorded as external replay with reconstructed
  holdout, not certified train/test independence.
- Added the result to the supplement as Table 21, keeping it separate from the
  theorem and from task-level or universal learned-world-model claims.

## Pass 26: Code-Reconstructed DINO-WM Split Closure

- Inspected the released DINO-WM loader and PointMaze config: `split_ratio=0.9`
  and episode-level `torch.randperm` with hard-coded `random_seed=42` are applied
  before temporal slicing.
- Reconstructed the exact 1,800/200 episode membership from the public dataset
  ordering and emitted an auditable split manifest with the algorithm, seed,
  IDs, provenance, and SHA-256 hashes of the loader/config source files. The
  archive still contains no serialized split file.
- Re-ran the DINO-WM temporal diagnostic on all 200 reconstructed validation
  episodes with the official full-archive normalization. Latent MSE is 0.00201,
  0.00491, and 0.01121 at horizons 1, 2, and 4; commutators are 37.89996,
  59.00962, and 96.48006 versus null means 1.46722, 1.41294, and 1.42394.
- Upgraded the cross-family evidence from a 50-episode tail replay to a
  code-reconstructed split replication, while preserving the non-universal and
  non-manifest-certified claim boundary.

## Pass 27: Second DINO-WM Checkpoint with Explicit PushT Validation

- Evaluated the released DINO-WM PushT checkpoint (epoch 2) on all 21 episodes
  in the public validation directory, with the explicit 18,685/21 train/val
  split, released normalization constants, and the same h=1/2/4 readouts.
- The checkpoint SHA-256 is `e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5`; the public PushT archive SHA-256 is `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08`.
- Latent MSE is 0.01672, 0.02316, and 0.06510; commutators are 0.62873,
  0.72850, and 2.08349 versus null means 1.38166, 1.38331, and 1.43895.
- The commutator crosses the null only at h=4, preserving the boundary-sensitive
  interpretation rather than inventing a monotone learned-dynamics law.
- This closes the second-checkpoint promotion request with explicit split
  provenance and keeps the remaining gap limited to a serialized PointMaze
  membership file or a third independently released checkpoint.

## Pass 28: Matched-Environment JEPA-WM PointMaze Cross-Check

- Downloaded and SHA-256 verified the official JEPA-WM PointMaze checkpoint
  (`a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc`).
- Evaluated all 200 episodes from the seed-42 PointMaze validation membership
  using the official full-archive normalization and two-frame context. Full-rank
  commutators are 60.48199, 43.44444, and 37.19464 versus null means 1.45494,
  1.57804, and 1.53736 at horizons 1, 2, and 4.
- Added rank-truncated fits (ranks 8, 16, 32, 64, 128); every tested rank and
  horizon remains above its spectrum-preserving null. This closes the sample
  coverage gate while retaining the conditioning caveat and theorem boundary.

## Pass 29: Cross-Fitted Temporal Operator Controls

- Added `analyze_temporal_operator_controls.py`, a leakage-controlled post-hoc
  analysis that fits operators on training folds and scores held-out episodes.
- Re-ran JEPA-WM and DINO-WM PointMaze evaluators with per-episode latent
  vectors saved alongside the original metrics, preserving the released
  checkpoint and code-reconstructed seed-42 split.
- Added paired, pair-shuffled, and spectrum-preserving random-basis controls
  across ranks 8/16/32/64 and ridge values $10^{-4}$--$10^{-1}$.
- At the primary setting (rank 16, ridge $10^{-2}$), paired held-out $R^2$ is
  0.082/0.183/0.025 for JEPA-WM and 0.935/0.808/0.554 for DINO-WM at
  horizons 1/2/4; pair-shuffled controls are negative at every horizon.
- Added a compact supplement table and a reader-facing boundary statement:
  full-rank commutators remain descriptive, while cross-fitted paired
  predictability is the claim-safe learned-checkpoint evidence.

## Pass 30: Temporal Interventions and Ablation Integration

- Added a main-text summary of within-episode action-order interventions and
  reconstructed PointMaze split robustness, keeping both results checkpoint-
  and protocol-specific.
- Added Supplement Tables 26--27 and the final estimator-ablation section for
  action-order controls, split-membership ranges, h=8, context length, operator
  direction, and nested sample-size diagnostics.
- Extended the claim ledger to C15--C17 and the display plan to T19--T21 with
  source-linked artifacts and rendered-page paths.
- Rebuilt the 12-page main paper and 11-page supplement; manuscript, table,
  display-link, and LaTeX warning audits pass with zero blocking findings.

## Pass 31: Cross-Environment Comparison and Ablation Closure

- Added a main-text future-only action figure spanning PushT, PointMaze, and
  Wall. The intervention preserves the historical action prefix and changes
  only the predicted-action suffix.
- Added a supplementary section with all 18 DINO-WM future-only cells, paired
  confidence intervals and Holm-adjusted tests, three reconstructed Wall
  memberships, distribution-preserving episode derangements, and the complete
  rank/ridge stability matrix.
- Retained the JEPA-WM Wall negative and mixed cells as an explicit checkpoint-
  specific boundary instead of promoting a universal action-sensitivity claim.
- Added compact source summaries, scripts, and SHA-256 provenance under
  `reproducibility/results/temporal_world_models/hardened_cross_environment/`
  and `reproducibility/code/hardened_temporal/`.
- Rebuilt the 12-page main paper and 14-page supplement. Manuscript morphology,
  reader-facing table, display-artifact, LaTeX reference, and rendered-page
  checks pass with zero blocking findings; no overfull boxes were introduced.

## Pass 35: ICLR 2027 Official Style Migration

- Downloaded the official ICLR 2027 style archive specified by the ICLR 2027
  Author Guidelines and added `iclr2027_conference.sty`, `iclr2027_conference.bst`,
  and the accompanying template dependencies to the collaboration package.
- Migrated both `main.tex` and `supplement.tex` from the local ICLR 2026 style
  to `iclr2027_conference`; the bibliography style now matches the same release.
- Removed the manual appendix-only header override. The official style now
  supplies the review-mode header `Under review as a conference paper at ICLR 2027`.
- Recompiled both entry points independently in the source tree and the
  Overleaf package: main 11 pages, supplement 12 pages, with no LaTeX errors
  or overfull boxes.

## Pass 36: Functional equivariance follow-up and package synchronization

- Added the controlled functional equivariance benchmark covering action-scale
  OOD, initial-state shift, combined OOD, h=8 rollout, and a candidate-set
  latent-planning proxy; the canonical artifact contains 48 per-seed records
  and 16 aggregate summaries.
- Integrated Supplement Table 24 and its reader-facing boundary: geometry can
  change OOD and planning-proxy behavior, but the effect is non-monotone and
  coexists with prediction-function trade-offs; the proxy is not online task
  success or a causal-control result.
- Synchronized the H3 result bundle, rendered page, functional source, latest
  refactored H4 source, and regularizers into the reproducibility package.
- Confirmed the functional benchmark remains CPU-only and synthetic-only;
  this host has no usable NVIDIA driver, so no CUDA or GPU-efficiency claim is
  made.

## Pass 37: Alignment functional consequences and audit closure

- Added the learned synthetic alignment functional follow-up with 33 per-seed
  records and 11 aggregate mode/weight summaries. The protocol evaluates random
  action-sequence rollout through $h=8$, ordered action composition, and a
  candidate-set planning proxy with an independent true target trajectory.
- Corrected the reader-facing display so Table 34 reports composition error,
  pooled/global commutator, and conditional commutator separately. The caption
  now states that table entries are means and that per-seed standard deviations
  are stored in the JSON artifact.
- Synchronized the source script, standalone package imports, result bundle,
  crosswalk, display plan, claim ledger, and CPU-only/no-NVIDIA boundary.

## Pass 38: SIGReg--alignment phase and isotropy-confound audit

- Added a fully crossed CPU synthetic phase experiment over SIGReg weights
  $0/0.01/0.1$, no/global/conditional alignment, alignment weights
  $0.01/0.1/1$, and three matched seeds, yielding 63 per-seed records and
  21 aggregate conditions.
- Added stable-rank diagnostics and matched no-alignment baselines to separate
  covariance flattening from alignment-sensitive geometry. The phase result
  shows that lower commutators can coexist with higher one-step, $h=8$, and
  composition errors; it is retained as a boundary result rather than a
  monotone improvement claim.
- Added Supplement Table 35, the phase figure, source snapshot, JSON artifact,
  crosswalk, display/provenance entries, and an explicit CPU-only/no-NVIDIA
  boundary. Same-seed exact-repeat, matrix closure, finite-value, and visual
  audits pass before manuscript compilation.

## Pass 39: Exact action-sequence composition closure and numbering audit

- Added the exact action-sequence composition audit for the learned synthetic
  action-conditioned benchmark: all binary sequences at $h=2,4,8$ (4, 16, and
  256 products), 33 per-seed records, matched ID/OOD rollout values, and the
  h=8 composition--rollout association diagnostic.
- Synchronized the standalone source, result JSON/Markdown, scatter plot,
  protocol notes, visual audit, rendered supplement pages, and provenance
  entries C26/T32/T33. The package now includes the new source and result
  bundle rather than only the manuscript prose.
- Recompiled the supplement to 20 pages and audited the actual labels: the
  new displays are Table 35, Table 36, and Supplementary Figure 2; the
  existing SIGReg displays consequently become Table 37 and Figure 3.
  Equation labels remain S1--S8 with no unresolved references.
- Confirmed four focused tests pass, all 33 records are finite and exact-repeat
  stable, the claim ledger parses with uniform columns, and the current host
  remains CPU-only (`torch.cuda.is_available() == False`, device count 0)
  because CUDA/NVML cannot initialize.

## Pass 40: External temporal directionality and adapter robustness

- Added the frozen external temporal-adapter cross-normalization audit: three
  episode split seeds, three adapter seeds, global/conditional modes, and
  weights `0.01/0.1/1.0` under public, validation z-score, and raw/100 action
  normalizations. The $h=8$ effect is non-monotone and normalization-sensitive.
- Added the stricter future-vs-past and within-episode deranged-action audit for
  the transferred dynamics basis. Across 36 cells, zero satisfy the strict
  dual-CI directionality criterion; the result is retained as a negative claim
  boundary, not upgraded to a causal or physical-dynamics claim.
- Added focused tests for normalization-summary uniqueness/protocol coverage,
  lag-feature construction, derangement, and clustered bootstrap. The combined
  targeted regression suite passes (`24 passed`).
- Integrated Supplement Tables 22--23 and equation (S7), recompiled the main
  paper (13 pages) and supplement (28 pages), checked rendered pages 10--11,
  passed manuscript and reader-facing-table audits, and refreshed the SHA-256
  manifest.
- Final full regression completed with `257 passed, 6 skipped` in 440.76s.

## Pass 41: Cross-horizon latent readout transfer

- Added the held-out cross-horizon audit for source-fitted dynamics, residual,
  covariance, and random latent bases. Readouts are fit at one horizon and
  evaluated at another under leave-one-episode-out cross-fitting; the result
  stores all 9 source--target horizon cells, six ranks, four target families,
  and 648 paired basis contrasts.
- The predeclared $h=4\to8$ dynamics-band transfer is
  $+0.364\ [ +0.156,+0.602 ]$ incremental $R^2$, with a paired
  dynamics-minus-random contrast of $+0.553\ [ +0.351,+0.773 ]$. The residual
  top-8 readout is selectively positive for the residual-band target,
  $+0.177\ [ +0.108,+0.252 ]$.
- Integrated the result into the main manuscript, Supplement Section 15.8,
  formula (S18), Table 54, the figure/table crosswalk, claim ledger, and
  display plan. Rendered supplement pages 29--30 were visually inspected.
- Current-device verification: `nvidia-smi` and PyTorch both expose one RTX
  5060 Ti with CUDA; the canonical replay artifact records `device=cuda`.
  This supersedes earlier historical CPU-only notes and does not change the
  claim boundary of the frozen predictive diagnostic.
- The full regression completed with `265 passed, 6 skipped`; manuscript,
  reader-facing-table, and SHA-256 audits passed.

## Pass 42: Richer observed-action history control

- Added a matched control audit comparing the legacy six-dimensional action
  summary with the complete sequence of per-block two-dimensional action means,
  left-aligned and zero-padded to a common 16-dimensional h=8 feature vector.
- Completed six canonical cells and twelve independent source-fit cells across
  public, validation-split z-score, and raw/100 normalization, with two/eight
  windows per episode, 21 held-out episodes, CUDA inference, leave-one-
  episode-out readouts, and 2,000 episode-cluster bootstrap replicates.
- The richer-history dynamics-minus-random lower-bound minimum is `+0.0278`,
  and the dynamics-minus-covariance minimum is `+0.0499`; the h=4-to-8
  residual-minus-dynamics lower bound is positive in all 18 cells. The result
  is recorded as robustness of frozen latent predictive specificity beyond an
  observed action history, not a causal action-effect or planning result.
- Added the reader-facing supplementary table and main-text paragraph, the
  aggregate JSON/Markdown report, source script, README entry, and focused
  feature/artifact tests. The richer control is retained alongside compact6
  because its higher dimension produces larger regularized condition numbers.

## Pass 43: Contact-rich predictor repair and negative-result diagnosis

- Added a teacher-forced multi-step repair runner for the PushT action-conditioned
  predictor. The visual encoder/projector remain frozen; only the action encoder,
  predictor, and prediction projection are updated. Test-time evaluation is fully
  autoregressive and uses a disjoint contact split.
- Corrected the intervention accounting so the reported `+a` endpoint error uses
  only the positive arm, while `+a/-a` physical separation remains paired by case.
  Two independent seeds use 48 training cases and 16 held-out cases each.
- The paired calibrated gain is positive at point-estimate level for both seeds at
  h=2/4/8; both h=4 intervals are above zero, while h=2 and h=8 intervals remain
  compatible with zero. The original readout worsens on repaired latents, exposing
  latent-distribution shift as a separate calibration failure mode.
- Added the main-text paragraph, Supplement Table 66, seed-level machine-readable
  results, targeted tests (`27 passed` across the contact/counterfactual suite),
  PDF/rendered-table checks, and refreshed package hashes.

## Pass 44: Compositional repair, candidate-ranking audit, and closed-loop boundary

- Added compositional contact-mode data with five action-sequence arms per case,
  including piecewise and direction-changing actions, plus a live repaired-
  predictor MPC runner and a matched candidate-ranking replay audit.
- On the disjoint 96/24 split, autoregressive h=8 latent endpoint MSE changed
  from `0.860` to `0.381`; matched candidate Spearman changed from `0.107` to
  `0.690`, and selected regret from `188.1` to `32.9`.
- The candidate oracle remained at task cost `169.6` with 64 candidates and
  `148.9` with 512 candidates. Live contact-bootstrap MPC retained zero true
  task successes, so the evidence separates predictor-ranking repair from
  unresolved action coverage and multi-stage task control.
- Added the main-text Discussion paragraph, Supplement Tables 73--74, scripts,
  regression tests, machine-readable artifacts, package synchronization, and
  refreshed SHA-256 checksums. The privileged-state arm is diagnostic only.

## Pass 45: Stable-contact bootstrap and staged candidate coverage

- Added a stable-contact bootstrap that continues after the first block motion,
  applies a braking phase, and records the settled-contact gate separately from
  the state-guided diagnostic. Added staged approach--brake--push candidates
  with angular contact variants and a simulator-in-the-loop CEM upper bound.
- On 16 held-out states at h=8, staged candidates increased best block progress
  from `49.217` for legacy guided candidates to `60.231` and reduced best task
  cost from `271.865` to `264.421`; all candidate sets and the simulator oracle
  had zero full-task successes. At h=32, block-pose success also remained zero
  on eight states, including the oracle bank.
- Stable-contact plus staged live MPC entered planning at contact rate `1.0`.
  The privileged repaired run had zero task successes but reduced endpoint pose
  error to `136.858` and final state distance to `176.457`; the matched visual
  run remained at zero success with endpoint error `422.242` and final distance
  `652.398`. These are bounded local-interface diagnostics, not planning-gain
  or causal-physical claims.
- Added the main-text paragraph, the stable-contact coverage supplement
  subsection, new runner/tests, machine-readable artifacts, and synchronized
  the reproducibility package. The downloaded PushT expert archive failed zstd
  integrity validation and was not used as evidence.
