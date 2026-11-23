# Batch schedules for checking factual conjunctions

## Question and prior studies

Ye, Santos-Rodriguez and Simpson (ACL 2026) decompose a claim into atomic facts and teach a small language model to apply a structured test-taking strategy. WiCE occurs in their LLM-AggreFact comparison. This study isolates the choice of which subclaims to check together and in what order. Human subclaim annotations serve as already-defined check outputs. No language model or new entailment annotator is used.

Daldal et al. study batch setup costs in sequential testing of a series system. The test here also has a conjunction, nonzero batch overhead and up to four concurrent checks. A two-state dependence model preserves the independent baseline's estimated marginal probabilities, creating a controlled comparison that changes only the joint prior. The dynamic program is a direct application of known sequential-test principles. The contributions here are the exact finite witness for the consequences of equal marginals, and an auditable WiCE test of whether accounting for dependence changes measured schedule cost.

## Data, exclusions and fitted signals

Use the authors' WiCE entailment retrieval subclaim split files pinned at ddeb6c183665e2a20c5f03c5aa07f03888b9870f. Group subclaims by their common original claim ID and require the same supplied evidence sentences. The operational decision is the conjunction of fully supported subclaim labels. Map WiCE's supported label to 1, and partially supported and not supported labels to 0. The compound label is computed from this mapping and differs in scope from native whole-claim classification. A 0 is a failure of the fully-supported rule, which may express insufficient evidence.

Fit logistic marginal scores on the official train split using claim length, evidence sentence count, number of subclaims, subclaim position, IDF-weighted lexical coverage of three best matching sentences and full evidence, and fraction of numeric tokens absent from the evidence text. Fit the scaler and IDF on train only. Use ridge-penalized training and clip predicted marginals to [.025,.975]. Train-only maximum likelihood on a grid of strengths selects the symmetric latent dependence. It raises or lowers each probability by delta times min(p, 1-p), with both latent states equally likely and conditionally independent. The held-out predictions do not depend on annotation content, evidence rationale indices or downstream observed check values, except for deciding to stop.

Exclude any dev parent claim whose document digest, normalized article title, or normalized claim context appears in train. Exclude any test parent whose corresponding key appears in the full original train or dev, whether or not a dev parent was excluded. Report exclusions, all test singletons (no possible saving) and any groups of more than eight (computational cap). Eligible parents have two to eight subclaims. Determine bootstrap blocks by joining parents with the same evidence digest, article title or context. Subclaims are not treated as independent sample units.

## Replay

Each check costs 20 plus its subclaim word count. Each executed batch also costs a contextual overhead of 0, 16, 48, 144 or its supplied evidence's word count. All arms pay the same schedule-dependent costs and have the same maximum batch capacity, 2 or 4. Costs are abstract units. The longest condition models full repeated document processing in word units. Context preprocessing and scheduling run in CPU memory and whole-program runtime includes feature extraction, model fit, development selection and test. Actual model tokens, hardware speed, backing annotations and generation costs are unmeasured.

Every stopping variant uses exact conjunction certificates. Return 0 only after observing a failing check, and 1 only after all checks pass. This keeps the same check-oracle decision under arbitrary dependence and model misspecification. A stronger or weaker prior can alter the schedule and its cost. The gold check outputs only enter replay.

Prespecify overhead 48 and capacity 2 for the main comparison. Benchmark:

1. Joint-prior minimum expected cost via exact subset DP.
2. Independent-prior minimum expected cost using the same learned marginals and exact DP.
3. Fixed-width schedule with subclaims ranked by cost divided by estimated failure probability. Select width separately in each cost regime and capacity on official development. Same ranking uses full context signals and equal information.
4. Always fill the available batch capacity, with the same ranking.
5. Check in natural order, one item at a time.
6. Fully evaluate all checks under the same capacity and overhead, even following a failure.

If check outputs are stable given the same subclaim and evidence, write q(S) for the prior probability all of S pass. A state S is only reached along the all-pass branch. An ordered partition therefore suffices for any deterministic sound policy. The expected downstream contribution weighted by the probability of reaching S satisfies F(S) = min_B [q(S) c(B) + F(S union B)], where feasible B is a nonempty subset of unresolved items satisfying the capacity. At the full set, F=0. Induction over remaining items gives the exact minimizer under the declared prior. It does not imply optimality under the empirical data-generating process. Exponential state space is practical only for short decompositions.

## Metrics and claims

Primary metric is the paired difference in mean proxy cost, joint DP minus strongest development-selected fixed-width method at capacity 2 and overhead 48. Exact zero oracle decision disagreement is required. An improvement claim requires a negative upper bound in the 4,000-replicate percentile bootstrap at this primary point. Report joint versus independent at the same point regardless. Claim that modeling dependence saved cost only if this comparison independently has a negative upper bound. Intervals resample connected blocks of parent claims with their within-block observations and use the pooled ratio of sums. The rest of the cost sweep is descriptive and intervals are not simultaneous. Always give both the number of oracle answers and the number of statistically resampled groups. Also report learned priors' Brier error for the all-pass outcome, batch rounds/checks, policy differences, and Python peak RSS and wall time. A cost win is conditional on trusted check outputs, chosen budgets, validation criterion and the benchmark. Honest negative and inconclusive results must remain visible.

Data and source files, including this protocol, are pinned and hashed. Dev mode cannot open the test file. Run the official test once source is frozen. Store every policy and outcome, case IDs and numerical features needed for auditing, excluding copyrighted text. Include an independent exhaustive test of the scheduler against every feasible ordered partition of small cases and a finite equal-marginals, different-optimal-schedule witness.
