# Certified schedules for factual conjunctions

**Follow-up:** [Exact early exits in a trained WiCE verifier](live-verifier/) preserves all 310 held-out decisions while skipping 61% of expensive alignment calls and reducing CPU time by 7.6% versus an optimized matched verifier.

A factuality checker can decompose a claim and verify its parts. Must it inspect every part, and when is it better to check two at once? We evaluated exact early stopping and batch schedules on official WiCE subclaim annotations, using a proxy cost for contextual setup and checks.

In the prespecified scenario, the learned joint scheduler preserved all **310** annotation-derived decisions and spent **6.70% less** than the validation-selected fixed-width scheduler. A strong exact scheduler given independent marginals was almost equally good. The comparison does **not** establish a gain from the fitted dependence model or predict wall-clock speed for language models.

| Comparison at two checks per batch and setup cost 48 | Mean cost units | Mean batches | Mean checks |
|---|---:|---:|---:|
| Joint-prior exact schedule | 115.34 | 1.235 | 1.794 |
| Independent-prior exact schedule | 115.65 | 1.229 | 1.813 |
| Validation-selected fixed batch width, same cost/failure ordering | 123.62 | 1.145 | 2.190 |
| Read every annotation at the same maximum batch capacity | 156.89 | 1.529 | 2.677 |

Joint minus fixed width was −8.28 cost units, with 95% paired block bootstrap interval **[−10.93, −5.52]** over 283 connected evidence/article/context blocks. Joint minus independent exact was −0.31 **[−1.40, +0.94]**. The learned joint law changed the full planned schedule on 41 cases. The joint versus independent cost and predictive Brier improvements were inconclusive. Differences across subclaims or cost regimes do not count as additional independent trials.

## Why dependence could matter

The exact policy certifies a negative once it sees any failed check. It certifies a positive once every check succeeds. The choice of which check to perform can use an estimated prior without trusting that prior for the verdict.

We give two four-check laws with identical individual pass probabilities that demand different first batches. With two checks per batch, a unit per check and two units of setup cost, the law-specific minimum expected cost is 4.1. A strategy that knows only the shared marginals has minimum average cost 5 under an equal mixture over the laws. The **0.9-unit regret lower bound** also applies to randomized policies. [Derivation and assumptions](ANALYSIS.md)

The real-data test asks whether this effect survives noisy dependence estimates on a defined checking task. We fit lexical marginals on 1,260 original training claims and selected fixed widths on 309 development claims after overlap filtering. We excluded 47 of 358 test parents with a prior evidence, article or context key, then excluded the remaining singleton. The 310 eligible test parents contain 99 all-pass cases. Partially supported annotations fail the study's fully-supported requirement. A failing check need not contradict a subclaim.

## How context costs change the result

| Setup cost | Capacity | Joint DP | Independent DP | Selected fixed |
|---|---:|---:|---:|---:|
| 0 | 2 | 49.34 | 49.34 | 49.34 |
| 16 | 2 | 73.77 | 74.05 | 74.68 |
| **48** | **2** | **115.34** | **115.65** | **123.62** |
| 144 | 2 | 229.79 | 230.46 | 233.56 |
| Document word count | 2 | 1,744.77 | 1,742.60 | 1,742.85 |
| 48 | 4 | 114.61 | 114.42 | 123.62 |
| Document word count | 4 | 1,576.19 | 1,580.75 | 1,580.16 |

These are proxy costs, with descriptive sensitivity conditions around the prespecified comparison. Setting setup to zero represents reusable context. Charging the document word count represents a full reread. In the latter regime at capacity two, the fitted joint scheduler costs slightly more than both controls. The complete output also covers the 0, 16 and 144 scenarios at capacity four. Unlike a claim verification benchmark, this study supplies ground-truth check results and measures how to use them. No language model was run. The guarantee applies to a fixed conjunction and to check outputs that do not depend on execution order or batching.

## Reproduce

Python 3.12 and NumPy 2.3.5. CPU only. Local observed peak process RSS was about **133 MiB**, including data, fitting and replay. Raw WiCE annotations are downloaded separately from the pinned release and checksum-verified.

```sh
python -m pip install -r requirements.txt
python fetch_inputs.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m unittest -v
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python study.py --output replayed
python verify.py --results replayed
python verify.py
```

The committed run contains [summary statistics](results/summary.json) and all [per-claim schedules](results/per_claim.jsonl) across 10 cost settings. The [protocol](PROTOCOL.md) and the executable hashes were frozen before acquiring test annotations. The independent verifier recomputes realised costs and verdicts and enumerates every feasible partition to confirm both optimizers on all primary cases. Names of Wikipedia pages, source text and raw datasets are omitted from the published results.

## Sources

- Ye, Santos-Rodriguez and Simpson. *Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies*, ACL 2026. We extend the scheduling interface implied by their atomic fact checking. Their SLM training and language-model results are separate. [Paper](https://aclanthology.org/2026.acl-long.1468/) · [Authors' implementation](https://github.com/Haruhi07/Test-Taking)
- Kamoi, Goyal, Rodriguez and Durrett. *WiCE*, EMNLP 2023. WiCE supplies naturally occurring Wikipedia claims and fine-grained textual entailment annotations. [Paper](https://aclanthology.org/2023.emnlp-main.470/) · [Dataset](https://github.com/ryokamoi/wice)
- Daldal, Özlük, Selçuk, Shahmoradi and Ünlüyurt. *Sequential testing in batches*, 2017. Fixed batch setup costs and series-system tests are established. [Paper](https://doi.org/10.1007/s10479-016-2345-7) · [Author thesis](https://research.sabanciuniv.edu/id/eprint/34690/1/RebiDaldal_thesis.pdf)
- Ghuge, Gupta and Nagarajan. *A general framework for sequential batch-testing*. Broader approximation theory addresses when batched nonadaptive policies approach optimal policies. [Author manuscript](https://optimization-online.org/wp-content/uploads/2024/06/sequential_batch_testing.pdf)

Code is MIT licensed. Obtain WiCE under the upstream license. No license to Wikipedia citations or external page texts is transferred here.
