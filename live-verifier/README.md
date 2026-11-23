# Exact early exits in a trained WiCE verifier

A factuality checker can stop after finding an unsupported subclaim. Can it also stop **within** a subclaim, before computing every evidence feature? In this CPU experiment, exact score bounds skipped **333 of 546 alignment computations (61.0%)** and reduced aggregate verification CPU time by **7.60%** against a development-selected short-circuit verifier. All six policies produced the same verdicts on **310** held-out WiCE parent claims.

## Experiment

- **Question.** Does a certificate over the frozen verifier's uncomputed features save time beyond a strong baseline that already uses a trained lexical router, shares a document index, and stops at the first unsupported subclaim?
- **Inputs.** The checker reads the original evidence text and released subclaims. It retrieves up to five candidate lines from each supplied evidence document, using lexical weights fitted on training documents. Gold supporting-line annotations do not enter its features. The verifier and router fit on training data. Model and order choices were fixed on development data before this study evaluated its held-out predictions. The preceding scheduling study used the same official WiCE split.
- **Method.** The verifier's expensive features are pairs of descending top-three scores. A triple in [0,1] lies in the convex hull of 000, 100, 110 and 111. Its signed contribution therefore lies between the smallest and largest fitted-weight prefix sums. Add the bounds for remaining stages to the exact partial score. Stop when the entire interval lies on one side of the decision threshold. Compute every feature when uncertain.
- **Comparators.** Check all subclaims, stop at the first failure in original order, use a learned lexical order, choose a fast order on development data, or use ordinary independent feature bounds. The strongest preselected comparator and both certificate variants used the same lexical order and block order in the held-out study. The same frozen verifier supplies every decision.

[Protocol and proof](PROTOCOL.md) · [Development record](DEV_SUMMARY.json) · [Source and policy lock](LOCK.json)

## Results

| Policy | Evaluated subclaims | Alignment calls | Total CPU, ms |
| --- | ---: | ---: | ---: |
| Exhaustive | 830 | 830 | 302.789 |
| Original-order short-circuit | 636 | 636 | 249.900 |
| Lexical-router short-circuit | 546 | 546 | 231.176 |
| Development-selected strong short-circuit | 546 | 546 | 230.935 |
| Independent feature bounds | 546 | 500 | 228.920 |
| Ordered top-three certificates | 546 | **213** | **213.395** |

The primary timing rebuilt the document index and included routing. CPU totals sum medians of three interleaved, randomly ordered runs for each parent. The relative reduction was **7.60%**, with a **95% paired interval of 6.30–8.91%** over 283 connected groups. Individual parent times were higher with the certificate in 83 of 310 cases. The two other feature stages both still ran 546 times.

A fresh process with the locked configuration reproduced a **7.08%** reduction **[5.76%, 8.41%]** on the same cases. A separate exploratory run used five paired repeats to check alternative parent orders. It saved **7.30%** for lexical order, **7.64%** for original order and **7.82%** for cost-based order, each against its matching eager short-circuit comparator. Lexical order had the lowest overall CPU time among these eager baselines and among the certified variants. [Order sensitivity](results/order_sensitivity.json)

Prebuilding the document index saved **11.63%** [9.35%, 13.45%] in the primary run. For the 59 parent claims with evidence documents above the development-defined length cutoff, including indexing saved **5.30%** [3.31%, 7.48%]. These secondary results describe the same hardware and test corpus. Intervals describe paired case variation.

All policies achieved **66.45% raw accuracy** and **68.39% balanced accuracy** against the strict WiCE parent endpoint. The always-negative majority rule achieved **68.06% raw** and **50.00% balanced accuracy**. The checker made 78 false positives and 26 false negatives. Exact inference preserves those errors. The measured gain applies to CPU evidence matching for this classifier. LLM calls, GPU time and token use were outside the experiment.

[Primary results](results/summary.json) · [Per-case timing and verdicts](results/per_claim.jsonl) · [Timing replication](results/replication_summary.json)

## Reproduce

Python 3.12.14, NumPy 2.3.5, arm64 macOS 26.5.1, one CPU process and one numerical library thread. Peak RSS for the primary run was **172 MiB**. From the repository root:

```sh
python -m pip install -r requirements.txt
python fetch_inputs.py
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python live-verifier/test_study.py -v
python live-verifier/verify.py
python live-verifier/study.py --mode test --out live-replay
python live-verifier/verify.py --results live-replay
python live-verifier/order_sensitivity.py --out live-orders.json
```

The ledger audit independently checks source hashes, frozen policies, stage counts, prediction equality, confusion counts and paired intervals. Timing changes between repeated processes. Raw documents and annotated source sentences are fetched separately from the pinned release.

## Research connection

Ye, Santos-Rodriguez and Simpson study how to turn grounded claims into reading-comprehension checks. This experiment starts with WiCE's supplied subclaims and tests how much work a deterministic verifier can omit before deciding one of those checks. [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1468/) · [Authors' implementation](https://github.com/Haruhi07/Test-Taking)

Kamoi, Goyal, Rodriguez and Durrett supply the naturally occurring evidence and subclaim labels in WiCE. [EMNLP 2023 paper](https://aclanthology.org/2023.emnlp-main.470/) · [Dataset](https://github.com/ryokamoi/wice)
