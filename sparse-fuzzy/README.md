# Exact fast fuzzy evidence matching

Does packing character-trigram postings accelerate a factuality verifier without changing its score? This follow-up kept the previously trained verifier fixed. It used **20.79% less end-to-end CPU** than the tuned, cached dense version on 310 WiCE parents, and **40.51% less** on 339 SciFact claim–cited-abstract pairs with the same frozen policies. All verdicts and all compared character scores agreed.

## Method

The prior verifier calculates weighted exact and fuzzy term coverage over a shortlist of evidence sentences. Its most expensive surviving character operation measures maximum trigram Jaccard similarity from a claim term to any term in a sentence. For a given sentence, assign bit positions to its distinct eligible terms. A trigram's posting mask has a one for every term containing it. For a query term, the bitwise union of its trigram postings finds **exactly** the terms with a nonempty overlap. Every omitted term has Jaccard zero. The score of each retained pair is computed exactly and reused within the same document. Zero-overlap elimination therefore preserves the maxima. The top-three features and signed margin remain equal to the original verifier's. Near the decision boundary the prior stage certificate already falls back to exact features.

The proof uses a standard inverted index and bitsets. The contribution is a resource-matched pipeline comparison on real evidence text, with the same model, routing order and stage-level certificate. Each method pays for document indexing, query preparation, fuzzy matching and compound verdict when required.

Compare with the original stage certificate, then with a stricter dense control that also caches term-pair overlaps per document and prunes by known length bounds. Development selects both the faster control and the fastest of three sparse representations. The prior WiCE study used the same 310 held-out parents. Reusing that corpus tests a different exact computation under the same scientific split, rather than replicating on unseen Wikipedia claims. Separately, apply the locked system on SciFact without retuning.

[WiCE protocol](PROTOCOL.md) · [Development](DEVELOPMENT.json) · [Frozen source and model](LOCK.json) · [SciFact transfer protocol](TRANSFER.md)

## Results

| Workload | Cases | Cached dense CPU, ms | Packed postings CPU, ms | Relative CPU reduction |
|---|---:|---:|---:|---:|
| WiCE compound claims | 310 | 266.546 | 211.135 | **20.79%** [18.37%, 23.20%] |
| SciFact claim–cited-abstract pairs | 339 | 112.476 | 66.909 | **40.51%** [38.84%, 42.07%] |

Timing adds per-case medians over five interleaved repeats. Intervals resample 283 WiCE or 247 SciFact connected case groups. These are paired intervals over benchmark requests on this CPU. Fresh processes on the same cases recorded reductions of **21.21%** [18.83%, 23.55%] on WiCE and **40.97%** [39.39%, 42.45%] on SciFact, with identical decision and work records. [WiCE repeat](results/timing-repeat.json) · [SciFact repeat](scifact/timing-repeat.json) The observed gains over the prior certificate were **26.65%** [24.55%, 28.71%] and **45.73%** [44.04%, 47.28%]. Relative to the packed index with the same parent-level stopping and every feature block evaluated, the stage certificate saved another **9.17%** on WiCE and **11.96%** on SciFact. Model weights, claim order and evidence are shared.

The six-feature vectors matched exactly for **830 WiCE subclaims** and all **339 SciFact pairs**, including subclaims omitted in actual inference. The separate all-subclaim work audit reduced exact Jaccard operations from **206,155 to 9,667** on WiCE, a **95.31%** reduction compared with the cached dense control. The index construction, fixed lexical and local blocks and independent alignment checks remain. The ratio of operations cannot be read as CPU savings.

WiCE accuracy stays at 66.45% raw and 68.39% balanced. The raw majority accuracy is 68.06%. SciFact uses the entire cited text and no rationale annotations or gold verdict in inference. The transfer timing does not establish quality on scientific verification. Observed CPU times include text preparation and first use of the per-document filter. Both workloads fit easily in memory. The WiCE primary process peaked at 181.7 MB RSS.

[WiCE summary](results/summary.json) · [WiCE ledger](results/per_claim.jsonl) · [SciFact summary](scifact/summary.json) · [SciFact ledger](scifact/per_claim.jsonl)

## Reproduce

Use Python 3.12, NumPy 2.3.5 and the original repository's pinned WiCE archive. The public result ledgers contain case IDs and measurements rather than evidence or claim text. Runtime figures can vary with machine and Python build.

```sh
python -m pip install -r requirements.txt
python fetch_inputs.py
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python sparse-fuzzy/verify.py
python sparse-fuzzy/study.py test --out /tmp/wice-packed-replay
curl -fLsS https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o /tmp/scifact-pinned.tar.gz
python sparse-fuzzy/transfer.py /tmp/scifact-pinned.tar.gz --out /tmp/scifact-packed-replay
```

[Live verifier](../live-verifier/) · [Ye, Santos-Rodriguez and Simpson, ACL 2026](https://aclanthology.org/2026.acl-long.1468/) · [WiCE](https://aclanthology.org/2023.emnlp-main.470/) · [SciFact](https://aclanthology.org/2020.emnlp-main.609/)
