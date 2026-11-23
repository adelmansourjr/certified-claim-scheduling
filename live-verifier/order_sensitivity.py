"""Exploratory CPU timing with alternative parent-level checking orders."""
import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import process_time_ns

import numpy as np

from verify import cpu_comparison, verify_lock

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("live_verifier", root / "study.py")
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock = verify_lock()
    data = root.parent / "data"
    for part, digest in study.DIGESTS.items():
        assert study.file_sha(data / f"{part}.jsonl") == digest, part
    train = study.load(data / "train.jsonl")
    dev, _ = study.avoid_overlap(study.load(data / "dev.jsonl"), train)
    test, _ = study.avoid_overlap(study.load(data / "test.jsonl"), train + dev)
    test = study.eligible(test)
    clusters = study.clustered_ids(test)
    idf, unknown = study.build_idf(train)
    verifier, router, _ = study.fit_model(study.eligible(train), study.eligible(dev), idf, unknown)
    block_order = tuple(lock["block_order"])
    alternatives = ("lexical", "original", "cost")
    methods = [(order, policy) for order in alternatives
               for policy in ("fused_short_circuit", "certified")]
    rng = np.random.default_rng(88193)
    rows = []
    for parent in test:
        times = {order: {"fused_short_circuit": [], "certified": []} for order in alternatives}
        verdicts = set()
        for _ in range(5):
            for i in rng.permutation(len(methods)):
                order, policy = methods[i]
                start = process_time_ns()
                verdict, _, _ = study.predict(parent, idf, unknown, verifier, router,
                                              policy, block_order, order)
                times[order][policy].append(process_time_ns() - start)
                verdicts.add(verdict)
        assert len(verdicts) == 1
        rows.append({"cluster": clusters[parent.key], "times": times})

    comparisons = {}
    for policy in alternatives:
        samples = [{"cluster": row["cluster"], "times": row["times"][policy]} for row in rows]
        groups, ratio, interval = cpu_comparison(samples, "times")
        totals = {method: sum(median(sample["times"][method]) for sample in samples)
                  for method in ("fused_short_circuit", "certified")}
        comparisons[policy] = {"sum_case_median_cpu_ns": totals, "relative_reduction": ratio,
                               "paired_95": interval.tolist(), "clusters": groups}
    result = {"parents": len(rows), "paired_repeats": 5, "seed": 88193,
              "all_methods_exact": True, "exploratory_orders": comparisons}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
