"""Benchmark exact character scoring inside the established WiCE verifier."""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np

from matching import CharacterMatcher

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location("live_wice", PARENT/"live-verifier/study.py")
live = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live
spec.loader.exec_module(live)

INDEXES = ("sparse", "bitmask", "signature")
CONTROLS = ("prior", "dense")
METHODS = (*CONTROLS, *INDEXES, *(m+"_eager" for m in INDEXES))
SEED = 974321
SOURCES = ("PROTOCOL.md", "matching.py", "study.py", "../live-verifier/study.py",
           "../live-verifier/LOCK.json", "../study.py", "../fetch_inputs.py")
ORIGINAL_FEATURE = live.feature
ARCHIVE = ROOT / "model.json"
LOCK = ROOT / "LOCK.json"


def archived_fit(data):
    train = live.load(data/"train.jsonl")
    dev, _ = live.avoid_overlap(live.load(data/"dev.jsonl"), train)
    idf, unknown = live.build_idf(train)
    return train, dev, idf, unknown


def fitted(train, dev, idf, unknown):
    model, router, selected = live.fit_model(live.eligible(train), live.eligible(dev), idf, unknown)
    lock = json.loads((PARENT/"live-verifier/LOCK.json").read_text())
    return {"verifier": model, "router": [router[0], router[1].tolist()],
            "fit": selected, "parent_order": lock["policy"], "block_order": lock["block_order"]}


def timed_case(case, idf, unknown, fit, method, repeats):
    weights = (fit["verifier"][0], tuple(fit["verifier"][1]), fit["verifier"][2])
    order = tuple(fit["block_order"])
    router = fit["router"]
    result = []
    totals = []
    for _ in range(repeats):
        start = time.process_time_ns()
        decision, count, blocks = live.predict(case, idf, unknown, weights, router, method, order,
                                               fit["parent_order"])
        totals.append(time.process_time_ns()-start)
        result.append((decision, count, blocks))
    return totals, result[0]


def runner(case, idf, unknown, fit, method):
    if method == "prior":
        timing, record = timed_case(case, idf, unknown, fit, "certified", 1)
    else:
        matcher = None

        def feature(name, doc, atom):
            nonlocal matcher
            if name != "character":
                return ORIGINAL_FEATURE(name, doc, atom)
            if matcher is None:
                matcher = CharacterMatcher(doc, live.trigrams, method.removesuffix("_eager"))
            return matcher.score(atom, live.top3)

        policy = "fused_short_circuit" if method.endswith("_eager") else "certified"
        with patch.object(live, "feature", feature):
            timing, record = timed_case(case, idf, unknown, fit, policy, 1)
    return timing[0], record


def verify_features(case, idf, unknown):
    """Check all individual character scores, including short-circuited atoms."""
    doc = live.Document.make(case.evidence)
    matchers = {key: CharacterMatcher(doc, live.trigrams, key, collect=True) for key in ("dense",*INDEXES)}
    for i, claim in enumerate(case.claims):
        atom = live.prepare(doc, claim, idf, unknown, i)
        reference = ORIGINAL_FEATURE("character", doc, atom)
        for matcher in matchers.values():
            assert matcher.score(atom, live.top3)==reference, (case.key, i)
    return {method: m.counts for method,m in matchers.items()}


def evaluate(cases, idf, unknown, fit, repeats):
    numeric = []
    groups = live.clustered_ids(cases)
    rng = np.random.default_rng(SEED)
    for case in cases:
        durations = defaultdict(list)
        decisions = {}
        for _ in range(repeats):
            for method in rng.permutation(METHODS):
                ns, verdict = runner(case, idf, unknown, fit, method)
                durations[method].append(ns)
                decisions.setdefault(method, verdict)
                assert verdict==decisions[method]
        assert len({v[0] for v in decisions.values()}) == 1
        work = verify_features(case, idf, unknown)
        numeric.append({"case": case.key, "cluster": groups[case.key], "labels": case.labels,
                        "decisions": decisions, "times": dict(durations), "character": work,
                        "n_atoms":len(case.claims)})
    return numeric


def total_cpu(rows, method):
    return sum(statistics.median(row["times"][method]) for row in rows)


def summarize(rows, baseline, candidate):
    methods = {}
    for method in METHODS:
        methods[method] = {
            "sum_case_median_cpu_ns":total_cpu(rows, method),
            "character_blocks":sum(r["decisions"][method][2]["character"] for r in rows),
            "alignment_blocks":sum(r["decisions"][method][2]["alignment"] for r in rows)
        }
    for method in ("dense",*INDEXES):
        keys=rows[0]["character"][method].keys()
        methods[method]["all_atoms_character_work"]={
            key:sum(r["character"][method][key] for r in rows) for key in keys
        }
    contrasts = {}
    for base in (*CONTROLS,candidate+"_eager"):
        contrasts[base]=live.uncertainty(rows,"times",base,candidate)
    labels=np.array([int(all(r["labels"])) for r in rows])
    pred=np.array([r["decisions"]["prior"][0] for r in rows])
    primary=contrasts[baseline]
    return {"parents":len(rows),"atoms":sum(r["n_atoms"] for r in rows),
            "accuracy":float((labels==pred).mean()),"balanced_accuracy":live.bal(labels,pred),
            "method":methods,"contrasts":contrasts,"baseline":baseline,"candidate":candidate,
            "passed":primary["relative_reduction"]>=.15 and primary["paired_95"][0]>.05,
            "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)}


def save(out, rows, summary):
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (out/"per_claim.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("development","freeze","test"))
    parser.add_argument("--data",type=Path,default=PARENT/"data")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.mode=="freeze":
        summary=json.loads((args.out/"summary.json").read_text())
        assert summary["cohort"]=="development"
        with LOCK.open("x") as handle:
            lock={"source":{f:live.file_sha(ROOT/f) for f in SOURCES}, "data":live.DIGESTS,
                  "model":live.file_sha(ARCHIVE), "baseline":summary["baseline"],
                  "candidate":summary["candidate"],
                  "development_summary":live.file_sha(args.out/"summary.json")}
            json.dump(lock,handle,indent=2)
        return
    for split in ("train","dev","test") if args.mode=="test" else ("train","dev"):
        assert live.file_sha(args.data/f"{split}.jsonl")==live.DIGESTS[split]
    if args.mode=="test":
        lock=json.loads(LOCK.read_text())
        assert live.file_sha(ARCHIVE)==lock["model"]
        assert all(live.file_sha(ROOT/f)==h for f,h in lock["source"].items())
        assert lock["data"]==live.DIGESTS
    train,dev,idf,unknown=archived_fit(args.data)
    if args.mode=="development" and not ARCHIVE.exists():
        fit=fitted(train,dev,idf,unknown)
        with ARCHIVE.open("x") as handle:
            json.dump(fit,handle,indent=2)
    else:
        fit=json.loads(ARCHIVE.read_text())
    cases=dev if args.mode=="development" else live.avoid_overlap(live.load(args.data/"test.jsonl"),train+dev)[0]
    rows=evaluate(live.eligible(cases),idf,unknown,fit,5)
    if args.mode=="development":
        baseline=min(CONTROLS,key=lambda method:total_cpu(rows,method))
        candidate=min(INDEXES,key=lambda method:total_cpu(rows,method))
    else:
        baseline,candidate=lock["baseline"],lock["candidate"]
    summary=summarize(rows,baseline,candidate)
    summary["cohort"]=args.mode
    save(args.out,rows,summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
