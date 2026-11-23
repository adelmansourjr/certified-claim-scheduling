"""Exact short-circuiting of a trained compound claim verifier."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from difflib import SequenceMatcher
from itertools import permutations
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from fetch_inputs import DIGESTS
from study import avoid_overlap, build_idf, clustered_ids, file_sha, load, words


EPSILON = 1e-9
SEED = 206753
BLOCKS = ("alignment", "character", "local")
POLICIES = ("exhaustive", "sequential", "lexical", "fused_short_circuit", "box_certificate", "certified")
WIDTH = 6
SHORTLIST = 5
ORDERS = tuple(permutations(BLOCKS))
NEGATION = {"not", "no", "none", "never", "neither", "without", "n't"}


@dataclass
class Document:
    sentences: list
    postings: dict
    terms: set

    @classmethod
    def make(cls, evidence):
        sentences = [tuple(words(sentence)[:160]) for sentence in evidence]
        postings = defaultdict(list)
        for i, s in enumerate(sentences):
            for t in set(s):
                postings[t].append(i)
        return cls(sentences, postings, set(postings))


@dataclass
class Atom:
    claim: tuple
    weighted: dict
    chosen: list
    base: tuple
    public_index: int


def coverage(tokens, weights):
    return sum(weights[t] for t in set(tokens) & weights.keys()) / max(1e-12, sum(weights.values()))


def prepare(document, text, idf, unknown, atom_id):
    claim = tuple(words(text)[:64])
    weights = {word: idf.get(word, unknown) for word in claim}
    sum_weights = max(1e-12, sum(weights.values()))
    ranking = defaultdict(float)
    for term, weight in weights.items():
        for i in document.postings.get(term, ()):
            ranking[i] += weight / sum_weights
    best = sorted(ranking, key=lambda i: (-ranking[i], len(document.sentences[i]), i))[:SHORTLIST]
    if not best and document.sentences:
        best = [0]
    best_scores = ([ranking[i] for i in best] + [0.0] * 3)[:3]
    nums = {word for word in weights if any(c.isdigit() for c in word)}
    present = set(weights) & document.terms
    base = (
        len(claim) / 64, min(len(document.sentences), 300) / 300,
        len(present) / max(1, len(weights)),
        sum(weights[t] for t in present) / sum_weights,
        len(nums & document.terms) / max(1, len(nums)),
        float(bool(nums)), float(bool(NEGATION & weights.keys())), *best_scores,
    )
    return Atom(claim, weights, best, base, atom_id)


def top3(values):
    return tuple((sorted(values, reverse=True) + [0.0] * 3)[:3])


@lru_cache(maxsize=8192)
def trigrams(word):
    w = "^" + word + "$"
    return {w[i:i+3] for i in range(max(1, len(w)-2))}


def feature(name, doc, a):
    if name == "alignment":
        longest = []
        match_ratio = []
        for idx in a.chosen:
            sent = doc.sentences[idx]
            matcher = SequenceMatcher(None, a.claim, sent, autojunk=False)
            longest.append(matcher.find_longest_match().size / max(1, len(a.claim)))
            match_ratio.append(matcher.ratio())
        return (*top3(longest), *top3(match_ratio))
    if name == "character":
        missing = {t: trigrams(t) for t in a.weighted if len(t) >= 4}
        denom = max(1e-12, sum(a.weighted[t] for t in missing))
        scores = []
        scores_soft = []
        for idx in a.chosen:
            words_here = set(doc.sentences[idx])
            other = [trigrams(t) for t in words_here if len(t) >= 4]
            matches = {t: max((len(grams & h) / max(1, len(grams | h)) for h in other), default=0)
                       for t, grams in missing.items() if t not in words_here}
            scores.append(sum(a.weighted[t] * (1.0 if t in words_here else matches[t] >= .65)
                              for t in missing) / denom)
            scores_soft.append(sum(a.weighted[t] * (1.0 if t in words_here else matches[t])
                                   for t in missing) / denom)
        return (*top3(scores), *top3(scores_soft))
    if name == "local":
        near = []
        ranked = []
        for idx in a.chosen:
            one = doc.sentences[idx]
            neighborhood = [t for block in doc.sentences[max(0, idx-1):idx+2] for t in block]
            near.append(coverage(neighborhood, a.weighted))
            ranked.append(coverage(one[:max(8, len(a.claim))], a.weighted))
        return (*top3(near), *top3(ranked))
    raise KeyError(name)


def encoded(rows, idf, unknown):
    result = []
    for row in rows:
        d = Document.make(row.evidence)
        a = [prepare(d, t, idf, unknown, i) for i, t in enumerate(row.claims)]
        result.append(np.array([(*x.base, *(z for b in BLOCKS for z in feature(b, d, x))) for x in a]))
    return result


def trained(x, y, penalty):
    mean, sd = x.mean(0), np.maximum(x.std(0), .02)
    z = (x-mean)/sd
    z = np.column_stack((np.ones(len(z)), z))
    coeff = np.zeros(z.shape[1])
    matrix = np.diag([1e-10, *([penalty] * x.shape[1])])
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(z @ coeff, -30, 30)))
        gradient = z.T @ (p-y) / len(y) + matrix @ coeff
        hessian = z.T @ (z * (p*(1-p))[:, None]) / len(y) + matrix
        coeff -= np.linalg.solve(hessian, gradient)
    weight = coeff[1:] / sd
    return float(coeff[0] - mean @ weight), weight


def linear(weight, values):
    return math.fsum(float(w) * v for w, v in zip(weight, values))


def interval_bounds(weight):
    # Each block produces two top-three vectors x1>=x2>=x3 in [0,1].
    # A descending triple is a mixture of the four prefix vertices.
    ranges = []
    for j in (0, 3):
        vertices = [math.fsum(weight[j:j+n]) for n in range(4)]
        ranges.append((min(vertices), max(vertices)))
    return math.fsum(x[0] for x in ranges), math.fsum(x[1] for x in ranges)


@lru_cache(maxsize=60)
def compiled(weights, order, box=False):
    base = weights[:-WIDTH*len(BLOCKS)]
    tail = weights[-WIDTH*len(BLOCKS):]
    blocks = {b:tail[i*WIDTH:(i+1)*WIDTH] for i,b in enumerate(BLOCKS)}
    bounds = {b:(math.fsum(min(0,w) for w in blocks[b]), math.fsum(max(0,w) for w in blocks[b]))
                   if box else interval_bounds(blocks[b]) for b in BLOCKS}
    suffixes = tuple((math.fsum(bounds[b][0] for b in order[i:]),
                      math.fsum(bounds[b][1] for b in order[i:])) for i in range(len(order)))
    return base, blocks, suffixes


def predict_atom(doc, a, bias, w_base, w_blocks, cut, order, suffixes, cert, counts, stage_calls):
    # Use a canonical summation order for any path that reaches full evidence.
    base = bias + linear(w_base, a.base) - cut
    if not cert:
        # Restrict eager scoring to fused sums and avoid interval bookkeeping.
        whole = [linear(w_blocks[stage], feature(stage, doc, a)) for stage in order]
        for stage in BLOCKS:
            stage_calls[stage] += 1
        return int(math.fsum((base, *(whole[order.index(b)] for b in BLOCKS))) >= 0)
    contributions = {}
    for i, stage in enumerate(order):
        lower, upper = suffixes[i]
        partial = math.fsum((base, *contributions.values()))
        low, high = partial+lower, partial+upper
        if low >= EPSILON or high < -EPSILON:
            counts["certificates"] += 1
            return int(low >= EPSILON)
        contributions[stage] = linear(w_blocks[stage], feature(stage, doc, a))
        stage_calls[stage] += 1
    exact = math.fsum((base, *(contributions[b] for b in BLOCKS)))
    return int(exact >= 0)


def predict(row, idf, unknown, model, router, policy, block_order, parent_order, doc=None):
    doc = doc if doc is not None else Document.make(row.evidence)
    bias, weights, cut = model
    w_base, w_blocks, suffixes = compiled(weights, block_order, policy=="box_certificate")
    counts = {"atoms": 0, "certificates": 0}
    checks = dict.fromkeys(BLOCKS, 0)
    atoms = (prepare(doc, t, idf, unknown, i) for i, t in enumerate(row.claims))
    if policy in ("lexical", "fused_short_circuit", "box_certificate", "certified"):
        use_order = parent_order if policy != "lexical" else "lexical"
        if use_order == "lexical":
            atom_order = sorted(atoms, key=lambda x: (linear(router[1], x.base), x.public_index))
        elif use_order == "cost":
            atom_order = sorted(atoms, key=lambda x: (len(x.claim)*max(1, len(x.chosen)), x.public_index))
        else:
            atom_order = atoms
    else:
        atom_order = atoms
    out = 1
    for a in atom_order:
        counts["atoms"] += 1
        yes = predict_atom(doc, a, bias, w_base, w_blocks, cut, block_order, suffixes, policy in ("box_certificate","certified"), counts, checks)
        out &= yes
        if policy != "exhaustive" and not out:
            break
    return out, counts, checks


def bal(y, a):
    y, a = np.asarray(y), np.asarray(a)
    return float(np.mean([np.mean(a[y == t] == t) for t in (0, 1)]))


def fit_model(train, dev, idf, unknown):
    train_encoded, dev_encoded = encoded(train, idf, unknown), encoded(dev, idf, unknown)
    y_train = np.concatenate([c.labels for c in train])
    estimator = []
    label_dev = [int(all(c.labels)) for c in dev]
    for penalty in (.001, .01, .1):
        bias, weights = trained(np.concatenate(train_encoded), y_train, penalty)
        scores = [np.min(bias + x @ weights) for x in dev_encoded]
        for cut in np.linspace(-1.5, .5, 41):
            estimator.append((bal(label_dev, np.array(scores)>=cut), -abs(cut), -penalty,
                              bias, weights, cut))
    selected = max(estimator, key=lambda e: e[:3])
    router = trained(np.concatenate([x[:, :10] for x in train_encoded]), y_train, .01)
    return (selected[3], tuple(map(float,selected[4])), selected[5]), router, {"penalty": -selected[2], "cut": float(selected[-1]),
                                  "dev_balanced_accuracy": selected[0]}


def select_plans(dev, idf, unknown, model, router):
    # Select strongest exact short-circuit order by complete measured dev CPU time.
    controls = {}
    for name in ("original", "lexical", "cost"):
        elapsed = 0
        for c in dev:
            start = time.process_time_ns()
            predict(c, idf, unknown, model, router, "fused_short_circuit", BLOCKS, name)
            elapsed += time.process_time_ns()-start
        controls[name] = elapsed
    control = min(controls, key=controls.get)
    order_scores = {}
    for order in ORDERS:
        elapsed = 0
        for c in dev:
            start = time.process_time_ns()
            predict(c, idf, unknown, model, router, "certified", order, control)
            elapsed += time.process_time_ns()-start
        order_scores[order] = elapsed
    return control, min(order_scores, key=order_scores.get), controls, {
        "/".join(k):v for k,v in order_scores.items()}


def evaluate(test, idf, unknown, model, router, control, order):
    groups = clustered_ids(test)
    rng = np.random.default_rng(SEED)
    rows = []
    repetition = 3
    for c in test:
        record = {"case": c.key, "cluster": groups[c.key], "labels": c.labels,
                  "document_lines": len(c.evidence), "times": defaultdict(list),
                  "cached_times": defaultdict(list), "decisions": {}}
        d = Document.make(c.evidence)
        for counter in range(repetition):
            for policy in rng.permutation(POLICIES):
                start = time.process_time_ns()
                verdict, count, blocks = predict(c, idf, unknown, model, router, policy, order, control)
                record["times"][policy].append(time.process_time_ns()-start)
                if counter == 0:
                    record["decisions"][policy] = [verdict,count,blocks]
        for _ in range(repetition):
            for policy in rng.permutation(("fused_short_circuit","certified")):
                start = time.process_time_ns()
                predict(c,idf,unknown,model,router,policy,order,control,doc=d)
                record["cached_times"][policy].append(time.process_time_ns()-start)
        assert len({v[0] for v in record["decisions"].values()}) == 1
        for b in BLOCKS:
            assert record["decisions"]["certified"][2][b] <= record["decisions"]["box_certificate"][2][b]
        rows.append(record)
    return rows


def uncertainty(rows, metric, baseline, candidate):
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["cluster"]].append((statistics.median(r[metric][baseline]),
                                    statistics.median(r[metric][candidate])))
    sums = np.array([np.sum(x, axis=0) for x in blocks.values()])
    rng = np.random.default_rng(SEED)
    ids = rng.integers(len(sums),size=(5000,len(sums)))
    agg = sums[ids].sum(1)
    ratios = 1-agg[:,1]/agg[:,0]
    return {"relative_reduction":float(1-sums.sum(0)[1]/sums.sum(0)[0]),
            "paired_95":np.quantile(ratios,[.025,.975]).tolist(),
            "clusters":len(blocks)}


def report(rows, long_at):
    y = [int(all(r["labels"])) for r in rows]
    yhat = [r["decisions"]["exhaustive"][0] for r in rows]
    assert len(rows)
    result = {"parents":len(rows), "parent_accuracy":float(np.mean(np.array(y)==yhat)),
              "parent_balanced_accuracy":bal(y,yhat),"all_supported_rate":float(np.mean(y)),
              "all_supported_predicted":float(np.mean(yhat)), "parent_disagreements":0,
              "majority_accuracy":float(max(np.mean(y),1-np.mean(y))),
              "verdict_counts": {f"label_{truth}_prediction_{pred}": sum(a==truth and b==pred for a,b in zip(y,yhat))
                                 for truth in (0,1) for pred in (0,1)}, "method":{}}
    for p in POLICIES:
        result["method"][p] = {
            "sum_case_median_cpu_ns":sum(statistics.median(r["times"][p]) for r in rows),
            "checks":int(sum(r["decisions"][p][1]["atoms"] for r in rows)),
            "certified_checks":int(sum(r["decisions"][p][1]["certificates"] for r in rows)),
            "feature_blocks":{b:int(sum(r["decisions"][p][2][b] for r in rows)) for b in BLOCKS},
        }
    for topic, subset in (("all",rows), ("long", [r for r in rows if r["document_lines"]>=long_at])):
        result[topic] = {channel:uncertainty(subset,channel,"fused_short_circuit","certified")
                         for channel in ("times","cached_times")} if subset else {}
    result["passes_primary"]=bool(result["all"]["times"]["relative_reduction"]>=.05
                                  and result["all"]["times"]["paired_95"][0]>0)
    return result


def eligible(rows):
    return [c for c in rows if 2<=len(c.claims)<=8]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",choices=("dev","test"),default="test")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--data",type=Path,default=ROOT.parent/"data")
    args=ap.parse_args()
    required=("train","dev") if args.mode=="dev" else ("train","dev","test")
    for split in required:
        assert file_sha(args.data/f"{split}.jsonl")==DIGESTS[split],split
    if args.mode=="test":
        lock=json.loads((ROOT/"LOCK.json").read_text())
        assert all(file_sha(ROOT/f)==h for f,h in lock["source"].items())
        assert all(file_sha(ROOT.parent/f)==h for f,h in lock["parent"].items())
    assert not args.out.exists(),"results path exists"
    train=load(args.data/"train.jsonl")
    dev,_=avoid_overlap(load(args.data/"dev.jsonl"),train)
    idf,unknown=build_idf(train)
    model,router,selected=fit_model(eligible(train),eligible(dev),idf,unknown)
    print("fit",selected,flush=True)
    # Use all eligible dev controls and select under the complete path.
    control,order,dev_controls,dev_stages=select_plans(eligible(dev),idf,unknown,model,router)
    print("policy",control,order,flush=True)
    if args.mode=="test":
        control, order = lock["policy"], tuple(lock["block_order"])
    long_at=float(np.quantile([len(d.evidence) for d in eligible(dev)],.8))
    sample=eligible(dev)[:60] if args.mode=="dev" else eligible(avoid_overlap(load(args.data/"test.jsonl"),train+dev)[0])
    rows=evaluate(sample,idf,unknown,model,router,control,order)
    result=report(rows,long_at)
    result.update({"cohort":args.mode,"fits":selected,"baseline_order":control,"certified_order":order,
                   "dev_order_cpu_ns":dev_controls,"dev_block_order_cpu_ns":dev_stages,
                   "long_document_lines":long_at,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=="darwin" else 1024),
                   "platform":sys.platform})
    args.out.mkdir(parents=True)
    (args.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.out/"per_claim.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
