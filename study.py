"""WiCE label-replay study. Run from the project root after fetching data."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time

import numpy as np

from scheduling import (
    fixed_schedule, lattice, mixture_event_probabilities, optimise,
    product_event_probabilities, replay,
)

TRAIN_SHA = "e8ba1ed589e22c5f9d41c8d858dd10bdc17665d224f8ce28be9d9024239cf513"
DEV_SHA = "4747a5a60c85c0c12cbf1cc7a2ab4bc9279fd4fd01a83d36630b7151330b6f57"
PROVENANCE = "https://raw.githubusercontent.com/ryokamoi/wice/ddeb6c183665e2a20c5f03c5aa07f03888b9870f/data/entailment_retrieval/subclaim/"
RATIOS = [0, 16, 48, 144, "document_words"]
PRIMARY = (48, 2)
CAPACITIES = (2, 4)
SEED = 731405
TOKEN = re.compile(r"[\w]+", re.UNICODE)
LABELS = {"supported": 1, "partially_supported": 0, "not_supported": 0}


def words(text):
    return TOKEN.findall(text.casefold())


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Case:
    key: str
    title: str
    doc_id: str
    context: str
    evidence: list[str]
    claims: list[str]
    labels: tuple[int, ...]


def load(path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            parent, number = row["meta"]["id"].rsplit("-", 1)
            groups[parent].append((int(number), row))
    out = []
    for parent, indexed in sorted(groups.items()):
        rows = [row for _, row in sorted(indexed)]
        first = rows[0]
        assert sorted(i for i, _ in indexed) == list(range(len(rows)))
        assert all(row["evidence"] == first["evidence"] for row in rows)
        title = " ".join(words(first["meta"]["claim_title"]))
        context = " ".join(words(first["meta"]["claim_context"]))
        doc_id = fingerprint(" ".join(words(" ".join(first["evidence"]))))
        out.append(Case(parent, title, doc_id, fingerprint(context), first["evidence"],
                        [row["claim"] for row in rows],
                        tuple(LABELS[row["label"]] for row in rows)))
    return out


def avoid_overlap(rows, seen):
    titles = {case.title for case in seen}
    docs = {case.doc_id for case in seen}
    contexts = {case.context for case in seen}
    filtered = [case for case in rows if (case.title not in titles and case.doc_id not in docs
                                          and case.context not in contexts)]
    return filtered, len(rows) - len(filtered)


def clustered_ids(cases):
    parent = {case.key: case.key for case in cases}
    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key
    seen = {}
    for case in cases:
        for link in (("doc", case.doc_id), ("title", case.title), ("context", case.context)):
            a = root(case.key)
            b = root(seen.setdefault(link, a))
            parent[max(a, b)] = min(a, b)
    return {case.key: root(case.key) for case in cases}


def build_idf(rows):
    docs = {}
    for case in rows:
        docs.setdefault(case.doc_id, case.evidence)
    df = Counter(term for body in docs.values() for term in set(words(" ".join(body))))
    n_docs = len(docs)
    return {term: math.log((n_docs + 1) / (n + 1)) + 1 for term, n in df.items()}, math.log(n_docs + 1) + 1


def features(case, idf, unknown):
    sentences = [set(words(sentence)) for sentence in case.evidence]
    doc = set().union(*sentences)
    width = len(case.claims)
    rows = []
    bill = []
    for i, sub in enumerate(case.claims):
        tokens = words(sub)
        weights = {token: idf.get(token, unknown) for token in tokens}
        denom = max(1.0, sum(weights.values()))
        overlap = sorted((sum(weights[t] for t in weights.keys() & sent) / denom
                          for sent in sentences), reverse=True)
        top = (overlap + [0.0, 0.0, 0.0])[:3]
        doc_coverage = sum(weights[t] for t in weights.keys() & doc) / denom
        numeric = {t for t in tokens if any(c.isdigit() for c in t)}
        numeric_missing = len(numeric - doc) / max(1, len(numeric))
        rows.append([math.log1p(len(tokens)), math.log1p(len(sentences)), math.log1p(width),
                     i / max(1, width - 1), *top, doc_coverage, numeric_missing])
        bill.append(20 + len(tokens))
    return np.array(rows), np.array(bill), sum(len(words(sent)) for sent in case.evidence)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def fit(cases):
    idf, unknown = build_idf(cases)
    per_case = [features(case, idf, unknown)[0] for case in cases]
    data = np.concatenate(per_case)
    y = np.array([y for case in cases for y in case.labels])
    mean = data.mean(axis=0)
    sd = data.std(axis=0).clip(.001)
    x = np.column_stack((np.ones(len(data)), (data - mean) / sd))
    w = np.zeros(x.shape[1])
    penalty = np.diag([1e-6] + [.01] * (x.shape[1] - 1))
    for _ in range(20):
        p = sigmoid(x @ w)
        g = x.T @ (p - y) / len(y) + penalty @ w
        h = x.T @ (x * (p * (1 - p))[:, None]) / len(y) + penalty
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = np.clip(sigmoid(x @ w), .025, .975)
    offset = 0
    grid = np.linspace(0, .95, 20)
    scores = np.zeros(len(grid))
    # Fit only a common dependence strength. Both schedulers receive the same marginals.
    for case in cases:
        pr = p[offset:offset + len(case.labels)]
        offset += len(case.labels)
        offsets = grid[:, None] * np.minimum(pr, 1 - pr)
        truth = np.array(case.labels, dtype=bool)
        plus = np.where(truth, pr + offsets, 1 - pr - offsets).prod(axis=1)
        minus = np.where(truth, pr - offsets, 1 - pr + offsets).prod(axis=1)
        scores += np.log((plus + minus) / 2)
    delta = float(grid[np.argmax(scores)])
    return idf, unknown, mean, sd, w, delta


def cases_for_split(cases, model, fixed_widths=None):
    idf, unknown, mean, sd, w, delta = model
    clusters = clustered_ids(cases)
    observations = []
    for case in cases:
        n = len(case.labels)
        # With 1 check, all methods make the same decision and cannot save its cost.
        if n < 2 or n > 8:
            continue
        feat, costs, doc_words = features(case, idf, unknown)
        pr = np.clip(sigmoid(np.column_stack((np.ones(n), (feat - mean) / sd)) @ w), .025, .975)
        order = sorted(range(n), key=lambda i: (costs[i] / (1 - pr[i]), i))
        y = sum(1 << i for i, bit in enumerate(case.labels) if bit)
        for cap in CAPACITIES:
            membership, _ = lattice(n, cap)
            qi = product_event_probabilities(pr, membership)
            qm = mixture_event_probabilities(pr, delta, membership)
            max_batch = fixed_schedule(order, cap)
            for ratio in RATIOS:
                fixed_key = f"{cap}/{ratio}"
                overhead = doc_words if ratio == "document_words" else ratio
                joint, _ = optimise(qm, costs, overhead, cap)
                independent, _ = optimise(qi, costs, overhead, cap)
                schedules = {
                    "joint_dp": joint,
                    "independent_dp": independent,
                    "max_batch": max_batch,
                    "natural_sequential": fixed_schedule(range(n), 1),
                    "full_all_checks": fixed_schedule(range(n), cap),
                }
                if fixed_widths is None:
                    schedules.update({f"fixed_{i}": fixed_schedule(order, i) for i in range(1, cap + 1)})
                else:
                    schedules["selected_fixed"] = fixed_schedule(order, fixed_widths[fixed_key])
                values = {name: replay(sch, y, costs, overhead, cap)
                          for name, sch in schedules.items() if name != "full_all_checks"}
                full = (n // cap + bool(n % cap))
                values["full_all_checks"] = (int(all(case.labels)), float(costs.sum() + full * overhead), full, n)
                assert all(v[0] == int(all(case.labels)) for v in values.values())
                independent_p = float(qi[-1])
                joint_p = float(qm[-1])
                observations.append({
                    "case": case.key, "cluster": clusters[case.key], "width": n, "label": int(all(case.labels)),
                    "label_mask": y, "capacity": cap, "overhead": ratio, "doc_words": doc_words,
                    "item_costs": costs.tolist(), "learned_probabilities": pr.tolist(),
                    "joint_all_pass_probability": joint_p, "independent_all_pass_probability": independent_p,
                    "schedules": schedules, "metrics": values,
                })
    return observations


def choose_widths(dev):
    return {
        f"{cap}/{ratio}": min(range(1, cap + 1), key=lambda b: sum(
            row["metrics"][f"fixed_{b}"][1] for row in dev
            if row["capacity"] == cap and row["overhead"] == ratio))
        for cap in CAPACITIES for ratio in RATIOS
    }


def interval(rows, contrast, seed=SEED):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cluster"]].append(contrast(row))
    pairs = np.array([(len(x), sum(x)) for x in groups.values()])
    rng = np.random.default_rng(seed)
    # Stream batches to cap memory independent of dataset expansion.
    values = []
    for start in range(0, 4000, 100):
        draws = rng.integers(len(pairs), size=(100, len(pairs)))
        batch = pairs[draws].sum(axis=1)
        values.extend((batch[:, 1] / batch[:, 0]).tolist())
    return [round(float(q), 8) for q in np.quantile(values, [.025, .975])]


def contrast(rows, first, second):
    f = lambda row: row["metrics"][first][1] - row["metrics"][second][1]
    return {
        "paired_mean_extra_units": float(np.mean([f(r) for r in rows])),
        "doc_title_group_bootstrap95": interval(rows, f),
    }


def summarize(observations):
    settings = {}
    for cap in CAPACITIES:
        for ratio in RATIOS:
            rows = [r for r in observations if r["capacity"] == cap and r["overhead"] == ratio]
            y = np.array([r["label"] for r in rows])
            methods = {
                method: dict(zip(("decision_positive_rate", "mean_proxy_cost_units", "mean_rounds", "mean_checks"),
                                 np.array([r["metrics"][method] for r in rows]).mean(axis=0).tolist()))
                for method in rows[0]["metrics"]
            }
            settings[f"{cap}/{ratio}"] = {
                "count": len(rows), "clusters": len({r["cluster"] for r in rows}),
                "all_pass_count": int(y.sum()), "methods": methods,
                "joint_vs_independent": contrast(rows, "joint_dp", "independent_dp"),
                "joint_vs_selected_fixed": contrast(rows, "joint_dp", "selected_fixed"),
                "independent_vs_selected_fixed": contrast(rows, "independent_dp", "selected_fixed"),
                "joint_and_independent_schedules_differ": sum(r["schedules"]["joint_dp"] != r["schedules"]["independent_dp"] for r in rows),
                "joint_brier": float(np.mean([(r["joint_all_pass_probability"] - r["label"])**2 for r in rows])),
                "independent_brier": float(np.mean([(r["independent_all_pass_probability"] - r["label"])**2 for r in rows])),
                "brier_diff_bootstrap95": interval(
                    rows, lambda r: (r["joint_all_pass_probability"] - r["label"])**2
                    - (r["independent_all_pass_probability"] - r["label"])**2),
            }
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    train, dev = (args.data / f"{s}.jsonl" for s in ("train", "dev"))
    for path, sha in ((train, TRAIN_SHA), (dev, DEV_SHA)):
        assert file_sha(path) == sha, f"Unexpected dataset SHA-256 {path}"
    fit_rows = load(train)
    original_dev = load(dev)
    dev_rows, excluded_dev = avoid_overlap(original_dev, fit_rows)
    model = fit(fit_rows)
    dev_outcome = cases_for_split(dev_rows, model)
    width_choices = choose_widths(dev_outcome)
    protocol_sha = file_sha(Path("PROTOCOL.md"))
    result = {
        "mode": "development" if args.dev_only else "official_test",
        "protocol_sha": protocol_sha,
        "train_sha": TRAIN_SHA, "dev_sha": DEV_SHA,
        "fitting_claims": len(fit_rows), "dev_claims": len(dev_rows),
        "dev_claims_excluded_for_overlap": excluded_dev,
        "learned_dependence_strength": model[-1], "width_choices": width_choices,
    }
    if args.dev_only:
        print(json.dumps(result, indent=2))
        return
    lock = json.loads(Path("protocol_lock.json").read_text())
    for source_path, checksum in lock["sha256"].items():
        assert file_sha(Path(source_path)) == checksum, f"Locked source differs: {source_path}"
    test_path = args.data / "test.jsonl"
    original_test = load(test_path)
    test_rows, excluded_test = avoid_overlap(original_test, fit_rows + original_dev)
    outcomes = cases_for_split(test_rows, model, width_choices)
    summary = summarize(outcomes)
    result.update({
        "test_sha": file_sha(test_path), "total_test_parent_claims": len(original_test),
        "test_claims_excluded_for_overlap": excluded_test,
        "test_singletons": sum(len(c.labels) == 1 for c in test_rows),
        "test_claims_above_8": sum(len(c.labels) > 8 for c in test_rows),
        "primary_setting": f"{PRIMARY[1]}/{PRIMARY[0]}",
        "resources": {
            "wall_seconds": round(time.perf_counter() - start, 4),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        },
        "settings": summary,
    })
    args.output.mkdir(exist_ok=True, parents=True)
    with (args.output / "per_claim.jsonl").open("w") as out:
        for row in outcomes:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["settings"][result["primary_setting"]], indent=2))


if __name__ == "__main__":
    main()
