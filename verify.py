"""Audit recorded schedules without the model, dataset, or scheduling implementation."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path


@lru_cache(maxsize=None)
def partitions(items, cap):
    if not items:
        return ((),)
    result = []
    for k in range(1, min(len(items), cap) + 1):
        for group in combinations(items, k):
            remaining = tuple(i for i in items if i not in group)
            result.extend((group,) + tail for tail in partitions(remaining, cap))
    return tuple(result)


def vertices(vector, delta):
    offsets = [delta * min(p, 1 - p) for p in vector]
    mixture = [tuple(p + sign * shift for p, shift in zip(vector, offsets)) for sign in (-1, 1)]
    size = len(vector)
    return {
        v: sum(math.prod(p if v // 2**i % 2 else 1 - p for i, p in enumerate(prob)) for prob in mixture) / 2
        for v in range(2**size)
    }


def expected(schedule, vertices, costs, overhead):
    total = 0
    previous = 0
    for group in schedule:
        mass = sum(w for outcome, w in vertices.items() if outcome & previous == previous)
        total += mass * (overhead + sum(costs[i] for i in group))
        previous += sum(2**i for i in group)
    return total


def check_schedule(groups, n, capacity, label, costs, overhead):
    flattened = tuple(i for group in groups for i in group)
    assert sorted(flattened) == list(range(n))
    assert all(1 <= len(g) <= capacity for g in groups)
    cost = checks = 0
    for rounds, batch in enumerate(groups, 1):
        checks += len(batch)
        cost += overhead + sum(costs[i] for i in batch)
        if any(not (label // 2**i % 2) for i in batch):
            return [0, cost, rounds, checks]
    return [1, cost, rounds, checks]


def verify(path):
    meta = json.loads((path / "summary.json").read_text())
    counted = defaultdict(list)
    counts = 0
    audited = 0
    case_ids = {}
    with (path / "per_claim.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            n, k = row["width"], row["capacity"]
            overhead = row["doc_words"] if row["overhead"] == "document_words" else row["overhead"]
            label = row["label_mask"]
            assert 0 <= label < 2**n
            assert row["label"] == int(label == 2**n - 1)
            setting = f"{k}/{row['overhead']}"
            for name, bits in row["schedules"].items():
                if name == "full_all_checks":
                    continue
                batches = [tuple(i for i in range(n) if b // 2**i % 2) for b in bits]
                realized = check_schedule(batches, n, k, label, row["item_costs"], overhead)
                assert realized == row["metrics"][name], (row["case"], setting, name)
                assert realized[0] == row["label"]
                if setting == meta["primary_setting"] and name in ("joint_dp", "independent_dp"):
                    law = vertices(row["learned_probabilities"],
                                   meta["learned_dependence_strength"] if name == "joint_dp" else 0.0)
                    best = min(expected(p, law, row["item_costs"], overhead)
                               for p in partitions(tuple(range(n)), k))
                    actual = expected(batches, law, row["item_costs"], overhead)
                    assert math.isclose(actual, best, rel_tol=1e-12, abs_tol=1e-12)
                    audited += 1
                counts += 1
            counted[setting].append(row)
            key = row["case"], setting
            assert key not in case_ids
            case_ids[key] = row["label"]
    for setting, rows in counted.items():
        reference = meta["settings"][setting]
        assert len(rows) == reference["count"]
        assert sum(r["label"] for r in rows) == reference["all_pass_count"]
        assert len({r["cluster"] for r in rows}) == reference["clusters"]
        for method, scores in reference["methods"].items():
            expected_mean = sum(r["metrics"][method][1] for r in rows) / len(rows)
            assert math.isclose(expected_mean, scores["mean_proxy_cost_units"], abs_tol=1e-9)
    primary_cases = len(counted[meta["primary_setting"]])
    assert audited == 2 * primary_cases
    with (path / "per_claim.jsonl").open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"PASS: {counts} verdict/cost replays, {audited} independent exhaustive primary schedule audits, "
          f"{len(counted)} cost settings. SHA-256: {sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    verify(parser.parse_args().results)
