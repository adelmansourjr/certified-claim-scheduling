"""Audit archived WiCE outputs without running or importing the trained verifier."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
strong = "fused_short_circuit"
candidate = "certified"
stages = {"alignment", "character", "local"}


def cpu_comparison(rows, channel):
    grouped = defaultdict(lambda: [0, 0])
    for row in rows:
        for i, policy in enumerate((strong, candidate)):
            grouped[row["cluster"]][i] += median(row[channel][policy])
    block_costs = np.array(list(grouped.values()), dtype=np.int64)
    full, cheap = block_costs.sum(axis=0)
    rng = np.random.default_rng(206753)
    draw = rng.integers(0, len(grouped), (5000, len(grouped)))
    replicas = block_costs[draw].sum(axis=1)
    changes = 1 - replicas[:, 1] / replicas[:, 0]
    return len(grouped), 1 - cheap / full, np.quantile(changes, (0.025, 0.975))


def verify_lock():
    lock = json.loads((ROOT / "LOCK.json").read_text())
    for section, root in (("source", ROOT), ("parent", ROOT.parent)):
        for name, expected in lock[section].items():
            assert sha256((root / name).read_bytes()).hexdigest() == expected, name
    dev_bytes = (ROOT / "DEV_SUMMARY.json").read_bytes()
    assert sha256(dev_bytes).hexdigest() == lock["dev_sha256"]
    dev = json.loads(dev_bytes)
    assert dev["baseline_order"] == lock["policy"]
    assert dev["certified_order"] == lock["block_order"]
    assert dev["baseline_order"] == min(dev["dev_order_cpu_ns"], key=dev["dev_order_cpu_ns"].get)
    assert "/".join(dev["certified_order"]) == min(
        dev["dev_block_order_cpu_ns"], key=dev["dev_block_order_cpu_ns"].get)
    return lock


def verify(result_dir):
    lock = verify_lock()
    summary = json.loads((result_dir / "summary.json").read_bytes())
    rows = [json.loads(line) for line in (result_dir / "per_claim.jsonl").read_bytes().splitlines()]
    assert summary["cohort"] == "test"
    assert summary["baseline_order"] == lock["policy"]
    assert summary["certified_order"] == lock["block_order"]
    methods = set(summary["method"])
    assert len(rows) == summary["parents"] == len({row["case"] for row in rows}) == 310

    matrix = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for row in rows:
        assert all(a in (0, 1) for a in row["labels"]) and 2 <= len(row["labels"]) <= 8
        assert set(row["times"]) == methods == set(row["decisions"])
        assert set(row["cached_times"]) == {strong, candidate}
        for channel in ("times", "cached_times"):
            assert all(len(t) == 3 and min(t) > 0 for t in row[channel].values())
        votes = {entry[0] for entry in row["decisions"].values()}
        assert len(votes) == 1
        matrix[(int(all(row["labels"])), votes.pop())] += 1
        assert row["decisions"]["exhaustive"][1]["atoms"] == len(row["labels"])
        assert row["decisions"][strong][1]["atoms"] == row["decisions"][candidate][1]["atoms"]
        for policy, (_, counts, detail) in row["decisions"].items():
            assert set(detail) == stages
            assert 0 <= counts["certificates"] <= counts["atoms"] <= len(row["labels"])
            assert all(0 <= value <= counts["atoms"] for value in detail.values())
            last = lock["block_order"][-1]
            assert detail[last] == counts["atoms"] - counts["certificates"]
            if policy not in (candidate, "box_certificate"):
                assert counts["certificates"] == 0 and set(detail.values()) == {counts["atoms"]}
        for name in stages:
            assert row["decisions"][candidate][2][name] <= row["decisions"]["box_certificate"][2][name]

    assert summary["parent_disagreements"] == 0
    assert summary["verdict_counts"] == {f"label_{a}_prediction_{b}": n for (a, b), n in matrix.items()}
    n = len(rows)
    tn, fp, fn, tp = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    assert np.isclose(summary["parent_accuracy"], (tn + tp) / n)
    assert np.isclose(summary["parent_balanced_accuracy"], (tn / (tn + fp) + tp / (tp + fn)) / 2)
    assert np.isclose(summary["majority_accuracy"], max(tn + fp, tp + fn) / n)
    assert np.isclose(summary["all_supported_rate"], (tp + fn) / n)
    assert np.isclose(summary["all_supported_predicted"], (tp + fp) / n)
    for policy, result in summary["method"].items():
        assert result["sum_case_median_cpu_ns"] == sum(median(row["times"][policy]) for row in rows)
        for metric, target in (("atoms", "checks"), ("certificates", "certified_checks")):
            assert result[target] == sum(row["decisions"][policy][1][metric] for row in rows)
        for stage, count in result["feature_blocks"].items():
            assert count == sum(row["decisions"][policy][2][stage] for row in rows)

    long_rows = [row for row in rows if row["document_lines"] >= summary["long_document_lines"]]
    for group, subset in (("all", rows), ("long", long_rows)):
        for channel in ("times", "cached_times"):
            clusters, reduction, interval = cpu_comparison(subset, channel)
            stored = summary[group][channel]
            assert clusters == stored["clusters"]
            assert np.isclose(reduction, stored["relative_reduction"])
            assert np.allclose(interval, stored["paired_95"])
    _, reduction, interval = cpu_comparison(rows, "times")
    assert summary["passes_primary"] == (reduction >= .05 and interval[0] > 0)
    slower = sum(median(row["times"][candidate]) > median(row["times"][strong]) for row in rows)
    print(f"Verified {n} identical parent verdicts. Certified pipeline slower on {slower}/{n} individual parents.")
    print(f"CPU reduction {reduction:.2%} [{interval[0]:.2%}, {interval[1]:.2%}]; "
          f"long-document parents {len(long_rows)}/{n}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    verify(parser.parse_args().results)
