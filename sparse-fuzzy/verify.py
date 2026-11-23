"""Recompute aggregate case costs, paired intervals and verdict equality."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ratios(rows, proposed, control):
    groups = defaultdict(lambda: [0,0])
    for row in rows:
        medians = tuple(statistics.median(row["times"][method]) for method in (proposed,control))
        group = groups[row["cluster"]]
        for index, median in enumerate(medians):
            group[index] += median
    costs = np.asarray(list(groups.values()))
    totals = costs.sum(axis=0)
    random = np.random.default_rng(206753)
    draws = random.integers(len(costs),size=(5000,len(costs)))
    resample = costs[draws].sum(axis=1)
    return float(1-totals[0]/totals[1]), np.quantile(1-resample[:,0]/resample[:,1],[.025,.975])


def audit(path, expected_cases, expected_atoms, candidate):
    directory = ROOT/path
    rows = [json.loads(line) for line in (directory/"per_claim.jsonl").read_text().splitlines()]
    official = json.loads((directory/"summary.json").read_text())
    assert official["candidate"]==candidate
    assert len(rows)==expected_cases==official["parents"]
    assert sum(row["n_atoms"] for row in rows)==expected_atoms
    assert len({row["case"] for row in rows})==expected_cases
    assert all(len({d[0] for d in row["decisions"].values()})==1 for row in rows)
    assert all(len(values)==5 and all(t>0 for t in values) for row in rows for values in row["times"].values())
    for control, summary in official["contrasts"].items():
        ratio, interval = ratios(rows, candidate, control)
        assert np.allclose(ratio,summary["relative_reduction"],rtol=0,atol=1e-12)
        assert np.allclose(interval,summary["paired_95"],rtol=0,atol=1e-12)
    if path=="results":
        for method in ("dense",candidate):
            totals = official["method"][method]["all_atoms_character_work"]
            for key,count in totals.items():
                assert count==sum(row["character"][method][key] for row in rows)
        assert official["passed"]
    print(f"{path}: verified {expected_cases} verdicts and all bootstrap intervals.")


def main():
    lock = json.loads((ROOT/"LOCK.json").read_text())
    assert all(digest(ROOT/f)==h for f,h in lock["source"].items())
    assert digest(ROOT/"DEVELOPMENT.json")==lock["development_summary"]
    assert digest(ROOT/"model.json")==lock["model"]
    for line in (ROOT/"TRANSFER_SHA256SUMS").read_text().splitlines():
        expected,name = line.split(maxsplit=1)
        assert digest(ROOT.parent/name)==expected
    audit("results",310,830,lock["candidate"])
    audit("scifact",339,339,lock["candidate"])


if __name__=="__main__":
    main()
