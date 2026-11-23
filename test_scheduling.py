"""Exhaustive small-instance checks and an exact equal-marginals witness."""
from fractions import Fraction as F
from itertools import product
import unittest

import numpy as np

from scheduling import expected, lattice, mixture_event_probabilities, optimise, replay


def brute_partitions(indices, cap):
    """Generate every feasible ordered partition independently of the bit-lattice DP."""
    if not indices:
        yield ()
        return
    for assignment in product((False, True), repeat=len(indices)):
        first = tuple(i for i, take in zip(indices, assignment) if take)
        if not 1 <= len(first) <= cap:
            continue
        remaining = tuple(i for i in indices if i not in first)
        for tail in brute_partitions(remaining, cap):
            yield (sum(2**i for i in first), *tail)


def independent_cost(schedule, assignments, overhead, costs):
    out = 0
    for labels, probability in assignments.items():
        for batch in schedule:
            out += probability * (overhead + sum(c for i, c in enumerate(costs) if batch // 2**i % 2))
            if batch & labels != batch:
                break
    return out


def witness():
    assignments = {
        a + 2 * (1 - a) + 4 * b + 8 * b: F(1, 4)
        for a, b in product((0, 1), repeat=2)
    }
    law1 = {s: F(9, 10) * assignments.get(s, 0) + F(1, 160) for s in range(16)}
    law2 = {s: law1[(s & 3) * 4 + (s >> 2)] for s in range(16)}
    schedules = list(brute_partitions(range(4), 2))
    costs = (1, 1, 1, 1)
    risks = [[independent_cost(s, law, 2, costs) for s in schedules] for law in (law1, law2)]
    opt = [min(r) for r in risks]
    optimal_first = [{s[0] for s, val in zip(schedules, risk) if val == floor}
                     for risk, floor in zip(risks, opt)]
    mean = [(a + b) / 2 for a, b in zip(*risks)]
    return law1, law2, opt, optimal_first, min(mean) - (opt[0] + opt[1]) / 2


class SchedulingTests(unittest.TestCase):
    def test_dp_against_exhaustive_expected_cost(self):
        rng = np.random.default_rng(1072)
        trials = 0
        for n in (2, 3, 4, 5):
            cost = rng.integers(1, 25, n)
            for cap in (1, 2, n):
                partitions = tuple(brute_partitions(range(n), cap))
                for _ in range(3):
                    weights = rng.integers(1, 100, 1 << n)
                    prob = weights / weights.sum()
                    assignments = dict(enumerate(prob))
                    q = np.array([sum(v for bits, v in assignments.items() if bits & mask == mask)
                                  for mask in range(1 << n)])
                    overhead = int(rng.integers(0, 15))
                    output, value = optimise(q, cost, overhead, cap)
                    self.assertAlmostEqual(value, min(independent_cost(s, assignments, overhead, cost)
                                                      for s in partitions), places=9)
                    self.assertAlmostEqual(value, expected(output, q, cost, overhead), places=9)
                    for labels in assignments:
                        self.assertEqual(replay(output, labels, cost, overhead, cap)[0], int(labels == (1 << n) - 1))
                    trials += 1
        self.assertEqual(trials, 36)

    def test_marginals_match_and_mass_is_valid(self):
        membership, _ = lattice(4, 2)
        p = np.array([.06, .2, .6, .97])
        q = mixture_event_probabilities(p, .95, membership)
        np.testing.assert_allclose(q[1 << np.arange(4)], p)
        self.assertAlmostEqual(q[0], 1)
        for i in range(16):
            for j in range(16):
                if i & j == i:
                    self.assertGreaterEqual(q[i] + 1e-12, q[j])

    def test_witness_pooled_regret(self):
        a, b, opt, best, regret = witness()
        for law in (a, b):
            self.assertEqual(sum(law.values()), 1)
            self.assertEqual([sum(v for s, v in law.items() if s & (1 << i)) for i in range(4)], [F(1, 2)] * 4)
        self.assertEqual(best, [{3}, {12}])
        self.assertEqual(opt, [F(41, 10), F(41, 10)])
        self.assertGreater(regret, 0)
        print(f"Exact common-schedule regret lower bound: {regret}")


if __name__ == "__main__":
    unittest.main()
