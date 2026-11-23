"""Exact schedules for a conjunction of finitely many costly checks."""
from functools import lru_cache
from itertools import combinations
import numpy as np


@lru_cache(maxsize=None)
def lattice(n, capacity):
    """Precompute bit tests, feasible batches and successor actions."""
    full = (1 << n) - 1
    integers = np.arange(full + 1)
    membership = (integers[:, None] & (1 << np.arange(n))) != 0
    batches = tuple(
        sum(1 << i for i in its)
        for width in range(1, min(n, capacity) + 1)
        for its in combinations(range(n), width)
    )
    available = tuple(tuple(b for b in batches if not (b & s)) for s in range(full))
    return membership, available


def product_event_probabilities(p, membership):
    return np.prod(np.where(membership, p, 1.0), axis=1)


def mixture_event_probabilities(p, delta, membership):
    """Symmetric two-state latent law with the same subclaim marginals as p."""
    offsets = delta * np.minimum(p, 1 - p)
    return (product_event_probabilities(p + offsets, membership)
            + product_event_probabilities(p - offsets, membership)) / 2


def optimise(q, item_costs, overhead, capacity):
    """Sunk-probability-weighted DP. q[s] = P(all checks in s pass)."""
    membership, available = lattice(len(item_costs), capacity)
    full = len(membership) - 1
    batch_costs = overhead + membership @ item_costs
    values = np.zeros(full + 1)
    choices = np.zeros(full, dtype=np.int64)
    for s in range(full - 1, -1, -1):
        best = min(available[s], key=lambda b: q[s] * batch_costs[b] + values[s | b])
        values[s] = q[s] * batch_costs[best] + values[s | best]
        choices[s] = best
    schedule = []
    s = 0
    while s < full:
        batch = int(choices[s])
        schedule.append(batch)
        s |= batch
    return tuple(schedule), values[0]


def fixed_schedule(order, width):
    return tuple(sum(1 << i for i in order[start:start + width])
                 for start in range(0, len(order), width))


def replay(schedule, label_mask, item_costs, overhead, capacity):
    """Stop at a witnessed failed check. Positive verdict requires all checks."""
    membership, _ = lattice(len(item_costs), capacity)
    full = len(membership) - 1
    seen = 0
    cost = 0.0
    checks = 0
    for rounds, batch in enumerate(schedule, 1):
        if not batch or batch & seen or batch >> len(item_costs) or batch.bit_count() > capacity:
            raise ValueError("Invalid batch: repeated, out-of-bounds or over capacity")
        cost += overhead + float(membership[batch] @ item_costs)
        checks += batch.bit_count()
        seen |= batch
        if label_mask & batch != batch:
            return 0, cost, rounds, checks
    if seen != full:
        raise ValueError("Positive certificate needs all checks")
    return 1, cost, rounds, checks


def expected(schedule, q, item_costs, overhead):
    seen = 0
    result = 0
    for batch in schedule:
        result += q[seen] * (overhead + sum(c for i, c in enumerate(item_costs) if batch & (1 << i)))
        seen |= batch
    return result
