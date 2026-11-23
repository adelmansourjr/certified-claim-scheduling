import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("live_verifier", Path(__file__).with_name("study.py"))
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class TestCertificate(unittest.TestCase):
    def test_signed_extrema_at_all_prefix_vertices(self):
        rng = np.random.default_rng(42)
        vertices = [(1,)*n+(0,)*(3-n) for n in range(4)]
        for _ in range(80):
            weights = tuple(map(float,rng.normal(size=6)))
            observed = [m.linear(weights, a+b) for a,b in itertools.product(vertices,repeat=2)]
            low, high = m.interval_bounds(weights)
            self.assertAlmostEqual(min(observed),low)
            self.assertAlmostEqual(max(observed),high)

    def test_real_feature_bounds(self):
        doc = m.Document.make(["The category sold 23 items Tuesday.", "No French category sold 32 items.",
                               "Next they sold forty-two items in Canada."])
        atom = m.prepare(doc,"The sale of 23 items on Tuesday.",{},1,0)
        for name in m.BLOCKS:
            vec = m.feature(name,doc,atom)
            self.assertTrue(all(0<=x<=1+1e-14 for x in vec))
            for offset in (0,3):
                self.assertEqual(list(vec[offset:offset+3]), sorted(vec[offset:offset+3],reverse=True))

    def test_certified_equals_full_in_every_stage_order(self):
        doc = m.Document.make(["The show performed in France in 2026.", "The show ran through Spain.",
                               "There were one hundred attendees."])
        atom = m.prepare(doc,"In France the show performed in 2026.",{},1,0)
        rng = np.random.default_rng(832)
        for _ in range(20):
            weights = tuple(map(float,rng.normal(size=28)))
            bias = float(rng.normal())
            for order in m.ORDERS:
                base, blocks, bounds = m.compiled(weights,order)
                eager = m.predict_atom(doc,atom,bias,base,blocks,0,order,bounds,False,{"certificates":0},
                                       dict.fromkeys(m.BLOCKS,0))
                lazy = m.predict_atom(doc,atom,bias,base,blocks,0,order,bounds,True,{"certificates":0},
                                      dict.fromkeys(m.BLOCKS,0))
                self.assertEqual(lazy,eager)


if __name__ == "__main__":
    unittest.main()
