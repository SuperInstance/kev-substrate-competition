"""Wheel round R6: QD archive + held-out continuation, tested."""

import unittest

from gan.population import Population
from gan.qd import QDArchive
from gan.mockbench import MockBench
from gan.runner import Runner, run_generations
from harness import split_log, verify_continuation


class TestQDArchive(unittest.TestCase):
    def _measured(self, genome, rid):
        bench = MockBench()
        c = type("C", (), {})()
        c.genome = genome
        c.measured_cost = bench.cost(genome)
        c.id = rid
        c.witness_cell_id = None
        c.predicted_speedup = 1.0
        return c

    def test_coverage_grows_with_diverse_population(self):
        arch = QDArchive()
        pop = Population(seed=1, size=40)
        bench = MockBench()
        n = 0
        for c in pop.genesis(bench):
            c.measured_cost = bench.cost(c.genome)
            if arch.try_add(c):
                n += 1
        cov, total = arch.coverage()
        self.assertGreaterEqual(cov, 4, "diverse genesis should fill several slots")
        self.assertEqual(total, 12)
        self.assertGreater(arch.qd_score(), 0.0)

    def test_slot_keeps_best_not_latest(self):
        arch = QDArchive()
        bench = MockBench()
        good = self._measured({"tile_m": 30, "tile_n": 32, "unroll": 2, "l2_promote": 0}, "good")
        bad = self._measured({"tile_m": 32, "tile_n": 32, "unroll": 2, "l2_promote": 0}, "bad")
        bad.measured_cost = good.measured_cost * 2
        self.assertTrue(arch.try_add(good))
        self.assertFalse(arch.try_add(bad))
        f = arch.features(good)
        self.assertEqual(arch.slots[f].id, "good")


class TestContinuation(unittest.TestCase):
    def test_split_and_verify_roundtrip(self):
        r = Runner(seed=9, pop_size=5)
        r.run(3)
        public, held = split_log(r.cells, every=2)
        self.assertEqual(len(public) + len(held), len(r.cells))
        ok, msg = verify_continuation(public, held)
        self.assertTrue(ok, msg)

    def test_spliced_continuation_fails(self):
        """A cheater stitches their held-out to a DIFFERENT public chain's
        head — the junction must collapse."""
        ra = Runner(seed=9, pop_size=4)
        ra.run(2)
        rb = Runner(seed=10, pop_size=4)
        rb.run(2)
        public, _ = split_log(ra.cells, every=2)
        _, held = split_log(rb.cells, every=2)
        ok, msg = verify_continuation(public, held)
        self.assertFalse(ok)
        self.assertIn("does not complete", msg)

    def test_forged_held_out_fails(self):
        r = Runner(seed=9, pop_size=4)
        r.run(2)
        public, held = split_log(r.cells, every=2)
        forged = type("F", (), dict(held[0].__dict__))()
        forged.metrics = {"measured_speedup": 99.9}
        ok, msg = verify_continuation(public, [forged] + list(held[1:]))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
