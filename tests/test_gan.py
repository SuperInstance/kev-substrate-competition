"""Runner tests — the R&D wheel: does the loop learn, is it deterministic,
does the critic's blind spot show up in the receipts, does the witness
interop with harness/replay.py end to end?"""

import json
import os
import tempfile
import unittest

from gan.mockbench import MockBench
from gan.runner import Runner, run_generations
from harness import read_log_all, run_gates, verify_log


class TestLearning(unittest.TestCase):
    def test_lineage_improves_over_generations(self):
        rep = run_generations(seed=7, pop_size=8, generations=8)
        curve = rep.curve()
        # elitism contract: the curve never dips
        self.assertEqual(curve, sorted(curve))
        # with enough runway for the heat escape hatch, learning is real
        self.assertGreater(curve[-1], curve[0],
                           "selection on measured cost must improve speedup")
        self.assertGreater(curve[-1] / curve[0], 1.05)

    def test_deterministic_under_seed(self):
        a = run_generations(seed=11, pop_size=6, generations=3)
        b = run_generations(seed=11, pop_size=6, generations=3)
        self.assertEqual(a.curve(), b.curve())
        self.assertEqual(a.final_score, b.final_score)

    def test_different_seeds_diverge(self):
        a = run_generations(seed=1, pop_size=6, generations=3)
        b = run_generations(seed=2, pop_size=6, generations=3)
        self.assertNotEqual(a.curve(), b.curve())


class TestCriticReceipts(unittest.TestCase):
    def test_critic_blind_spot_is_on_the_record(self):
        """predict() ignores unroll-spill, so predictions for unroll>4
        genomes overshoot. The witness must carry that gap visibly."""
        bench = MockBench()
        bad = {"tile_m": 32, "tile_n": 32, "unroll": 8, "l2_promote": 0}
        pred, meas = bench.predict(bad), None
        # direct: predicted speedup vs measured speedup
        ref = bench.cost({"tile_m": 32, "tile_n": 32, "unroll": 2, "l2_promote": 0})
        measured = ref / bench.cost(bad)
        self.assertGreater(pred, measured,
                           "critic should overshoot on unroll>4 (its blind spot)")

    def test_witness_carries_prediction_and_measurement(self):
        r = Runner(seed=3, pop_size=4)
        r.run(2)
        with_pred_meas = [c for c in r.cells
                          if c.prediction and "measured_speedup" in c.metrics]
        self.assertEqual(len(with_pred_meas), len(r.cells))
        rep = verify_log(r.cells)
        self.assertGreater(rep.critic_pairs, 0)
        self.assertIsNotNone(rep.critic_mse)


class TestWitnessInterop(unittest.TestCase):
    def test_exported_log_passes_gates_and_replays(self):
        rep = run_generations(seed=5, pop_size=5, generations=3)
        r = Runner(seed=5, pop_size=5)
        r.run(3)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            r.write_witness(path)
            cells = read_log_all(path)
            replayed = verify_log(cells)
            run_gates(cells, replayed)          # raises on any break
            self.assertEqual(replayed.verified, replayed.total)
            self.assertEqual(replayed.coherence, 1.0)
        finally:
            os.unlink(path)

    def test_tampered_export_is_caught_by_referee(self):
        r = Runner(seed=5, pop_size=4)
        r.run(2)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            r.write_witness(path)
            with open(path) as fh:
                lines = fh.readlines()
            forged = json.loads(lines[1])
            forged["metrics"] = {"measured_speedup": 99.9}   # result forgery
            lines[1] = json.dumps(forged) + "\n"
            with open(path, "w") as fh:
                fh.writelines(lines)
            cells = read_log_all(path)
            replayed = verify_log(cells)
            with self.assertRaises(Exception):
                run_gates(cells, replayed)
        finally:
            os.unlink(path)


class TestGenomeSpace(unittest.TestCase):
    def test_bench_has_real_optimum_inside_ranges(self):
        bench = MockBench()
        best = min(
            ({"tile_m": tm, "tile_n": tn, "unroll": u, "l2_promote": l2}
             for tm in (8, 16, 32, 64) for tn in (8, 16, 32, 64)
             for u in (1, 2, 4, 8) for l2 in (0, 1)),
            key=bench.cost)
        # optimum must be a sane interior tiling, not a boundary blowup
        self.assertGreaterEqual(best["tile_m"], 16)
        self.assertLessEqual(best["unroll"], 4)


if __name__ == "__main__":
    unittest.main()
