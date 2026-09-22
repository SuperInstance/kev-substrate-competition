"""Harness core tests — stdlib unittest, zero dependencies (F-no-one-else).

Run: python3 -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from harness import (GateFailure, ReplayReport, WitnessCell, canary_ok,
                     cell_hash, composite, field_bests, fnv1a64, hash16,
                     normalize_field, read_log_all, run_gates, verify_log)
from harness.hashutil import CANARY_VEC, GENESIS_PREV_HASH


def make_cell(conv, cid, tick, prev, answer='"a"', metrics=None, prediction=None):
    state = json.dumps({"temperature": 0.6})
    answers = f"[{answer}]"
    ch = cell_hash(cid, state, answers, prev)
    c = WitnessCell(conv_id=conv, cell_id=cid, tick=tick, state_json=state,
                    answers_json=answers, prev_hash=prev, cell_hash=ch,
                    metrics=metrics or {}, prediction=prediction)
    return c


class TestCanary(unittest.TestCase):
    def test_canary_numeric_match(self):
        # scout K #6: string forms differ by a leading zero; numeric is law
        self.assertTrue(canary_ok())
        self.assertEqual(fnv1a64(CANARY_VEC), 0x024A555471370B18D)
        self.assertEqual(hash16(CANARY_VEC), "0x24a555471370b18d")


class TestReplay(unittest.TestCase):
    def test_good_chain_verifies(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH)
        c1 = make_cell("run", "c1", 2, c0.cell_hash)
        rep = verify_log([c0, c1])
        self.assertEqual(rep.total, 2)
        self.assertEqual(rep.verified, 2)
        self.assertEqual(rep.coherence, 1.0)
        self.assertIsNone(rep.first_break_index)
        self.assertEqual(rep.conv_chains["run"], [c0.cell_hash, c1.cell_hash])

    def test_tampered_payload_breaks_and_measures_radius(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH)
        c1 = make_cell("run", "c1", 2, c0.cell_hash)
        c1.state_json = '{"temperature":0.9}'      # tamper, no re-hash
        rep = verify_log([c0, c1])
        self.assertEqual(rep.verified, 1)
        self.assertEqual(rep.coherence, 0.5)
        self.assertEqual(rep.first_break_index, 1)
        self.assertIn("cell_hash mismatch", rep.first_break_reason)

    def test_broken_link_invalidates_descendants_but_scans_on(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH)
        c1 = make_cell("run", "c1", 2, "0xdeadbeef")   # wrong link
        c2 = make_cell("run", "c2", 3, c1.cell_hash)   # chains to bad cell
        rep = verify_log([c0, c1, c2])
        self.assertEqual(rep.verified, 1)              # only c0
        self.assertEqual(rep.first_break_index, 1)

    def test_genesis_must_be_genesis(self):
        c0 = make_cell("run", "c0", 1, "0xnotgenesis")
        rep = verify_log([c0])
        self.assertEqual(rep.verified, 0)
        self.assertIn("non-genesis", rep.first_break_reason)

    def test_tick_rewind_rejected(self):
        c0 = make_cell("run", "c0", 5, GENESIS_PREV_HASH)
        c1 = make_cell("run", "c1", 3, c0.cell_hash)   # tick goes backwards
        rep = verify_log([c0, c1])
        self.assertEqual(rep.verified, 1)
        self.assertIn("tick", rep.first_break_reason)

    def test_multi_conv_independent_chains(self):
        a0 = make_cell("A", "a0", 1, GENESIS_PREV_HASH)
        b0 = make_cell("B", "b0", 1, GENESIS_PREV_HASH)
        a1 = make_cell("A", "a1", 2, a0.cell_hash)
        rep = verify_log([a0, b0, a1])
        self.assertEqual(rep.verified, 3)
        self.assertEqual(set(rep.conv_chains), {"A", "B"})


class TestCritic(unittest.TestCase):
    def test_receipted_prediction_scored(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH,
                       metrics={"measured_speedup": 1.05},
                       prediction={"predicted_speedup": 1.05,
                                   "hypothesis": "tile L2"})
        c1 = make_cell("run", "c1", 2, c0.cell_hash,
                       metrics={"measured_speedup": 1.00},
                       prediction={"predicted_speedup": 1.20})
        rep = verify_log([c0, c1])
        self.assertEqual(rep.critic_pairs, 2)
        self.assertAlmostEqual(rep.critic_mse, (0.0 + 0.04) / 2)

    def test_prediction_without_measurement_not_scored(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH,
                       prediction={"predicted_speedup": 1.5})
        rep = verify_log([c0])
        self.assertIsNone(rep.critic_mse)
        self.assertEqual(rep.critic_pairs, 0)


class TestScore(unittest.TestCase):
    def test_ratio_to_best_directions(self):
        self.assertEqual(normalize_field(200.0, 200.0), 1.0)
        self.assertEqual(normalize_field(100.0, 200.0), 0.5)
        self.assertEqual(normalize_field(0.5, 0.5, lower_is_better=True), 1.0)
        self.assertEqual(normalize_field(1.0, 0.5, lower_is_better=True), 0.5)

    def test_field_bests_mixed_directions(self):
        subs = [{"cells_per_second": 100.0, "power_watts": 300.0},
                {"cells_per_second": 200.0, "power_watts": 150.0}]
        self.assertEqual(field_bests(subs),
                         {"cells_per_second": 200.0, "power_watts": 150.0})

    def test_composite_withdraws_on_broken_chain(self):
        rep = ReplayReport(total=2, verified=1, coherence=0.5,
                           first_break_index=1,
                           first_break_reason="x", conv_chains={},
                           critic_mse=None, critic_pairs=0)
        self.assertIsNone(composite(rep, {"cells_per_second": 200.0},
                                    {"cells_per_second": 200.0}))

    def test_composite_full_marks(self):
        rep = ReplayReport(total=1, verified=1, coherence=1.0,
                           first_break_index=None, first_break_reason=None,
                           conv_chains={"run": ["h"]},
                           critic_mse=None, critic_pairs=0)
        s = composite(rep,
                      {"cells_per_second": 200.0, "power_watts": 150.0},
                      {"cells_per_second": 200.0, "power_watts": 150.0})
        self.assertAlmostEqual(s, 1.0)

    def test_composite_missing_field_withdraws(self):
        rep = ReplayReport(total=1, verified=1, coherence=1.0,
                           first_break_index=None, first_break_reason=None,
                           conv_chains={"run": ["h"]},
                           critic_mse=None, critic_pairs=0)
        self.assertIsNone(composite(rep, {"cells_per_second": 200.0},
                                    {"power_watts": 150.0}))


class TestGates(unittest.TestCase):
    def test_gate1_empty_log(self):
        rep = verify_log([])
        with self.assertRaises(GateFailure) as cm:
            run_gates([], rep)
        self.assertEqual(cm.exception.gate, 1)

    def test_gate1_broken_chain_names_index(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH)
        c1 = make_cell("run", "c1", 2, "0xbad")
        rep = verify_log([c0, c1])
        with self.assertRaises(GateFailure) as cm:
            run_gates([c0, c1], rep)
        self.assertIn("cell 1", cm.exception.reason)


class TestWitnessIO(unittest.TestCase):
    def test_roundtrip_and_verbatim_strings(self):
        c0 = make_cell("run", "c0", 1, GENESIS_PREV_HASH,
                       metrics={"cells_per_second": 42.0})
        d = {"conv_id": c0.conv_id, "cell_id": c0.cell_id, "tick": c0.tick,
             "state_json": c0.state_json, "answers_json": c0.answers_json,
             "prev_hash": c0.prev_hash, "cell_hash": c0.cell_hash,
             "metrics": c0.metrics}
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as f:
            f.write(json.dumps(d) + "\n")
            path = f.name
        try:
            cells = read_log_all(path)
            self.assertEqual(len(cells), 1)
            rep = verify_log(cells)
            self.assertEqual(rep.verified, 1)
        finally:
            os.unlink(path)

    def test_missing_key_is_format_error(self):
        from harness.witness import WitnessFormatError
        with self.assertRaises(WitnessFormatError):
            WitnessCell.from_dict({"conv_id": "x"}, 7)


if __name__ == "__main__":
    unittest.main()
