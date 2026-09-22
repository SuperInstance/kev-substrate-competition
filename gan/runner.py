"""Runner — generations end to end: mutate -> predict -> measure -> chain.

Each candidate is one BIND cell carrying BOTH the receipted prediction and
the measured speedup, so harness/replay.py scores the critic's belief with
no extra machinery (R2.2). Selection runs on MEASURED cost only — the
predictor advises, the measurement decides.

The expected finding, on the record: MockBench.predict has a deliberate
blind spot (no unroll-spill penalty), so the critic over-loves unroll>4.
The lineage stops proposing it within a few generations anyway, because
selection reads the measurement, not the belief. Receipts don't just verify
the run — they expose the critic. That is the thesis in miniature.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from harness.gates import run_gates
from harness.hashutil import GENESIS_PREV_HASH, cell_hash
from harness.replay import verify_log
from harness.score import composite, field_bests
from harness.witness import WitnessCell
from .mockbench import MockBench
from .population import Candidate, Population
from .qd import QDArchive


@dataclass
class GenRecord:
    generation: int
    best_id: str
    best_cost: float
    best_speedup: float
    critic_mse: Optional[float]
    cells: int


@dataclass
class RunReport:
    seed: int
    generations: List[GenRecord]
    final_score: Optional[float]
    coherence: float
    witness_path: Optional[str] = None

    def curve(self) -> List[float]:
        return [g.best_speedup for g in self.generations]


class Runner:
    def __init__(self, seed: int, pop_size: int = 8, bench: MockBench = None):
        self.seed = seed
        self.pop = Population(seed, pop_size)
        self.bench = bench or MockBench()
        self.ref_cost = self.bench.cost(
            {"tile_m": 32, "tile_n": 32, "unroll": 2, "l2_promote": 0})
        self.conv_id = f"kev-gan:seed{seed}"
        self.cells: List[WitnessCell] = []
        self._cell_seq = 0

    def _emit(self, op: str, payload: dict, metrics=None, prediction=None) -> WitnessCell:
        self._cell_seq += 1
        state_json = json.dumps({"op": op, "payload": payload})
        answers_json = "[]"
        prev = self.cells[-1].cell_hash if self.cells else GENESIS_PREV_HASH
        cid = f"{self.conv_id}:cell-{self._cell_seq}"
        ch = cell_hash(cid, state_json, answers_json, prev)
        cell = WitnessCell(conv_id=self.conv_id, cell_id=cid, tick=self._cell_seq,
                           state_json=state_json, answers_json=answers_json,
                           prev_hash=prev, cell_hash=ch,
                           metrics=metrics or {}, prediction=prediction)
        self.cells.append(cell)
        return cell

    def _evaluate(self, c: Candidate) -> Candidate:
        cost = self.bench.cost(c.genome)
        c.measured_cost = cost
        c.measured_speedup = self.ref_cost / cost
        cell = self._emit(
            "BIND",
            {"kind": "candidate", "id": c.id, "generation": c.generation,
             "genome": c.genome, "parent": c.parent_hash,
             "hypothesis": c.hypothesis, "bench": self.bench.name,
             "bench_kind": self.bench.kind,
             # measurements ride INSIDE the hashed payload — a result
             # forgery must break the chain, or the referee only proves
             # existence, not value.
             "metrics": {"measured_speedup": round(c.measured_speedup, 6),
                         "cost": round(cost, 6)},
             "prediction": {"predicted_speedup": round(c.predicted_speedup, 6),
                            "hypothesis": c.hypothesis}},
            metrics={"measured_speedup": round(c.measured_speedup, 6),
                     "cost": round(cost, 6)},
            prediction={"predicted_speedup": round(c.predicted_speedup, 6),
                        "hypothesis": c.hypothesis},
        )
        c.witness_cell_id = cell.cell_hash
        return c

    def run(self, generations: int = 4) -> RunReport:
        records: List[GenRecord] = []
        stagnant = 0
        archive = QDArchive()
        pop = [self._evaluate(c) for c in self.pop.genesis(self.bench)]
        for c in pop:
            archive.try_add(c)
        best_cost = min(c.measured_cost for c in pop)
        for gen in range(generations):
            rep = verify_log(self.cells)
            best = min(pop, key=lambda c: c.measured_cost)
            if best.measured_cost < best_cost - 1e-12:
                best_cost = best.measured_cost
                stagnant = 0
            else:
                stagnant += 1
            cov, cov_total = archive.coverage()
            self._emit("VIEW", {"generation": gen, "best_id": best.id,
                                "best_speedup": round(best.measured_speedup, 6),
                                "coverage": cov, "coverage_total": cov_total,
                                "qd_score": round(archive.qd_score(), 6),
                                "critic_mse": rep.critic_mse})
            records.append(GenRecord(
                generation=gen, best_id=best.id, best_cost=best.measured_cost,
                best_speedup=best.measured_speedup,
                critic_mse=rep.critic_mse, cells=rep.total))
            if gen == generations - 1:
                break
            elite = min(pop, key=lambda c: c.measured_cost)   # true elitism:
            survivor = self.pop.tournament(pop, k=3)          # the best stays,
            heat = 1.0 if stagnant < 2 else 3.0               # the search kicks
            pop = [elite] + [
                self._evaluate(self.pop.mutate(survivor, self.bench, gen + 1, heat))
                for _ in range(self.pop.size - 1)]
            for c in pop[1:]:
                archive.try_add(c)

        rep = verify_log(self.cells)
        run_gates(self.cells, rep)
        fields = [{"measured_speedup": r.best_speedup} for r in records]
        bests = field_bests(fields)
        from harness.replay import ReplayReport
        final = composite(
            ReplayReport(total=rep.total, verified=rep.verified, coherence=rep.coherence,
                         first_break_index=None, first_break_reason=None,
                         conv_chains={}, critic_mse=rep.critic_mse,
                         critic_pairs=rep.critic_pairs),
            {"measured_speedup": records[-1].best_speedup}, bests)
        return RunReport(seed=self.seed, generations=records,
                         final_score=final, coherence=rep.coherence)

    def write_witness(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            for c in self.cells:
                f.write(json.dumps({
                    "conv_id": c.conv_id, "cell_id": c.cell_id, "tick": c.tick,
                    "state_json": c.state_json, "answers_json": c.answers_json,
                    "prev_hash": c.prev_hash, "cell_hash": c.cell_hash,
                    "metrics": c.metrics, "prediction": c.prediction}) + "\n")
        return path


def run_generations(seed: int = 7, pop_size: int = 8, generations: int = 4,
                    witness_path: Optional[str] = None) -> RunReport:
    r = Runner(seed, pop_size)
    report = r.run(generations)
    if witness_path:
        report.witness_path = r.write_witness(witness_path)
    return report


if __name__ == "__main__":
    import sys
    gens = int(sys.argv[sys.argv.index("--gens") + 1]) if "--gens" in sys.argv else 4
    rep = run_generations(generations=gens, witness_path="/tmp/kev-gan-witness.jsonl")
    print(f"seed {rep.seed} · coherence {rep.coherence:.4f} · "
          f"final composite {rep.final_score:.4f}")
    for g in rep.generations:
        mse = f"{g.critic_mse:.4f}" if g.critic_mse is not None else "  n/a"
        print(f"gen {g.generation}: best {g.best_id} speedup {g.best_speedup:.4f} "
              f"(cost {g.best_cost:.4g}) · critic mse {mse} · {g.cells} cells")
