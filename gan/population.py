"""Population — seeded, hypothesis-carrying mutation with receipts.

Every mutation carries a one-line hypothesis (anti-slop discipline, R2).
Every candidate remembers its lineage as the parent's cell hash — the
lineage graph IS the substrate animation, per PLAYTEST-GAN.md.

Genome space (mock-GEMM stand-in): tile_m, tile_n, unroll, l2_promote.
The real runner swaps these for PTX/Mojo/TRT/CK source strings; the
selection machinery does not care what the genome IS, only that mutate()
is seeded and receipted.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Candidate:
    genome: Dict[str, int]
    parent_hash: Optional[str]     # lineage: the cell that spawned this one
    hypothesis: str                # one line, mandatory — no anonymous mutations
    generation: int
    id: str
    predicted_speedup: float       # receipted BEFORE measurement (R2.2)
    measured_cost: Optional[float] = None
    witness_cell_id: Optional[str] = None


class Population:
    GENOME_RANGES = {
        "tile_m": (8, 128),
        "tile_n": (8, 128),
        "unroll": (1, 8),
        "l2_promote": (0, 1),
    }

    def __init__(self, seed: int, size: int):
        self.rng = random.Random(seed)
        self.size = size
        self._seq = 0

    def genesis(self, bench) -> List[Candidate]:
        """Generation 0: seeded random genomes, no parent, honest hypotheses."""
        pop = []
        for _ in range(self.size):
            g = {k: self.rng.randint(*v) for k, v in self.GENOME_RANGES.items()}
            pop.append(self._candidate(g, None, "genesis: uniform sample", 0, bench))
        return pop

    def _candidate(self, genome, parent_hash, hypothesis, generation, bench,
                   predicted=None) -> Candidate:
        self._seq += 1
        c = Candidate(
            genome=dict(genome),
            parent_hash=parent_hash,
            hypothesis=hypothesis,
            generation=generation,
            id=f"cand-{self._seq}",
            predicted_speedup=predicted if predicted is not None
                              else bench.predict(genome),
        )
        return c

    def mutate(self, survivor: Candidate, bench, generation: int,
               heat: float = 1.0) -> Candidate:
        """One seeded mutation step from a survivor, hypothesis mandatory.
        heat scales the step size — the runner raises it when the best
        stagnates, so plateaus get kicked rather than worshipped."""
        g = dict(survivor.genome)
        key = self.rng.choice(list(g))
        lo, hi = self.GENOME_RANGES[key]
        span = hi - lo
        delta = self.rng.choice([-1, -1, 1, 1]) * max(1, int(span * 0.08 * heat))
        g[key] = max(lo, min(hi, g[key] + delta))
        direction = "up" if delta > 0 else "down"
        hypo = (f"{key} {direction} {abs(delta)} (parent {survivor.id} "
                f"cost {survivor.measured_cost:.4g}; heat {heat:g})")
        return self._candidate(g, survivor.witness_cell_id, hypo, generation, bench)

    def tournament(self, measured: List[Candidate], k: int = 3) -> Candidate:
        """Deterministic tournament over MEASURED candidates (lower cost wins)."""
        pool = self.rng.sample(measured, min(k, len(measured)))
        return min(pool, key=lambda c: c.measured_cost)
