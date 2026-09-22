"""QD archive — MAP-Elites over the population, because a single best
candidate is a monoculture and monocultures collapse.

Feature dims (mock): tile_m bucket, unroll bucket. Each archive slot keeps
the best-measured occupant. The composite gains a QD term — coverage and
QD-score are first-class alongside the single best, per the fleet's
long-standing breeding doctrine (diversity is not decoration).
"""

from typing import Dict, List, Optional, Tuple

from .population import Candidate


def _bucket(value: int, edges: List[int]) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


TM_EDGES = [16, 32, 64]       # buckets: <=16, 17-32, 33-64, >64
UR_EDGES = [2, 4]             # buckets: <=2, 3-4, >4


class QDArchive:
    def __init__(self):
        self.slots: Dict[Tuple[int, int], Candidate] = {}

    def features(self, c: Candidate) -> Tuple[int, int]:
        g = c.genome
        return (_bucket(g["tile_m"], TM_EDGES), _bucket(g["unroll"], UR_EDGES))

    def try_add(self, c: Candidate) -> bool:
        if c.measured_cost is None:
            return False
        f = self.features(c)
        cur = self.slots.get(f)
        if cur is None or c.measured_cost < cur.measured_cost:
            self.slots[f] = c
            return True
        return False

    def coverage(self) -> Tuple[int, int]:
        return len(self.slots), (len(TM_EDGES) + 1) * (len(UR_EDGES) + 1)

    def qd_score(self) -> float:
        """Sum of per-slot speedup vs a naive reference genome."""
        ref = 1.0 / 4.0   # placeholder replaced by runner's ref speedup
        return sum(1.0 / c.measured_cost for c in self.slots.values())

    def best_per_slot(self) -> List[dict]:
        return [{"features": f, "id": c.id, "cost": c.measured_cost}
                for f, c in sorted(self.slots.items())]
