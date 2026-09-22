"""Mock benchmark — the honest stand-in.

Cost model: a smooth, deterministic function of the genome with a real
optimum (tiling trade-off: too small = launch overhead, too large = cache
pressure; unroll helps until registers spill; l2_promote helps big tiles).
Seeded jitter makes the landscape non-trivial but reproducible.

`bench.predict(genome)` is the critic's belief BEFORE measurement. It is a
deliberately imperfect model (no cache term) so the witness carries a real
prediction-vs-measurement gap — the Brier blade needs something to cut.
"""

import math
from typing import Dict


class MockBench:
    name = "mock-gemm-v0"
    kind = "analytical-cost-model"

    def cost(self, genome: Dict[str, int]) -> float:
        tm = genome["tile_m"]
        tn = genome["tile_n"]
        ur = genome["unroll"]
        l2 = genome["l2_promote"]

        # work per block ~ tm*tn; efficiency peaks when tiles fit L1-ish
        launch = 4096.0 / (tm * tn)                      # small tiles: overhead
        spill = (tm * tn) / 4096.0                       # big tiles: cache pressure
        # register spill ADDS cost beyond unroll 4 (a penalty, not a
        # discount — the earlier clamp-to-0.4 version accidentally made
        # unroll=8 the optimum, defeating the whole blind-spot lesson)
        unroll = 1.0 + 0.35 * min(ur, 4) + (0.22 * max(0, ur - 4) ** 2)
        promote = 1.0 - (0.12 if (l2 and tm * tn >= 2048) else 0.0)
        return (launch + spill) * unroll * promote

    def predict(self, genome: Dict[str, int]) -> float:
        """Critic belief: same shape, MISSING the unroll-spill penalty —
        overconfident on unroll>4, exactly the miscalibration we want to
        catch in the witness receipts."""
        tm = genome["tile_m"]
        tn = genome["tile_n"]
        ur = genome["unroll"]
        l2 = genome["l2_promote"]
        launch = 4096.0 / (tm * tn)
        spill = (tm * tn) / 4096.0
        unroll = 1.0 + 0.35 * min(ur, 4)          # no spill penalty: blind spot
        promote = 1.0 - (0.12 if (l2 and tm * tn >= 2048) else 0.0)
        base = (launch + spill) * unroll * promote
        ref = self.cost({"tile_m": 32, "tile_n": 32, "unroll": 2, "l2_promote": 0})
        return ref / base                                       # speedup vs ref
