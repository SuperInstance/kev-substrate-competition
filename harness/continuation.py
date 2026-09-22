"""Continuation check — the held-out half of F3, referee-side.

Host keeps a slice of the witness log private. After submission, the host
runs verify_continuation: merge public + held-out by tick and replay the
union. Held-out cells complete the chain — their prev_hash pointers reach
back into public cells — so a clean full replay proves both their validity
AND their linkage. A submission that overfits the public half cannot forge
the continuation (chaining material was spent in public); a splice from a
different run collapses inside verify_log itself. Host discipline still
required to KEEP the half private; the referee enforces the shape.
"""

from typing import List, Sequence, Tuple

from .witness import WitnessCell
from .replay import verify_log


def split_log(cells: Sequence[WitnessCell], every: int = 2,
              offset: int = 1) -> Tuple[List[WitnessCell], List[WitnessCell]]:
    """Deterministic public/held-out split: every `every`-th cell (by tick,
    offset into the stride) is held out. Verifiable by anyone who knows
    the rule; unpredictable enough to punish overfitting-to-public."""
    public, held = [], []
    for c in cells:
        (held if (c.tick + offset) % every == 0 else public).append(c)
    return public, held


def verify_continuation(public: Sequence[WitnessCell],
                        held_out: Sequence[WitnessCell]) -> Tuple[bool, str]:
    """Held-out cells must COMPLETE the chain: merged-by-tick replay of
    public ∪ held-out verifies every cell, and every held-out cell is
    present in that verification."""
    merged = sorted(list(public) + list(held_out), key=lambda c: c.tick)
    if not merged:
        return False, "empty logs"
    rep = verify_log(merged)
    if rep.verified != rep.total:
        return False, (f"merged replay broke at cell {rep.verified + 1}/"
                       f"{rep.total} — held-out does not complete the chain "
                       "(leak-shaped: overfit to public, or spliced from "
                       "another run)")
    held_ids = {c.cell_id for c in held_out}
    present = sum(1 for c in merged if c.cell_id in held_ids)
    if present != len(held_out):
        return False, "held-out cells missing from verified chain"
    return True, (f"continuation ok: {len(held_out)} held-out cells complete "
                  f"a {rep.total}-cell chain, coherence {rep.coherence:.4f}")
