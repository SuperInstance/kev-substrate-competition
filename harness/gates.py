"""Hard gates — pass/fail before any scoring happens.

Gate #0 (canary): the harness's hash function agrees with the substrate
on the pinned vector. If this fails, every later verdict is void.
Gate #1 (replay): the submission's witness log verifies end to end.
Broken chain = withdrawal (F5), not a low score.
"""

from typing import List

from .hashutil import canary_ok
from .replay import ReplayReport
from .witness import WitnessCell


class GateFailure(RuntimeError):
    def __init__(self, gate: int, reason: str):
        super().__init__(f"gate #{gate} failed: {reason}")
        self.gate = gate
        self.reason = reason


def run_gates(cells: List[WitnessCell], report: ReplayReport) -> None:
    if not canary_ok():
        raise GateFailure(0, "canary hash drift — harness disagrees with "
                             "substrate; re-pin before scoring anything")
    if report.total == 0:
        raise GateFailure(1, "empty witness log; a metric without a chain "
                             "is a withdrawal (F5)")
    if report.verified != report.total:
        raise GateFailure(
            1,
            f"chain breaks at cell {report.first_break_index}: "
            f"{report.first_break_reason} "
            f"({report.verified}/{report.total} verified, "
            f"coherence {report.coherence:.4f})")
