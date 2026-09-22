"""Composite scoring — F1 fixed: no unit mixing, ever.

normalize_field: each term divided by the field BEST (ratio-to-best ∈ (0,1]).
composite: coherence × product of ratio-to-best terms. Higher is better;
a submission at the field best on every axis with a clean chain scores 1.0.

This replaces the spec's `cost_per_M + power_draw` sum, which added
dollars to watts (PLAYTEST-GAN.md F1). Higher-is-better is kept so the
leaderboard reads naturally; cost and power are inverted (best = lowest).
"""

from typing import Dict, List, Mapping, Optional

from .replay import ReplayReport

# Fields where lower raw values are better (inverted before ratio-to-best).
LOWER_IS_BETTER = frozenset({"cost_per_m_tokens", "power_watts", "tt_1p2x_hours"})


def normalize_field(raw: float, best: float, lower_is_better: bool = False) -> float:
    """Ratio-to-best in (0,1]. 1.0 == you ARE the field best."""
    if best <= 0:
        return 0.0
    r = (best / raw) if lower_is_better else (raw / best)
    # clamp: floating noise must never push a ratio past the boundary
    return max(0.0, min(1.0, r))


def composite(report: ReplayReport,
              fields: Mapping[str, float],
              bests: Mapping[str, float]) -> Optional[float]:
    """One submission's score, or None if the chain is broken (F5/F2:
    an unverifiable submission is a withdrawal, not a zero — zeros can
    be luck, withdrawals are verdicts)."""
    if report.total == 0 or report.verified != report.total:
        return None
    score = report.coherence
    for f, raw in fields.items():
        best = bests.get(f)
        if best is None or raw is None:
            return None            # missing term = incomplete = withdrawal
        score *= normalize_field(float(raw), float(best),
                                 lower_is_better=(f in LOWER_IS_BETTER))
    return score


def field_bests(submissions: List[Mapping[str, float]]) -> Dict[str, float]:
    """Field best per metric across submissions. For LOWER_IS_BETTER the
    best is the minimum; otherwise the maximum."""
    bests: Dict[str, float] = {}
    for sub in submissions:
        for f, v in sub.items():
            v = float(v)
            if f not in bests:
                bests[f] = v
            elif f in LOWER_IS_BETTER:
                bests[f] = min(bests[f], v)
            else:
                bests[f] = max(bests[f], v)
    return bests
