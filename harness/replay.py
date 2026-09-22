"""Chain replay — the witness log's referee.

verify_log re-derives every cell hash and every chain link from the
verbatim wire strings. A cell counts as verified iff:
  - its cell_hash recomputes (payload untampered)
  - its prev_hash matches the prior verified cell's cell_hash in the
    same conversation (chain unbroken)
  - its tick is strictly greater than the prior tick (order unrewound)

Coherence (F2) = verified / total. A log that fails replay has coherence
< 1, and the composite score multiplies by it — the literal optimum is
no longer maximum incoherent throughput.

Critic scoring (R2.2): for cells carrying both prediction.predicted_speedup
and metrics.measured_speedup, replay computes mean squared error. The
discriminator grades the belief as formally as the act.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hashutil import GENESIS_PREV_HASH, cell_hash, hash_int
from .witness import WitnessCell, WitnessFormatError


@dataclass
class ReplayReport:
    total: int
    verified: int
    coherence: float                      # verified / total (1.0 if empty)
    first_break_index: Optional[int]      # 0-based cell index of first failure
    first_break_reason: Optional[str]
    conv_chains: Dict[str, List[str]]     # conv_id -> verified cell hashes in order
    critic_mse: Optional[float]           # belief error over cells with both fields
    critic_pairs: int

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "verified": self.verified,
            "coherence": round(self.coherence, 6),
            "first_break_index": self.first_break_index,
            "first_break_reason": self.first_break_reason,
            "conversations": len(self.conv_chains),
            "critic_mse": (round(self.critic_mse, 6)
                           if self.critic_mse is not None else None),
            "critic_pairs": self.critic_pairs,
        }


def verify_log(cells: List[WitnessCell]) -> ReplayReport:
    conv_last: Dict[str, dict] = {}   # conv_id -> {"hash": str, "tick": int}
    conv_chains: Dict[str, List[str]] = {}
    total = 0
    verified = 0
    first_break_index: Optional[int] = None
    first_break_reason: Optional[str] = None
    sq_err = 0.0
    critic_pairs = 0

    for idx, c in enumerate(cells):
        total += 1
        reasons = []
        recomputed = cell_hash(c.cell_id, c.state_json, c.answers_json, c.prev_hash)
        if hash_int(recomputed) != hash_int(c.cell_hash):
            reasons.append("cell_hash mismatch (payload tampered)")
        # FIELD/PAYLOAD CROSS-CHECK: metrics and prediction must match the
        # HASHED payload, or a forger rewrites results without breaking
        # the chain (caught by the runner's own tamper test, 2026-09-22).
        import json as _json
        try:
            body = _json.loads(c.state_json)
            pl = body.get("payload", {}) if isinstance(body, dict) else {}
            if "metrics" in pl and pl["metrics"] != c.metrics:
                reasons.append("metrics field disagrees with hashed payload")
            if "prediction" in pl and pl["prediction"] != c.prediction:
                reasons.append("prediction field disagrees with hashed payload")
        except (ValueError, AttributeError):
            pass  # state_json shape is the format layer's job, not replay's
        prior = conv_last.get(c.conv_id)
        if prior is None:
            if c.prev_hash != GENESIS_PREV_HASH:
                reasons.append("genesis cell with non-genesis prev_hash")
        else:
            if c.prev_hash != prior["hash"]:
                reasons.append("prev_hash does not chain to prior cell")
            if c.tick <= prior["tick"]:
                reasons.append("tick not increasing (order rewound)")
        if reasons:
            if first_break_index is None:
                first_break_index = idx
                first_break_reason = "; ".join(reasons)
            # chain state does NOT advance on a broken cell — everything
            # after an unverified link is unverifiable, but we keep
            # scanning so coherence measures the blast radius.
            continue
        verified += 1
        conv_last[c.conv_id] = {"hash": c.cell_hash, "tick": c.tick}
        conv_chains.setdefault(c.conv_id, []).append(c.cell_hash)

        if c.prediction and "predicted_speedup" in c.prediction:
            meas = c.metrics.get("measured_speedup")
            if meas is not None:
                d = float(c.prediction["predicted_speedup"]) - float(meas)
                sq_err += d * d
                critic_pairs += 1

    coherence = (verified / total) if total else 1.0
    mse = (sq_err / critic_pairs) if critic_pairs else None
    return ReplayReport(
        total=total, verified=verified, coherence=coherence,
        first_break_index=first_break_index,
        first_break_reason=first_break_reason,
        conv_chains=conv_chains,
        critic_mse=mse, critic_pairs=critic_pairs,
    )
