"""Witness log format — the receipt chain for every reported number.

A witness log is JSONL, one object per line. Line shape (all keys
required unless marked optional):

    {
      "conv_id":      "run-0007",          // chain grouping; new conv = genesis
      "cell_id":      "cell-3",            // unique within the log
      "tick":         3,                   // monotonically increasing per conv
      "state_json":   "{\"temperature\":0.6}",   // VERBATIM wire strings
      "answers_json": "[\"...\"]",
      "prev_hash":    "0x…",               // GENESIS value for conv starts
      "cell_hash":    "0x…",               // as computed by the substrate
      "metrics": {                          // optional but scored only if present
        "cells_per_second": 183000.0,
        "cost_per_m_tokens": 0.31,
        "power_watts": 165.0,
        "measured_speedup": 1.07           // the critic's target
      },
      "prediction": {                       // R2.2 — critic's RECEIPTED belief
        "predicted_speedup": 1.08,
        "hypothesis": "larger L2 tile"
      }
    }

F5 doctrine: metrics without a verifiable chain are withdrawals. The
witness log IS the chain; scoreable numbers live inside it or nowhere.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


class WitnessFormatError(ValueError):
    pass


@dataclass
class WitnessCell:
    conv_id: str
    cell_id: str
    tick: int
    state_json: str
    answers_json: str
    prev_hash: str
    cell_hash: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], line_no: int) -> "WitnessCell":
        required = ("conv_id", "cell_id", "tick", "state_json",
                    "answers_json", "prev_hash", "cell_hash")
        missing = [k for k in required if k not in d]
        if missing:
            raise WitnessFormatError(
                f"line {line_no}: missing keys {missing}; "
                f"metrics without a chain are withdrawals")
        if not isinstance(d["state_json"], str) or not isinstance(d["answers_json"], str):
            raise WitnessFormatError(
                f"line {line_no}: state_json/answers_json must be the verbatim "
                f"serialized strings, not objects")
        return cls(
            conv_id=str(d["conv_id"]),
            cell_id=str(d["cell_id"]),
            tick=int(d["tick"]),
            state_json=d["state_json"],
            answers_json=d["answers_json"],
            prev_hash=d["prev_hash"],
            cell_hash=d["cell_hash"],
            metrics=dict(d.get("metrics") or {}),
            prediction=d.get("prediction"),
        )


def read_log(path: str) -> Iterator[WitnessCell]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            import json
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise WitnessFormatError(f"line {i}: bad JSON: {e}") from e
            yield WitnessCell.from_dict(d, i)


def read_log_all(path: str) -> List[WitnessCell]:
    return list(read_log(path))
