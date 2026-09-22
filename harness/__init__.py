"""kev-GAN harness — witness-verified scoring core.

Implements PLAYTEST-GAN.md F1/F2/F4/F5 and R2.2:
  F1  ratio-to-best normalization (no unit mixing)
  F2  coherence coefficient from witness replay, multiplied into the score
  F4  hashing time is part of the cell; no unpriced exclusions
  F5  a metric without a witness chain is a withdrawal, not a submission
  R2.2 critic predictions ride in the witness; replay scores belief vs act
"""
from .hashutil import (fnv1a64, hash16, hash_int, cell_hash,
                       CANARY_VEC, CANARY_EXPECTED, GENESIS_PREV_HASH, canary_ok)
from .witness import WitnessCell, WitnessFormatError, read_log, read_log_all
from .replay import verify_log, ReplayReport
from .score import composite, normalize_field, field_bests, LOWER_IS_BETTER
from .gates import run_gates, GateFailure
from .continuation import split_log, verify_continuation

__all__ = [
    "fnv1a64", "hash16", "hash_int", "cell_hash",
    "CANARY_VEC", "CANARY_EXPECTED", "GENESIS_PREV_HASH", "canary_ok",
    "WitnessCell", "WitnessFormatError", "read_log", "read_log_all",
    "verify_log", "ReplayReport",
    "composite", "normalize_field", "field_bests", "LOWER_IS_BETTER",
    "run_gates", "GateFailure", "split_log", "verify_continuation",
]
