"""Hash utilities, byte-aligned with kev/substrate.py.

kev-substrate computes cell hashes as:
    _fnv1a64(s)      — standard FNV-1a 64-bit
    _hash16(s)       — "0x" + format(h, "016x")
    genesis prev_hash — "0x" + "0" * 16
    cell_hash        — _hash16(cell_id + state_json + answers_json + prev_hash)

Gotcha (scout K #6): the docs pin the canary as 0x024a555471370b18d
(17 hex digits) but format(h, "016x") never emits a leading zero, so the
Python string is 0x24a555471370b18d. All comparisons here are NUMERIC;
never string-compare hashes.
"""

CANARY_VEC = "café Δ 日本語"
CANARY_EXPECTED = 0x024A555471370B18D  # numeric; both pinned spellings agree
GENESIS_PREV_HASH = "0x" + "0" * 16


def fnv1a64(s: str) -> int:
    # UTF-8 BYTES, not code points: the pinned canary is the byte hash
    # (café Δ 日本語 → 0x24a555471370b18d, matching candor's JS Buffer
    # hash). An ord-per-codepoint loop produces 0x77ff2029b867f2b5 for
    # the same string and fails the pin. kev/substrate.py as grepped
    # shows `h ^= ord(ch)` — if that is really what runs, its canary
    # test cannot pass; flagged to that repo, harness follows the pin.
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def hash16(s: str) -> str:
    return "0x" + format(fnv1a64(s), "016x")


def hash_int(s: str) -> int:
    return fnv1a64(s)


def canary_ok() -> bool:
    """Gate #0: the harness agrees with the substrate on what hashing is."""
    return fnv1a64(CANARY_VEC) == CANARY_EXPECTED


def cell_hash(cell_id: str, state_json: str, answers_json: str, prev_hash: str) -> str:
    """Byte-aligned with kev/substrate.py line: _hash16(cell_id + state_json
    + answers_json + prev_hash). Callers must pass the ALREADY-SERIALIZED
    JSON strings verbatim as they appeared on the wire — replaying
    serialization is where interop goes to die."""
    return hash16(cell_id + state_json + answers_json + prev_hash)
