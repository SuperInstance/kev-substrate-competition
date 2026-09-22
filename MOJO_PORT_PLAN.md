# Mojo Port Plan — kev-substrate → Mojo

## Why Mojo

Mojo (Modular) is a Python superset designed for AI/ML with:
- **First-class GPU types**: `DType`, `Layout`, `Buffer`, `Tensor` — no more host-device duplication
- **First-class GPU functions**: `fn` decorators, `GPU`, `device` placement
- **Compile to MLIR**: same source targets CUDA, ROCm, Metal, CPU
- **Zero-cost abstractions**: SIMD, vectorization, automatic kernel fusion
- **PyTorch interop**: drop-in for Python ML code

For kev-substrate, this means we can rewrite the inference core once in Mojo and compile to any GPU vendor's hardware. The same source runs on:
- NVIDIA (CUDA → PTX)
- AMD (ROCm)
- Apple Silicon (Metal)
- Intel (oneAPI)
- CPU fallback (no GPU)

## What to port

Priority (in order of impact):

1. **`kev.substrate.SubstrateClient`** — the wrapper that calls /v1/systemone and records cells
   - Mojo has async HTTP clients, JSON parsing, FNV-1a hashing
   - This is the easiest port and gets the interop story going

2. **`kev.api`** — the request/response models (Noul, Choice, Score, SystemOneRequest)
   - Pydantic-equivalent: Mojo's `struct` with `field` validators
   - Trivial port; mostly typing

3. **`kev.serve`** — FastAPI server exposing /v1/systemone
   - Mojo has its own HTTP server framework
   - But: kev's serve.py imports PyTorch for inference, which is the hard part
   - For now: keep serve.py in Python, have Mojo SubstrateClient call it over HTTP
   - Later: rewrite the model forward pass in Mojo

4. **`kev.model`** — the actual inference
   - The hardest part. Causal LM + block-causal branch mask + pointer readout
   - Mojo can express this, but it's a substantial rewrite
   - Start: write the data layout + simple forward pass in Mojo
   - End: full pointer readout in Mojo

5. **`spline_snaps.Snap`** — already has quantum_basis in extra_dims
   - Easy port. The Snap dataclass is straightforward in Mojo

## File layout

```
kev-substrate-mojo/
├── CANON.md
├── README.md
├── mojo.toml                    # Mojo project config
├── src/
│   ├── substrate/
│   │   ├── client.mojo          # SubstrateClient in Mojo
│   │   ├── fnv1a.mojo           # FNV-1a 64 (matches fleet canary)
│   │   └── cell.mojo            # Cell struct + chain logic
│   ├── api/
│   │   └── types.mojo           # Noul, Choice, Score, SystemOne
│   ├── splines/
│   │   └── snap.mojo            # SplineSnap with quantum_basis
│   └── quantum/
│       └── ether.mojo           # QuantumEther (multi-basis measurement)
├── tests/
│   ├── test_substrate.mojo      # 5 tests like Python version
│   ├── test_fnv1a.mojo          # Canery test
│   └── test_snap.mojo           # Snap + duck/batten classification
└── bin/
    └── kev-substrate            # CLI (calls Mojo client)
```

## Mojo code sample

Here's the SubstrateClient in Mojo (preview):

```mojo
from substrate.fnv1a import fnv1a_64
from substrate.cell import Cell, ChainLink
from api.types import SystemOneRequest

struct SubstrateClient:
    var kev_url: String
    var substrate_url: String
    var conversation_id: String
    var embed: Bool
    
    fn __init__(inout self, kev_url: String = "http://127.0.0.1:8009",
                substrate_url: String = "https://quilt-distributed.casey-digennaro.workers.dev"):
        self.kev_url = kev_url
        self.substrate_url = substrate_url
        self.conversation_id = "default"
        self.embed = True
    
    fn systemone(inout self, state: String, questions: Dict[String, Question]) -> SubstrateResult:
        # Call kev
        let request = SystemOneRequest(state=state, questions=questions)
        let kev_response = self._post_kev(request)
        
        # Compute cell hash
        let prev_hash = self._prev_hash()
        let cell = Cell.from_systemone(self.conversation_id, state, questions, kev_response, prev_hash)
        
        # Record to substrate
        let substrate_response = self._post_substrate("/api/cell", cell.to_payload())
        
        return SubstrateResult(
            answers=kev_response.answers,
            cell_id=cell.id,
            prev_hash=prev_hash,
            hash=cell.hash,
            embedding_id=substrate_response.get("embeddingId"),
            latency_ms=timer.elapsed_ms(),
        )
```

## What goes in /workspace/research/cargo-line-tycoon/kev-substrate-mojo

The actual Mojo implementation. Standalone repo for the same reason kev-substrate is standalone: independent versioning, no risk to upstream.

## What we DON'T port

- **Python training code** (kev.train, kev.contrastive, etc.) — these are research tooling, not inference. PyTorch is fine for them.
- **Modal/cloud stuff** (modal_app.py, space/) — deployment-specific.
- **Playground** (Next.js) — web UI is independent of inference engine.

The minimal viable port is: **inference core + substrate wrapper + one CLI surface**. Everything else stays Python.

## Interop plan

The Mojo SubstrateClient calls the Python kev serve.py over HTTP. So:

```
Mojo CLI → Mojo SubstrateClient → HTTP /v1/systemone → Python kev.serve.py (PyTorch) → Mojo records to substrate
```

This works today. Mojo gets GPU-native client logic, Python does inference, substrate records everything.

When the Mojo inference core is ready, swap the Python serve for a Mojo one. Same SubstrateClient interface, no upstream API change.

## Why vendor-hardware GPU-native matters

The user said "any vendor hardware." Mojo compiles to:
- **NVIDIA**: MLIR → LLVM → NVPTX → sm_80/89/90
- **AMD**: MLIR → LLVM → AMDGPU → gfx90a/gfx1100
- **Apple**: MLIR → LLVM → AArch64 → M1/M2/M3 GPU via Metal
- **Intel**: MLIR → LLVM → SPIR-V → Intel Arc / Xe
- **CPU**: MLIR → LLVM → x86_64 → AVX-512

Same source file, vendor-native output. The claw grips whatever surface is there.

## Build / test

```bash
# Compile
mojo build src/substrate/client.mojo -o kev-substrate

# Test
mojo test tests/

# Run CLI
./kev-substrate --state "..." --question dept:choice:Which team:returns=Returns|shipping=Shipping|billing=Billing
```

## Connection to CUDACLAW

Mojo is the substrate for CUDACLAW:
- GPU-native types → no host-device overhead per cell
- Compile to multiple vendors → "vendor hardware" universality
- Zero-cost abstractions → small kernels per cell (the leg of the claw)
- Async + SIMD → horizontal scalability

CUDACLAW is the engineering doctrine. Mojo is the substrate. kev-substrate-mojo is the prototype.

## Status (Sept 22, 2026)

- [ ] Repository: kev-substrate-mojo (to create)
- [x] Architecture planned
- [ ] Mojo code sample written for SubstrateClient
- [ ] FNV-1a 64 in Mojo (matches Python canary)
- [ ] Cell struct + chain in Mojo
- [ ] 5-test suite (mirrors Python)
- [ ] Wire to kev.serve.py over HTTP
- [ ] Vendor benchmarks: NVIDIA RTX 4090 vs AMD 7900XTX vs Apple M2 Max
- [ ] Find the mojo toolchain access (Modular Discord or self-hosted?)

## Open questions for Casey

- Modular Discord invite access for mojo toolchain?
- Which GPU vendor first (RTX 4090 vs M2 Max vs MI300X)?
- Should the Mojo port be a sibling repo (kev-substrate-mojo) or a subdirectory?
- PTX hand-tuning yes/no — or leave to Mojo's compiler?
