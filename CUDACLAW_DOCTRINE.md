# CUDACLAW — Horizontal-Abilities GPU Engineering

> Casey, 2026-09-22: "challenge your apis of different models to compete on innovative methods for a cuda/ptx native version that's more like our cudaclaw in engineering it's horizontal abilities"

## What is cudaclaw?

A pod, not a pack. A claw, not a fist. **Horizontal abilities** — every leg doing its own work, peer-to-peer, no central coordination.

The name combines:
- **CUDA** — NVIDIA's GPU compute framework, the de-facto standard
- **PTX** — Parallel Thread eXecution, the IR that lets you target specific SM architectures
- **CLAW** — like a crab: many legs, no hierarchy, all working at once

## The doctrine

**Conventional GPU engineering** (the "pack"):
- One kernel launches, coordinates everything
- One big optimization, central scheduling
- Hierarchical: SM → warp → thread, top-down

**CUDACLAW** (the horizontal claw):
- Many small kernels, peer-to-peer
- Each "leg" (substrate cell) does its own work
- No central coordinator; the substrate itself is the coordination
- Bottom-up emergence

| Conventional | CUDACLAW |
|---|---|
| One big kernel | Many small substrate-cell kernels |
| Synchronous barriers | Async, fire-and-forget with witness-log |
| ThreadIdx = position | CellId = position (substrate carries semantics) |
| Shared memory = coordination | Witness-log = coordination |
| Block size = optimization | Cell graph = optimization |
| One SM, one purpose | Many SMs, one substrate |

## Why this fits kev-substrate

kev's inference is fundamentally:
- Pack state tokens into context
- Run model forward
- Read out probabilities at decision positions

With CUDACLAW:
- Each **decision** is its own cell (its own small kernel)
- State tokens are loaded once into a **substrate shared memory** (the witness-log)
- Each cell reads from the substrate, computes its decision, writes its witness
- No global sync needed — the substrate IS the shared state
- Horizontal: many cells run in parallel, each one independent

This matches kev's `option_isolation=True` mode (per-option sub-branches) — but extended to per-cell.

## Implementation layers

1. **Mojo layer** (`kev-substrate-mojo/`) — Python-superset with GPU-native features
   - First-class GPU types: `DType.float32`, `Layout.row_major`, etc.
   - Compiles to MLIR, runs on CUDA / ROCm / Metal / CPU
   - "Vendor-hardware GPU native" — same source, different backends

2. **CUDA/PTX native layer** (`kev-substrate-ptx/`) — lowest level
   - Hand-tuned PTX for target SM architectures (sm_80, sm_89, sm_90)
   - Warp-level primitives for cell-level parallelism
   - Async memcpy + TMA for witness-log streaming

3. **API competition layer** (`kev-substrate-competition/`) — cross-API benchmark
   - Each vendor/team submits an implementation
   - Runs on identical test suites
   - Scored on: accuracy (calibration), speed (ms), horizontal-scale (cells per ms)
   - The winner is the one with the best **horizontal ability** — not the fastest single call

## The horizontal abilities benchmark

For kev-substrate-competition, we don't measure "fastest single inference." We measure:

```
horizontal_score = (cells_per_second × calibration_quality) / (cost + power_draw)
```

Where:
- **cells_per_second**: how many substrate cells can the system process per second across all "legs" simultaneously?
- **calibration_quality**: Brier score on the held-out suite (lower is better)
- **cost**: dollars per million cells processed
- **power_draw**: watts under benchmark load

The winner is the implementation that processes the most substrate cells per joule per dollar, while keeping the substrate coherent.

## Connection to substrate doctrine

This IS the persona-iteration doctrine applied to compute:
- Every cell is observable (its witness is on the log)
- Every cell is queryable (the substrate holds its state)
- Every cell is traceable (prev_hash chain)
- The substrate IS the coordination layer — no top-down scheduler needed

A CUDACLAW system doesn't have a "main thread." It has many legs, each one a cell. The substrate is the body they share.

## Why "claw" not "fist"

- **Fist**: one big thing, all fingers working together, hierarchical
- **Claw**: many small things, each independent, peer-to-peer, horizontal

Crabs don't punch. They grip with many legs at once. CUDA/PTX can do that — we just have to think of compute as grip, not punch.

## Files

- `CUDACLAW_DOCTRINE.md` (this file)
- `MOJO_PORT_PLAN.md` — how to port kev-substrate to Mojo
- `COMPETITION_SPEC.md` — the cross-API competition rules
- Reference implementations to come

— Filed by Mavis, 2026-09-22
