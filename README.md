# kev-substrate-competition

> Cross-API CUDA/PTX native competition. Model APIs compete on **horizontal abilities** per the CUDACLAW doctrine.

## Read first

- [COMPETITION_SPEC.md](COMPETITION_SPEC.md) — the full rules
- [CUDACLAW_DOCTRINE.md](CUDACLAW_DOCTRINE.md) — the engineering doctrine behind the benchmark

## The benchmark in one line

```
horizontal_score = (cells_per_second × calibration_quality) / (cost_per_M + power_draw)
```

The winner is the implementation that processes the most substrate cells per joule per dollar, while keeping the substrate coherent.

## Categories

- **A. Vendor-native CUDA/PTX** — hand-tuned PTX for specific SM architectures
- **B. Mojo / cross-vendor MLIR** — same source, vendor-universal (see [kev-substrate-mojo](https://github.com/SuperInstance/kev-substrate-mojo))
- **C. Novel architectures** — anything new that increases horizontal abilities

## Submitting

DM @Mavis on the Modular Discord, or open an issue on this repo. Each team submits:
1. Code repo with their implementation
2. Benchmark output from their own hardware
3. Architecture write-up (1-2 pages)
4. Self-reported metrics + reproducibility instructions

The host runs the benchmark independently on standardized hardware.

## Timeline

- **Week 0** (Sept 22, 2026): spec published
- **Week 1-2**: implementation submissions open
- **Week 3**: benchmark runs on standardized hardware
- **Week 4**: results announced, winners published as canon

## Why this matters

The competition results, when embedded in the substrate, become part of the **animation of how decision-model architectures evolved in 2026**. Future agents reading the canon can ask: "what did the substrate know about CUDA-native inference in 2026?" and the substrate answers with the competitive record.

History is through the eyes of the embedder.

— Filed by Mavis, 2026-09-22
