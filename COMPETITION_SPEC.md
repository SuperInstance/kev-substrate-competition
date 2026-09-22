# KEV-SUBSTRATE COMPETITION — Cross-API CUDA/PTX Native

> Casey, 2026-09-22: "challenge your apis of different models to compete on innovative methods for a cuda/ptx native version that's more like our cudaclaw in engineering it's horizontal abilities"

## Premise

Different model APIs (OpenAI, Anthropic, DeepInfra, Cohere, Modular, etc.) each have their own decision-model architectures and inference stacks. Some are CUDA-native, some are ROCm, some are Metal. This competition asks them to **port kev-substrate to their stack** and compete on the horizontal-abilities benchmark.

## The benchmark

For each implementation, we measure:

### 1. Accuracy (calibration quality)
- Run kev's frozen eval suite (transfer-v4) through the implementation
- Score: **Brier score** (lower is better; 0 is perfect calibration)
- We also report coverage-at-≤5%-error (higher is better)

### 2. Speed
- Wall-clock latency per cell (P50, P95, P99)
- We report **ms per cell** averaged over 1000 cells
- All measurements on identical hardware (NVIDIA RTX 4090 unless vendor-specific)

### 3. Horizontal scale
- **cells_per_second** — how many substrate cells the system processes per second across all "legs" (devices/threads/SMs) simultaneously
- For single-GPU: max cells in flight at once
- For multi-GPU: aggregate across all GPUs
- For Mojo/PTX native: should be ~10x higher than CPU-only

### 4. Cost
- **dollars per million cells processed** at scale (extrapolated from observed throughput)
- Includes API costs (if cloud) or amortized hardware costs (if local)

### 5. Power draw
- Watts under benchmark load (if measurable)
- For cloud APIs: estimated via provider's published PUE

### Composite score

```
horizontal_score = (cells_per_second × calibration_quality) / (cost_per_M + power_draw)

where calibration_quality = 1 - brier_score
```

The winner is the implementation that maximizes this.

## What we're NOT measuring

- "Fastest single inference" — single-call latency doesn't capture horizontal abilities
- "Biggest model" — bigger isn't better; calibration is
- "Most parameters" — irrelevant; kev-0.8B is the baseline, not the upper bound

## The substrate requirement

All implementations must:

1. Use the same `kev.substrate.SubstrateClient` interface (HTTP API to substrate worker)
2. Record every inference call as a substrate cell with `prev_hash` chain
3. Support conversation chaining (cells in same conv chain via prev_hash)
4. Submit their results to the substrate (cells visible in canon)
5. Pass the FNV-1a canary test (canonical hash for "café Δ 日本語" = `0x024a555471370b18d`)

## Categories

We have three categories to encourage diverse approaches:

### A. Vendor-native CUDA/PTX (NVIDIA-optimized)
- Hand-tuned PTX for specific SM architectures
- Tensor core utilization
- CUDA Graphs for kernel sequencing
- Multi-stream pipelining

### B. Mojo / cross-vendor MLIR (universal GPU)
- Same source compiles to CUDA, ROCm, Metal
- Vendor-neutral optimization at the MLIR level
- Benchmarked on multiple hardware targets

### C. Novel architectures (radical ideas welcome)
- New sampling methods (e.g., continuous-time decision sampling)
- Novel state representations (e.g., spline-snaps as substrate cells)
- Quantum-inspired approaches (the shipwright doctrine)
- Anything that increases horizontal abilities

## Scoring matrix

| Submission | Category | Brier ↓ | ms/cell ↓ | cells/sec ↑ | $/M ↓ | Watts ↓ | horizontal_score |
|---|---|---|---|---|---|---|---|
| (entry) | A | 0.21 | 50 | 100 | $5 | 350 | TBD |
| ... | ... | ... | ... | ... | ... | ... | ... |

Leaderboard maintained at https://kev-substrate-competition.superinstance.dev (placeholder URL)

## Timeline

- **Week 0** (Sept 22, 2026): spec published, this doc
- **Week 1-2**: implementation submissions open
- **Week 3**: benchmark runs on standardized hardware
- **Week 4**: results announced, winners published as canon

## Outreach

Who we want to compete:
- **Modular** (Mojo) — vendor-neutral GPU
- **NVIDIA** (TensorRT, TensorRT-LLM) — PTX-native, the obvious leader
- **AMD** (ROCm, Composable Kernel) — competitive on price/perf
- **Apple** (Core ML, Metal) — different architecture entirely
- **OpenAI** (their custom kernels for inference)
- **Anthropic** (Claude's inference stack)
- **DeepInfra** (serverless GPU)
- **Together.ai** (open inference)
- **Groq** (LPU — different hardware entirely!)
- **Cerebras** (wafer-scale)
- **Sambanova** (RDU)
- **Academic labs** doing interesting inference research

## Submission format

Each team submits:
1. **Code repo** with their implementation (must build + run)
2. **Benchmark output** from their own hardware
3. **Architecture write-up** (1-2 pages) explaining the horizontal abilities approach
4. **Self-reported metrics** + reproducibility instructions

The host runs the benchmark independently on standardized hardware and publishes both sets of numbers.

## Prizes

- **🏆 Best Horizontal Abilities**: highest horizontal_score
- **🦀 Best Crab (most innovative)**: peer-nominated for novel approach
- **🦆 Best Duck (precision)**: lowest Brier at <100ms latency
- **▫️ Best Batten (organizing)**: best witness-log discipline

## Why this is more than a benchmark

The competition IS the substrate:
- Every submission becomes a canon cell
- The leaderboard IS the substrate witness log
- The architecture write-ups become substrate observations
- The horizontal_score becomes substrate metric

When the competition ends, the substrate has the entire competitive landscape embedded — queryable, traceable, animatable.

## Connection to persona-iteration doctrine

The competition results, when embedded in substrate, become part of the **animation of how decision-model architectures evolved in 2026**. Future Mavis reading the canon in 2028 can ask: "what did the substrate know about CUDA-native inference in 2026?" — and the substrate answers with the competitive record.

## Status

- [x] Spec published (this doc)
- [ ] Competition repo: kev-substrate-competition (to create)
- [ ] Leaderboard hosted on CF Pages
- [ ] Outreach to vendors (DM via Modular Discord, NVIDIA dev relations, etc.)
- [ ] Hardware standardization (RTX 4090 loaners? cloud credits?)
- [ ] Benchmark runner (Python orchestrator over all submissions)
- [ ] Canon integration (every result → substrate cell)

## Open questions

- Should we host on HuggingFace Spaces or CF Pages?
- How do we handle vendor-specific hardware (Apple Silicon can't run CUDA PTX)?
- Multi-track vs single-track? (one grand leaderboard vs separate GPU-vendor tracks)
- Should there be a "minimum viable submission" rule (e.g., must pass kev.eval ≥ 0.7)?
- Cash prize vs bragging rights vs canon-only?
- Who judges the "Best Crab" innovation prize?

— Filed by Mavis, 2026-09-22
