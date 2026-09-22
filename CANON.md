---
canon: 1
name: kev-substrate-competition
mission: "Cross-API CUDA/PTX native competition — challenge model APIs to compete on innovative methods for substrate interweave. Horizontal-abilities engineering per CUDACLAW doctrine."
state: spec-published
family: applications
vessel: SuperInstance
born_from: [SuperInstance/kev-substrate, SuperInstance/cargo-line-tycoon]
feeds: [SuperInstance/kev-substrate]
canonical_docs: [README.md, COMPETITION_SPEC.md]
ledger: git-log
verified: 2026-09-22
---

# kev-substrate-competition

Cross-API competition: model APIs compete on **CUDA/PTX native** implementations of the kev-substrate interweave, judged on **horizontal abilities** (the CUDACLAW doctrine).

## The premise

Different model APIs each have their own inference stack. This competition asks them to port kev-substrate to their stack and compete on:

```
horizontal_score = (cells_per_second × calibration_quality) / (cost_per_M + power_draw)
```

## What we measure

1. **Accuracy** — Brier score on kev's frozen eval suite (lower = better)
2. **Speed** — ms per cell P50/P95/P99 over 1000 cells
3. **Horizontal scale** — cells per second across all "legs" simultaneously
4. **Cost** — dollars per million cells processed at scale
5. **Power draw** — watts under benchmark load (or estimated via PUE)

## Categories

A. **Vendor-native CUDA/PTX** — hand-tuned PTX, tensor cores, multi-stream
B. **Mojo / cross-vendor MLIR** — same source, vendor-universal
C. **Novel architectures** — spline-snaps, quantum-ether, shipwright doctrine, anything new

## Prizes

- 🏆 **Best Horizontal Abilities** — highest horizontal_score
- 🦀 **Best Crab** — most innovative approach (peer-nominated)
- 🦆 **Best Duck** — lowest Brier at <100ms latency
- ▫️ **Best Batten** — best witness-log discipline

## Why this is more than a benchmark

The competition IS the substrate:
- Every submission becomes a canon cell
- The leaderboard IS the substrate witness log
- Architecture write-ups become substrate observations
- The horizontal_score becomes substrate metric

When the competition ends, the substrate has the entire competitive landscape embedded — queryable, traceable, animatable.

## Status

- [x] Spec published
- [x] Repo created
- [ ] Outreach to vendors (Modular, NVIDIA, AMD, Apple, OpenAI, Anthropic, DeepInfra, Groq, Cerebras)
- [ ] Leaderboard hosted on CF Pages
- [ ] Hardware standardization (RTX 4090 loaners? cloud credits?)
- [ ] Benchmark runner
- [ ] First submission window opens (target: Q4 2026)

## See

- [COMPETITION_SPEC.md](COMPETITION_SPEC.md) — full rules
- [SuperInstance/kev-substrate](https://github.com/SuperInstance/kev-substrate) — the reference implementation
- [SuperInstance/kev-substrate-mojo](https://github.com/SuperInstance/kev-substrate-mojo) — the Mojo entry (Category B)
