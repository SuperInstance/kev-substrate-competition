# Round 2 — deep-research refinements (playtest pass 2)

2026-09-22, kimi1. Explorer loop pass 2: playtest → deep research →
innovate. Sources: GPUMODE + MLSys 2026 FlashInfer contest (OpenEvolve
agent baseline), K-Search (arXiv 2602.19128), speed-of-light guidance
(arXiv 2603.29010), KernelFalcon deep-agent architecture, Red Hat
cross-vendor Triton study, KernelBench retrospective. Claims tagged
VERIFIED (read) / INFERRED (bridge).

## The landscape fact that changes our lane

Agent-evolved kernels are no longer novel — **they are the baseline**.
MLSys 2026's FlashInfer contest ships an OpenEvolve-based agent track;
GPUMODE runs public leaderboard kernel competitions; KernelBench's own
retrospective says benchmarks saturate and the frontier is the *loop*.
So kev-GAN's moat cannot be "agents evolve kernels." It must be the
three things the incumbents don't measure:

1. **Economics in the score** (cells/s × calibration ÷ (cost+power)) —
   nobody's leaderboard prices joules and dollars against coherence.
2. **The witness chain** (F5 from pass 1) — nobody requires a receipt
   per reported number. A metric without a chain is a withdrawal.
3. **Calibration of the critic itself** (new, R2.2 below) — the search
   system's own beliefs get scored, recursively.

## R2.1 — Speed-of-light model in the discriminator (VERIFIED→INFERRED)

arXiv 2603.29010: DSL + speed-of-light guidance — giving the generator
a roofline model of what perfect would be — is the current efficiency
frontier. Map to the shipwright: SOL is the batten's fair curve; the
gap between the current kernel and SOL is the amplitude field. Add to
the composite a **SOL-efficiency term** `measured_throughput /
roofline_ceiling(kernel_shape)` — it is instrument-independent,
portable across vendors, and it answers "is there any duck left worth
placing here?" at a glance. This is information-per-duck, generalized.

## R2.2 — Co-evolve the critic; score its calibration (VERIFIED→INFERRED)

K-Search's critique of the FunSearch/AlphaEvolve lineage is precise:
those systems "treat the LLM merely as a stochastic code generator."
K-Search co-evolves an intrinsic world model and beats OpenEvolve /
ShinkaEvolve by 1.75–2.21×. kev-GAN's round-2 architecture: the
population contains **kernel candidates AND a critic** that predicts
each candidate's measured score. Every prediction is receipted;
prediction vs measurement is the critic's Brier. The discriminator
thus scores twice: once for the kernel, once for the belief about the
kernel. A search loop that can say "my critic is miscalibrated on
memory-bound shapes" is a search loop that knows where its negative
space is — the honesty doctrine, recursive.

## R2.3 — Deterministic control plane (VERIFIED)

KernelFalcon's deep-agent principle, adopted verbatim: orchestration,
timeouts, artifact paths, and admission decisions run in deterministic
code; model APIs only *propose*. The harness validates by execution,
never by model opinion. This is candor's gate-at-write wearing a
benchmark suit: proposals are receipts, admissions are predicates,
refusals are visible rows.

## R2.4 — Time-to-performance is the real killer-app metric (INFERRED)

The Red Hat cross-vendor study's sharpest finding: peak performance and
time-to-results trade hard (Helion wins, after hours of tuning;
Inductor is the strong default). Incumbent contests measure the peak.
The world buys the *time*. Add a **TT(1.2×)** term: wall-clock from
cold start to first kernel beating compiled-baseline by 1.2×, capped
and receipted. Composite candidate:

```
horizontal_score = (cells/s × calib × coherence × SOL_eff) /
                   (norm_cost + norm_power)
TT(1.2×) reported as a first-class column, tie-break before Brier
```

The killer app, restated honestly: not "beat FlashInfer" (VERIFIED:
generated kernels rarely beat expert baselines — set expectations in
the spec), but **acceptable performance at 1/10th the tuning latency,
witness-verified, on any vendor's silicon**.

## R2.5 — Anti-saturation clause (VERIFIED)

KernelBench retrospective: benchmarks saturate; design for movable
targets. The spec's composite parameters (thresholds, the 1.2× in
TT, the eval suite) must be canon-versioned cells, re-tunable by
competition governance, with every tuning receipted — or year two of
this competition is a solved game with a leaderboard fossil.

## Sequenced into the lane plan

Pass 1 gave the harness skeleton (canary, chain replay, coherence,
normalized composite). Pass 2 upgrades the discriminator: SOL model,
critic population with receipted predictions, TT(1.2×) column,
governed parameter cells. Build order unchanged, one addition after
step 2: `sol.mjs` — the roofline calculator per kernel shape, vendored
constants per target hardware, first target the RTX 4090 standard.

## Honest gaps, round 2

- Roofline models lie for irregular kernels; SOL_eff is a guide term,
  not truth — keep it out of hard gates.
- Critic co-evolution doubles the generation budget problem; caps from
  pass 1 now cover both populations.
- TT(1.2×) on cloud APIs includes queue latency outside a team's
  control; report provider-side and client-side separately.
