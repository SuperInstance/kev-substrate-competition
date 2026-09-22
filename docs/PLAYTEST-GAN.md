# Playtest findings + kev-GAN — adversarial evolution lane

2026-09-22, kimi1 (playtest pass 1 of N, per Casey's "extensively
iteratively playtest then build"). Spec read cold; findings are concrete
and checkable; each names its fix.

## Playtest findings (the spec, stress-tested)

**F1. The composite mixes units.** `cost_per_M + power_draw` adds dollars
to watts. Submissions can arbitrage the sum (cheap cloud + high PUE vs
local low-watt). Fix: normalize each term to the field minimum
(ratio-to-best), then sum — dimensionless, order-independent, honest.

**F2. No coherence term in the score.** The README says "while keeping
the substrate coherent"; the composite has no coherence factor, so the
literal optimum is maximum incoherent throughput. Fix: multiply by a
coherence coefficient c ∈ (0,1] from the witness log (fraction of cells
that verify against prev_hash on independent replay). Coherence is not a
vibe; it's a measured ratio, and it becomes the discriminator's blade.

**F3. The frozen eval can be memorized.** A submission can distill
transfer-v4 and farm Brier. Fix: split the suite — public half for
development, held-out half the host runs and never publishes.

**F4. Hash-chain cost is unpriced.** At target ~10⁶ cells/s, the
FNV-1a prev_hash chain is real work. The spec doesn't say if hashing
counts in ms/cell. Fix: count it (honest default); exclude only if
declared, and then the declared exclusion goes in the witness log.

**F5. Self-reported metrics need receipts.** Submission metrics are
self-reported + host-verified, but the format doesn't demand the
evidence chain. Fix: every reported number must cite its run's witness
log (hash-chained cells); a number without a chain is a withdrawal, not
a submission. (The fleet's candor WAL — receipts booked before storage,
replay-verify at boot — is the reference implementation; Best Batten
already points this way.)

## kev-GAN — the adversarial evolution lane (the build)

Casey's directive: build further **in a GAN directed towards the
killer-app**. The spec has competitors; it lacks the loop that makes
competition *adversarial in structure*. kev-GAN adds Category D —
open adversarial evolution — and makes it the main event:

```
GENERATOR (any model API — OpenAI, Anthropic, Kimi, z.ai, Mojo agent)
  mutates a submission: kernel code (PTX / Mojo / TRT / CK), batching
  policy, sampling scheme, memory layout. Mutation is seeded and
  receipted — every candidate is a substrate cell.

DISCRIMINATOR (the standardized harness, nothing else)
  = transfer-v4 Brier (held-out half)          — precision
  × coherence coefficient from witness replay   — F2
  × cells/s on the standard 4090                — horizontal
  ÷ normalized cost + power                     — F1, fixed units
  + hard gates: canary, interface, chain verify — F5

LOOP
  population → evaluate (receipted) → select top-k → mutate
  → re-submit as new cells. Generations are canon; the lineage
  graph IS the substrate animation Casey quoted in the spec.
```

**Why this is the killer-app shape:** the first competition where the
entrant is an agent and the judge is a receipts-verified benchmark —
vendor teams and model APIs evolve kernels against the same immutable
discriminator. Whoever's generator learns fastest wins, and the learning
curve is embedded in the canon as first-class cells. The killer app is
not a faster kernel; it is **the substrate that evolves its own
inference** — and sells/democratizes that loop.

**Generator discipline (anti-slop):** mutation proposals must carry a
one-line hypothesis ("larger L2 tiling reduces power at fixed brier");
the receipt records it; rejected hypotheses stay in the chain (no
erasure — authority-without-erasure doctrine). Best Crab judging gets
its evidence for free.

**Smallest first build (one lane-week):**
1. Fix F1–F5 in the spec (this doc's fixes; host decision on F3 split).
2. `harness/` — witness-verifying benchmark runner (canary, chain
   replay, coherence coefficient, normalized composite).
3. `gan/` — population runner: seeded mutations over ONE baseline
   submission, 8 candidates/generation, generation = 1 git commit +
   receipt batch.
4. Demonstration generation 0→3 on a CPU-fallback kernel (the 4090
   lanes follow).

## Category C note (shipwright tie-in)

The spec already cites the shipwright doctrine for spline-snap cells —
consistent with the loft design (ducks = snapped exact anchors,
battens = splines, negative space honest). kev-GAN gives that doctrine
its competitive habitat: spline-snap state representations evolve
against the same discriminator as PTX hand-tuning. Ducks and battens,
competing for joules.

## Honest gaps

- Held-out eval half (F3) requires host discipline; leak = competition
  integrity death. Acknowledged, not solved, by this doc.
- Generator APIs cost money per mutation; budget caps needed (per-team
  generation budget in the rules, or the loop becomes a wallet race).
- CPU-fallback demonstration ≠ 4090 truth; standardization still the
  host's burden (spec's open questions stand).
