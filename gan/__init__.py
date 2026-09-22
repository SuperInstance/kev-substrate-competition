"""gan/ — the generator side of kev-GAN: a population of kernel candidates
that mutate, predict, and are measured, every act receipted into the witness
chain. Category D of PLAYTEST-GAN.md, CPU-realizable today.

MOCK BENCH DOCTRINE: the benchmark here is an analytical cost model, clearly
labeled, deterministic per seed. It exists so the FULL loop (mutate ->
predict -> measure -> replay -> score) runs without a GPU. Swapping in the
real transfer-v4 eval changes one function; the witness format, the gates,
and the composite do not move. A number without its chain is a withdrawal
even when the number is a mock — the mock is in the payload, on the record.
"""

from .population import Candidate, Population
from .runner import run_generations, RunReport

__all__ = ["Candidate", "Population", "run_generations", "RunReport"]
