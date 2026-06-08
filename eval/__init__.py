"""AMK evaluation, oracle (correctness), bench (latency), baselines, roofline.

The fixed eval of the autoresearch loop. Public surface:

  * :mod:`eval.oracle`   , ``logit_equivalence``, ``token_divergence``, ``Verdict``.
  * :mod:`eval.bench`    , ``bench`` (correctness-gated latency), ``BenchResult``.
  * :mod:`eval.roofline` , ``report`` (distance to the HBM-bandwidth bound), ``RooflineReport``.
  * :mod:`eval.baselines`, ``eager_baseline`` (runs), competitor stubs (honest ``not_run``).
  * :mod:`eval.perplexity`- ``amk_perplexity`` / ``hf_perplexity`` (teacher-forced fidelity vs HF).

The honesty invariant that ties them together: a latency only exists for a kernel an oracle
:class:`~eval.oracle.Verdict` certifies ``correct``, see :func:`eval.bench.bench`.
"""
from eval.bench import BenchResult, CorrectnessGateError, bench
from eval.oracle import Verdict, logit_equivalence, token_divergence, tolerances_for
from eval.peak_bandwidth import BandwidthResult, measure_peak_bandwidth
from eval.perplexity import PerplexityResult, amk_perplexity, hf_perplexity, toy_perplexity
from eval.roofline import RooflineReport
from eval.roofline import report as roofline_report

__all__ = [
    "Verdict", "logit_equivalence", "token_divergence", "tolerances_for",
    "bench", "BenchResult", "CorrectnessGateError",
    "roofline_report", "RooflineReport",
    "measure_peak_bandwidth", "BandwidthResult",
    "PerplexityResult", "amk_perplexity", "hf_perplexity", "toy_perplexity",
]
