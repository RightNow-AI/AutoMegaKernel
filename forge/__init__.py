"""AMK Forge -- empirical GPU reverse-engineering probes.

Probes measure a GPU's memory system (latency, bandwidth, memory-level parallelism) so a
megakernel generator can pick the occupancy / ILP each GPU needs to saturate HBM.

The finding this package addresses: AMK's persistent cooperative megakernel launches a FIXED
1 block/SM x 256 threads = 8 warps/SM on every GPU. Forge measures the bandwidth-vs-occupancy
saturation curve, its knee, the HBM latency and the Little's-law bytes-in-flight so a generator
KNOWS the occupancy each GPU needs to saturate HBM.

MEASURED SCOPE + HONEST CAVEATS (read before citing any number):
  * WHAT IS MEASURED (RTX 5090 Laptop, BOOST clocks, a single float4 streaming-READ pattern at
    ILP=1): the saturation curve, the empirical 95%-of-achieved-peak knee ~= 11 warps/SM, HBM
    dependent-load latency ~= 330 ns, achieved read-peak ~800-827 GB/s. REPRODUCED across the main
    run + two independent adversarial passes (own kernel/timer; knee 10.75/10.81/11.2, 8-warp
    86.8/87.0/84.5% of peak, latency 329/347/327 ns); HBM-bound confirmed by a halve-buffer guard.
  * CLOCK-DEPENDENT: under the laptop's SW power cap the same probe saturates by ~4 warps/SM (less
    bandwidth to feed); the ~11-warp knee is a BOOST-clock statistic, not a clock-invariant constant.
  * 8 warps/SM reaches ~85% of the read-only peak (~93-100% of a 2-way copy/triad peak): on the 5090
    AMK is only MILDLY under-saturated (consistent with the prior ~80% real-decode roofline). The
    raise-to-2-blocks/SM lever (-> 16 warps) reaches the plateau on streaming microbenchmarks.
  * NOT YET VALIDATED (honest backlog): the advisor's predict-vs-measure is a self-consistency check
    of the streaming probe, NOT yet a prediction of the real megakernel; the H100 warp-starvation is
    a MODEL PREDICTION (knee ~28-52 warps/SM, 8-warp ~4-8% of peak) that is consistent with the prior
    measured H100 ~8% but uses an assumed H100 latency + a register-frame haircut, not re-measured;
    Little's-law bytes/warp is back-solved from the knee (a re-parameterization, not an independent
    confirmation). Only a single float4 READ pattern is characterized (no write/mixed/tensor-core).

Import-clean and torch-free at module scope (mirrors ``schedule/ir.py``): ``import forge`` pulls
no CUDA, so a measured profile JSON can be inspected/diffed on any laptop. The measurement engine
(``forge.gpu_profiler``, which imports torch) is reached lazily via :func:`profile_gpu`.

New code only; the verified core (``vm/ops.cuh``, ``vm/loader.py``) is never edited.
"""
__version__ = "0.1.0"

from .gpu_model import (
    GpuModel,
    KernelFootprint,
    LatencyModel,
    OccupancyLimits,
    SaturationCurve,
    fit_hyperbolic,
)
from .kernel_advisor import Action, Advice, KernelAdvisor


def profile_gpu(*args, **kwargs):
    """Lazily import the torch-backed measurement engine and run a full GPU profile.

    Kept as a thin lazy delegate so ``import forge`` stays GPU-free; the heavy
    ``forge.gpu_profiler`` (and torch) is only imported when you actually measure.
    Returns a :class:`GpuModel`. See ``forge.gpu_profiler.profile_gpu`` for kwargs.
    """
    from . import gpu_profiler
    return gpu_profiler.profile_gpu(*args, **kwargs)


__all__ = [
    "__version__",
    "GpuModel",
    "SaturationCurve",
    "LatencyModel",
    "OccupancyLimits",
    "KernelFootprint",
    "fit_hyperbolic",
    "KernelAdvisor",
    "Advice",
    "Action",
    "profile_gpu",
]
