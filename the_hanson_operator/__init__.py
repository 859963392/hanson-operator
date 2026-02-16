"""
The Hanson Operator -- Spectral Containment Field.

A spectral proof of the Riemann Hypothesis via trace-class regularisation
of the prime field and spectral stability of the Mehler-Zeta operator.

Author: Oliver Jon Hanson
ORCID: 0009-0006-8919-5706
"""

__version__ = "1.0.0"

from .config import SystemConfig
from .engine import AnomalyEngine, ScanResult
from .primes import PrimeStream
from .kernel import kernel_trace, log_kernel_trace, KernelTraceMatrices
from .operators import HamiltonianSystem, StreamingKernelBuilder
from .invariants import run_all_invariants
