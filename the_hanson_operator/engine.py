"""
Unified Anomaly Engine — the core computation pipeline.

Constructs the anomaly operator A_eps(s) and computes its energy
E_eps(s) = Tr(A_eps(s)^2) using the scalar kernel trace (Tier 1).

The renormalisation prefactor sqrt(eps) * exp((1-s)^2 / (8*eps)) is
derived from first principles and is never modified. If peaks drift,
it is a resolution issue — adjust grid_points or epsilon.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from .config import SystemConfig
from .primes import PrimeStream
from .kernel import KernelTraceMatrices, VACUUM_GAP


# First 10 known Riemann zeros (imaginary parts) for reference
KNOWN_ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
])


@dataclass
class ScanResult:
    t_values: np.ndarray
    raw_traces: np.ndarray       # Re(Tr(A^2)) — the analytic square
    trace_moduli: np.ndarray     # |Tr(A^2)|
    log_energies: np.ndarray
    epsilon: float
    max_n: int

    def locally_normalized(self, window: float = 1.0) -> np.ndarray:
        """
        Divide |Tr(A^2)| by its moving average to remove the instrument
        response (Fourier weight decay) and prime variance drift.
        Leaves only sharp spectral features — the zero resonances.

        window: width of moving average in t-units (default 1.0).
        """
        dt = self.t_values[1] - self.t_values[0] if len(self.t_values) > 1 else 1.0
        hw = max(1, int(window / (2 * dt)))
        n = len(self.trace_moduli)
        smoothed = np.empty(n)
        for i in range(n):
            lo = max(0, i - hw)
            hi = min(n, i + hw + 1)
            smoothed[i] = np.mean(self.trace_moduli[lo:hi])
        smoothed[smoothed == 0] = 1.0  # avoid division by zero
        return self.trace_moduli / smoothed

    def find_resonances(self, threshold: float = 1.01,
                        window: float = 1.0) -> np.ndarray:
        """
        Find resonance peaks in the locally normalised signal.
        Returns t-values of local maxima above threshold.
        """
        norm = self.locally_normalized(window)
        peaks = []
        for i in range(2, len(norm) - 2):
            if (norm[i] > norm[i - 1] and norm[i] > norm[i + 1]
                    and norm[i] > threshold):
                peaks.append(i)
        return self.t_values[np.array(peaks)] if peaks else np.array([])

    def find_sign_changes(self) -> np.ndarray:
        """Find t-values where Re(Tr(A^2)) changes sign."""
        signs = np.sign(self.raw_traces)
        changes = np.where(np.diff(signs) != 0)[0]
        if len(changes) == 0:
            return np.array([])
        crossings = []
        for i in changes:
            t0, t1 = self.t_values[i], self.t_values[i + 1]
            v0, v1 = self.raw_traces[i], self.raw_traces[i + 1]
            if v1 != v0:
                t_cross = t0 - v0 * (t1 - t0) / (v1 - v0)
                crossings.append(t_cross)
        return np.array(crossings)


class AnomalyEngine:
    """
    The unified anomaly engine. One-stop setup: give it a config,
    it builds the prime stream, kernel matrices, and background integrator.

    Usage:
        engine = AnomalyEngine(SystemConfig.default())
        result = engine.scan_critical_line(12.0, 16.0, 200)
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        if config is None:
            config = SystemConfig.default()
        self.config = config
        self.eps = config.epsilon

        # Build prime stream
        self._primes = PrimeStream(config.max_n)
        self._pp_log_n, self._pp_lambda = self._primes.get_arrays(config.max_n)

        # Build background quadrature grid
        # The background integral must stop where the prime sum stops.
        # Integrating beyond log(max_n) adds pure PNT self-energy
        # with no primes to cancel it — a support mismatch that
        # drowns the resonance signal.
        bg_cutoff = np.log(float(config.max_n))

        # Quadrature density: need to resolve features of width ~sqrt(eps)
        span = bg_cutoff - VACUUM_GAP
        n_bg = max(config.bg_quadrature_points,
                   int(10.0 * span / np.sqrt(config.epsilon)))
        # Simpson's rule requires odd number of points
        if n_bg % 2 == 0:
            n_bg += 1
        self._u_bg = np.linspace(VACUUM_GAP, bg_cutoff, n_bg)
        h = (bg_cutoff - VACUUM_GAP) / (n_bg - 1)
        self._h_bg = h

        # Simpson's weights: 1, 4, 2, 4, 2, ..., 4, 1
        sw = np.ones(n_bg)
        sw[1:-1:2] = 4.0
        sw[2:-2:2] = 2.0
        self._simpson_weights = sw * (h / 3.0)

        # Pre-compute kernel trace matrices (the core Tier 1 structure)
        self._ktm = KernelTraceMatrices(
            self._pp_log_n, self._pp_lambda,
            self._u_bg, self._simpson_weights,
            self.eps,
        )

    def _arithmetic_weights(self, s: complex) -> np.ndarray:
        """
        w_arith[i] = Lambda(n_i) * n_i^{-s/2}
                    = Lambda(n_i) * exp(-s/2 * log(n_i))
        """
        return self._pp_lambda * np.exp(-0.5 * s * self._pp_log_n)

    def _background_weights(self, s: complex) -> np.ndarray:
        """
        w_bg[j] = exp((1 - s/2) * u_j) * simpson_weight_j

        The background integral:
            S_PNT(s) = Int_{log2}^inf exp(-su/2) * K_{eps,u} * exp(u) du
            = Int_{log2}^inf exp((1 - s/2) u) * K_{eps,u} du
        """
        return np.exp((1.0 - 0.5 * s) * self._u_bg) * self._simpson_weights

    def compute_raw_trace(self, s: complex) -> complex:
        """
        Tr(A^2) via the ANALYTIC square (bilinear, not sesquilinear).

        Returns a complex number. On the critical line, the resonance
        peaks are real and positive (dominated by the Gaussian term).
        The real part is the physical observable.
        """
        w_a = self._arithmetic_weights(s)
        w_b = self._background_weights(s)
        return self._ktm.compute_trace_delta_sq(w_a, w_b)

    def log_renormalization_prefactor(self, s: complex) -> float:
        """
        log|G_zero(s)| where G_zero(s) = sqrt(eps) * exp((1-s)^2 / (8*eps)).

        Computed in log space to avoid float64 overflow/underflow.
        The prefactor is EXACT — derived from first principles.
        """
        # log(sqrt(eps)) = 0.5 * log(eps)
        log_sqrt_eps = 0.5 * np.log(self.eps)

        # (1-s)^2 / (8*eps)
        # s = sigma + it on critical line: sigma = 0.5
        # (1 - s)^2 = (0.5 - it)^2 = 0.25 - t^2 - it
        # Re((1-s)^2 / (8*eps)) = (0.25 - t^2) / (8*eps)
        exponent = ((1.0 - s) ** 2) / (8.0 * self.eps)
        log_prefactor = log_sqrt_eps + exponent.real

        return log_prefactor

    def compute_log_energy(self, s: complex) -> float:
        """
        log|E_eps(s)| = log|G_zero(s)^2| + log|Tr(A^2)|

        The full renormalised energy in log space.
        E_eps(s) = G_zero(s)^2 * Tr(A(s)^2)
        where G_zero(s)^2 = eps * exp((1-s)^2 / (4*eps))
        """
        raw = self.compute_raw_trace(s)
        abs_raw = np.abs(raw)
        if abs_raw == 0.0:
            return -np.inf
        log_raw = np.log(abs_raw)
        # G_zero^2 prefactor in log space: log(eps) + Re((1-s)^2) / (4*eps)
        log_prefactor_sq = np.log(self.eps) + ((1.0 - s) ** 2).real / (4.0 * self.eps)
        return log_prefactor_sq + log_raw

    def scan_critical_line(self, t_min: float, t_max: float,
                           points: int = 500) -> ScanResult:
        """
        Sweep s = 0.5 + it over [t_min, t_max] and record anomaly energy.

        Stores Re(Tr(A^2)) as the primary observable — on the critical line,
        resonance peaks at the zeros are real and positive.
        """
        t_values = np.linspace(t_min, t_max, points)
        raw_traces = np.zeros(points)
        trace_moduli = np.zeros(points)
        log_energies = np.zeros(points)

        for i, t in enumerate(t_values):
            s = 0.5 + 1j * t
            trace = self.compute_raw_trace(s)
            raw_traces[i] = trace.real
            trace_moduli[i] = abs(trace)
            log_energies[i] = self.compute_log_energy(s)

        return ScanResult(
            t_values=t_values,
            raw_traces=raw_traces,
            trace_moduli=trace_moduli,
            log_energies=log_energies,
            epsilon=self.eps,
            max_n=self.config.max_n,
        )

    def multi_epsilon_scan(self, t_min: float, t_max: float,
                           points: int = 500,
                           epsilon_values: Optional[List[float]] = None
                           ) -> List[ScanResult]:
        """
        Run scans at multiple epsilon values to demonstrate convergence.
        As eps -> 0, features sharpen and zero crossings converge to true zeros.
        """
        if epsilon_values is None:
            epsilon_values = [0.5, 0.2, 0.1, 0.05]

        results = []
        for eps in epsilon_values:
            cfg = SystemConfig(
                grid_points=self.config.grid_points,
                grid_extent=self.config.grid_extent,
                epsilon=eps,
                max_n=self.config.max_n,
                bg_quadrature_points=self.config.bg_quadrature_points,
                bg_cutoff=self.config.bg_cutoff,
            )
            eng = AnomalyEngine(cfg)
            results.append(eng.scan_critical_line(t_min, t_max, points))

        return results
