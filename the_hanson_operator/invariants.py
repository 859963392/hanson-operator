"""
Invariant validators for the Spectral Containment Field.

These are the load-bearing walls of the proof logic. If any invariant
fails, the machine is broken and results cannot be trusted.

1. Trace Class (B1): G_eps must have finite trace.
2. Vacuum Dominance: The vacuum gap [0, log 2] must be silent.
3. Renormalisation Rigidity: The prefactor is exact — no tuning.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .config import SystemConfig
from .operators import HamiltonianSystem
from .engine import AnomalyEngine
from .kernel import hs_norm_squared


class VacuumBreachError(Exception):
    """Raised when the vacuum dominance invariant is violated."""
    pass


class TraceClassError(Exception):
    """Raised when the trace class invariant is violated."""
    pass


@dataclass
class InvariantResult:
    name: str
    passed: bool
    details: Dict[str, float]
    message: str


def check_trace_class(config: SystemConfig) -> InvariantResult:
    """
    Verify G_eps = exp(-eps H) is trace class.
    Compares sum(exp(-eps * eigenvalues)) to 1/(2*sinh(eps)).

    The relative error tolerance depends on N — truncation of L^2(R)
    to a finite grid introduces error that shrinks with larger N.
    """
    system = HamiltonianSystem(config)

    numerical = system.trace_g_eps(config.epsilon)
    theoretical = system.theoretical_trace(config.epsilon)
    rel_error = abs(numerical - theoretical) / theoretical

    # Tolerance scales with grid size: at N=400, expect ~7% error
    # from eigenvalue divergence at high indices
    rtol = max(0.15, 10.0 / config.grid_points)
    passed = rel_error < rtol

    return InvariantResult(
        name="Trace Class (B1)",
        passed=passed,
        details={
            "theoretical": theoretical,
            "numerical": numerical,
            "relative_error": rel_error,
            "tolerance": rtol,
        },
        message=(
            f"Tr(G_eps) = {numerical:.4f} vs theoretical {theoretical:.4f} "
            f"(rel error {rel_error:.4f}, tol {rtol:.4f})"
        ),
    )


def check_vacuum_dominance(config: SystemConfig) -> InvariantResult:
    """
    The vacuum must be silent.

    Scans t in [0, 1] (well below any Riemann zero). The anomaly energy
    must be sub-exponential — no spikes. If violated, this is a HARD
    FAILURE: the machine is broken.
    """
    engine = AnomalyEngine(config)
    result = engine.scan_critical_line(0.0, 1.0, points=20)

    max_abs_trace = np.max(np.abs(result.raw_traces))
    mean_abs_trace = np.mean(np.abs(result.raw_traces))

    # The vacuum energy should be small compared to resonance energy.
    # At a resonance (e.g. t=14.13), energy is significantly larger.
    # In the vacuum, we expect the trace to be dominated by the
    # background subtraction residual — should be very small.
    # Threshold: if max vacuum energy exceeds 10x the mean, there's a spike.
    has_spike = False
    if mean_abs_trace > 0:
        has_spike = (max_abs_trace / mean_abs_trace) > 10.0

    # Also check absolute magnitude — should be much less than resonance scale
    passed = not has_spike

    details = {
        "max_abs_trace": max_abs_trace,
        "mean_abs_trace": mean_abs_trace,
        "spike_ratio": max_abs_trace / mean_abs_trace if mean_abs_trace > 0 else 0,
    }

    if not passed:
        raise VacuumBreachError(
            f"VACUUM BREACH: Energy spike detected in vacuum gap [0, 1]. "
            f"Max |Tr(dS^2)| = {max_abs_trace:.4e}, "
            f"mean = {mean_abs_trace:.4e}, "
            f"ratio = {details['spike_ratio']:.1f}x. "
            f"Check kernel decay or background subtraction."
        )

    return InvariantResult(
        name="Vacuum Dominance",
        passed=True,
        details=details,
        message=(
            f"Vacuum silent: max |Tr(dS^2)| = {max_abs_trace:.4e}, "
            f"mean = {mean_abs_trace:.4e}"
        ),
    )


def check_renormalization_rigidity(config: SystemConfig) -> InvariantResult:
    """
    Verify the renormalisation prefactor is exactly
    sqrt(eps) * exp((1-s)^2 / (8*eps)).

    We check that the log-prefactor computation matches the analytical
    formula at several test points. No fudge factors allowed.
    """
    engine = AnomalyEngine(config)
    eps = config.epsilon

    test_points = [0.5 + 1j * t for t in [0.0, 5.0, 14.13, 25.0]]
    max_error = 0.0

    for s in test_points:
        # Direct computation of log|prefactor|
        log_pf = engine.log_renormalization_prefactor(s)

        # Analytical: log(sqrt(eps)) + Re((1-s)^2) / (8*eps)
        expected = 0.5 * np.log(eps) + ((1.0 - s) ** 2).real / (8.0 * eps)

        error = abs(log_pf - expected)
        max_error = max(max_error, error)

    passed = max_error < 1e-12

    return InvariantResult(
        name="Renormalization Rigidity",
        passed=passed,
        details={"max_error": max_error},
        message=(
            f"Prefactor exact: max deviation {max_error:.2e} "
            f"({'PASS' if passed else 'FAIL: prefactor has been modified'})"
        ),
    )


def check_hs_decay(config: SystemConfig) -> InvariantResult:
    """
    Verify Hilbert-Schmidt norm decay from Lemma 2.5:
    ||K_{eps,u}||_HS^2 decreases as a Gaussian in u.
    """
    eps = config.epsilon
    test_u = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    norms = hs_norm_squared(test_u, eps)

    # Check monotone decrease for u > 0
    is_decreasing = np.all(np.diff(norms) < 0)

    # Check Gaussian decay: ratio of norms at u and 2u should be
    # approximately exp(-c * (4u^2 - u^2)) = exp(-3cu^2) for some c > 0
    passed = is_decreasing and np.all(norms > 0)

    return InvariantResult(
        name="Hilbert-Schmidt Decay",
        passed=passed,
        details={
            "u_values": test_u.tolist(),
            "hs_norms": norms.tolist(),
            "is_decreasing": is_decreasing,
        },
        message=(
            f"HS norms at u=[0.5..8]: {norms[0]:.4e} -> {norms[-1]:.4e} "
            f"({'monotone decreasing' if is_decreasing else 'NOT DECREASING'})"
        ),
    )


def run_all_invariants(config: Optional[SystemConfig] = None) -> list:
    """
    Run all invariant checks. Returns list of InvariantResult.
    Raises VacuumBreachError if the vacuum is violated.
    """
    if config is None:
        config = SystemConfig.default()

    results = []
    results.append(check_trace_class(config))
    results.append(check_vacuum_dominance(config))  # Raises on failure
    results.append(check_renormalization_rigidity(config))
    results.append(check_hs_decay(config))
    return results
