"""
Exact scalar kernel trace formula from Proposition 3.2 of the manuscript.

Implements Tr(K_{eps,a} * K_{eps,b}) as a closed-form scalar function,
eliminating the need for N x N operator matrices during scanning.

Numerical stability: uses np.logaddexp for the two exponential terms
to prevent overflow/underflow. The coth(2*eps) and tanh(2*eps)
coefficients are computed exactly via numpy — no approximations.
"""

import numpy as np


VACUUM_GAP = np.log(2)


def _log_kernel_trace(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """
    Log-space kernel trace: returns log(Tr(K_{eps,a} * K_{eps,b})).

    Proposition 3.2:
        Tr(K_{eps,a} K_{eps,b}) = [1 / (8 sinh(4eps))] * (
            exp(-tanh(2eps)/2 * r^2 - coth(2eps)/8 * d^2)
          + exp(-coth(2eps)/2 * r^2 - tanh(2eps)/8 * d^2)
        )
    where r = (a+b)/2, d = a-b.

    All terms are strictly positive for eps > 0 (Proposition 3.2 proof).
    """
    r = (a + b) / 2.0
    d = a - b

    th = np.tanh(2.0 * eps)
    cth = 1.0 / th  # coth(2*eps) = 1/tanh(2*eps)

    log_prefactor = -np.log(8.0 * np.sinh(4.0 * eps))

    # Exponent of term 1: -tanh(2eps)/2 * r^2 - coth(2eps)/8 * d^2
    log_term1 = -0.5 * th * r**2 - 0.125 * cth * d**2

    # Exponent of term 2: -coth(2eps)/2 * r^2 - tanh(2eps)/8 * d^2
    log_term2 = -0.5 * cth * r**2 - 0.125 * th * d**2

    # log(exp(log_term1) + exp(log_term2)) via logaddexp for stability
    log_sum = np.logaddexp(log_term1, log_term2)

    return log_prefactor + log_sum


def kernel_trace(a, b, eps: float):
    """
    Tr(K_{eps,a} * K_{eps,b}) — the exact second-moment kernel.

    Accepts scalar or array inputs via numpy broadcasting.
    Returns the kernel trace value(s) as float(s).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.exp(_log_kernel_trace(a, b, eps))


def log_kernel_trace(a, b, eps: float):
    """
    log(Tr(K_{eps,a} * K_{eps,b})) — log-space version for numerical stability.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return _log_kernel_trace(a, b, eps)


def hs_norm_squared(u, eps: float):
    """
    ||K_{eps,u}||_HS^2 = Tr(K_{eps,u}^2).
    Specialisation of Proposition 3.2 with a = b = u (so r = u, d = 0).
    """
    return kernel_trace(u, u, eps)


class KernelTraceMatrices:
    """
    Pre-computes the scalar kernel trace matrices K_pp, K_pb, K_bb
    for the quadratic form decomposition of Tr(delta_S^2).

    The energy Tr(delta_S(s)^2) = w_arith^T @ K_pp @ w_arith
                                  - 2 * w_arith^T @ K_pb @ w_bg
                                  + w_bg^T @ K_bb @ w_bg

    where w_arith and w_bg are s-dependent weight vectors, but the
    K matrices depend only on the prime power positions and quadrature grid.
    """

    def __init__(self, pp_log_n: np.ndarray, pp_lambda: np.ndarray,
                 u_bg: np.ndarray, bg_weights_simpson: np.ndarray,
                 eps: float):
        self.pp_log_n = pp_log_n
        self.pp_lambda = pp_lambda
        self.u_bg = u_bg
        self.bg_weights_simpson = bg_weights_simpson
        self.eps = eps

        P = len(pp_log_n)
        U = len(u_bg)
        self._P = P
        self._U = U

        # Build COMBINED kernel matrix K_full of size (P+U) x (P+U).
        # The anomaly weight vector is w = [w_arith; -w_bg], so
        # Tr(A^2) = w^T K_full w in a single bilinear form.
        all_positions = np.concatenate([pp_log_n, u_bg])
        self.K_full = kernel_trace(
            all_positions[:, np.newaxis], all_positions[np.newaxis, :], eps
        )

    def memory_usage_mb(self) -> float:
        """Total memory used by the kernel matrix in MB."""
        return self.K_full.nbytes / (1024 * 1024)

    def compute_trace_delta_sq(self, w_arith: np.ndarray,
                                w_bg: np.ndarray) -> complex:
        """
        Computes Tr(A^2) via the ANALYTIC square (bilinear, not sesquilinear).

        Uses combined weight vector w = [w_arith; -w_bg] and single
        bilinear form w^T K_full w. The anomaly (S - S_PNT) is computed
        in one pass.

        w_arith: (P,) complex — arithmetic weights Lambda(n) * n^{-s/2}
        w_bg: (U,) complex — background weights (density * Simpson weight * du)
        """
        w = np.concatenate([w_arith, -w_bg])
        return w @ self.K_full @ w
