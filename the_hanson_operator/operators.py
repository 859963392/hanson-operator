"""
Tier 2: Eigendecomposition-based operator construction.

Builds the Hamiltonian H = -d^2/dx^2 + x^2 on a discrete grid,
computes its eigendecomposition once, and provides streaming
construction of Mehler kernel blocks K_{eps, u} on demand.

Used for invariant validation (trace class, operator diagnostics).
Not needed for scanning — that uses the scalar kernel trace (kernel.py).
"""

import numpy as np
from scipy.sparse import diags

from .config import SystemConfig


class HamiltonianSystem:
    """
    The physical substrate: L^2(R) truncated to a finite grid.
    Eigendecomposition computed once on construction.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self._H = self._build_hamiltonian()
        # One-time eigendecomposition — O(N^3) but done once
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(self._H)

    def _build_hamiltonian(self) -> np.ndarray:
        """
        Harmonic oscillator: H = -d^2/dx^2 + x^2.
        Central difference Laplacian on the grid.
        """
        N = self.config.grid_points
        dx = self.config.dx
        x = self.config.x_axis

        # Kinetic: -d^2/dx^2 via [1, -2, 1] / dx^2
        main_diag = np.full(N, -2.0)
        off_diag = np.ones(N - 1)
        laplacian = diags([off_diag, main_diag, off_diag], [-1, 0, 1],
                          shape=(N, N)).toarray() / (dx ** 2)
        K = -laplacian

        # Potential: x^2
        V = np.diag(x ** 2)

        return K + V

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self._eigenvectors

    def propagator_eigenvalues(self, eps: float) -> np.ndarray:
        """exp(-eps * eigenvalues) — the heat operator in eigenbasis."""
        return np.exp(-eps * self._eigenvalues)

    def trace_g_eps(self, eps: float) -> float:
        """Tr(G_eps) = sum(exp(-eps * eigenvalues))."""
        return np.sum(self.propagator_eigenvalues(eps))

    def theoretical_trace(self, eps: float) -> float:
        """Theoretical Tr(G_eps) = 1 / (2 * sinh(eps))."""
        return 1.0 / (2.0 * np.sinh(eps))


class StreamingKernelBuilder:
    """
    Constructs K_{eps, u} = G_eps cos(uD) G_eps blocks on demand
    using the eigendecomposition. Never stores more than 2 blocks.

    The shift operator cos(uD) is applied via FFT on the eigenvectors.
    """

    def __init__(self, system: HamiltonianSystem, eps: float):
        self.system = system
        self.eps = eps
        self.N = system.config.grid_points
        self.dx = system.config.dx

        # Propagator eigenvalues: g_n = exp(-eps * E_n)
        self._g = system.propagator_eigenvalues(eps)

        # Pre-compute FFT frequencies for the shift operator
        self._k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi

    def get_block(self, u: float) -> np.ndarray:
        """
        Compute K_{eps, u} = G_eps cos(uD) G_eps as an N x N matrix.

        Uses vectorised FFT: shifts all eigenvectors simultaneously,
        then applies the propagator weights.
        """
        V = self.system.eigenvectors  # (N, N)
        g = self._g  # (N,)

        # G_eps in eigenbasis: G_eps = V diag(g) V^T
        # cos(uD) applied to columns of V via FFT
        # cos(uD) f = Re(F^{-1}[exp(iku) F[f]]) averaged with exp(-iku)
        # Actually: cos(uD) = (T_u + T_{-u}) / 2

        phase = np.exp(1j * self._k * u)  # (N,)

        # Apply T_u to all columns of V simultaneously
        V_fft = np.fft.fft(V, axis=0)  # (N, N)
        V_shifted_pos = np.fft.ifft(V_fft * phase[:, np.newaxis], axis=0).real
        V_shifted_neg = np.fft.ifft(V_fft * phase[:, np.newaxis].conj(), axis=0).real

        # cos(uD) V = (T_u V + T_{-u} V) / 2
        cos_uD_V = 0.5 * (V_shifted_pos + V_shifted_neg)

        # K_{eps,u} = G_eps cos(uD) G_eps
        # = (V diag(g) V^T) cos(uD) (V diag(g) V^T)
        # = V diag(g) (V^T cos(uD) V) diag(g) V^T
        # Middle part: M = V^T cos(uD) V, but cos(uD) V is already computed
        # So: V^T (cos(uD) V) gives the middle matrix
        M = V.T @ cos_uD_V  # (N, N) in eigenbasis

        # Apply g on both sides: diag(g) M diag(g)
        gMg = (g[:, np.newaxis] * M) * g[np.newaxis, :]

        # Back to position basis: V gMg V^T
        return V @ gMg @ V.T
