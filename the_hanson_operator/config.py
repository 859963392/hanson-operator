"""
System configuration for the Spectral Containment Field.

Controls the three primary dials: grid resolution (N), box extent (L),
and heat parameter (epsilon). Includes auto-validation of the aliasing
criterion and memory estimation.
"""

import warnings
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    grid_points: int = 400
    grid_extent: float = 12.0
    epsilon: float = 0.05
    max_n: int = 10_000
    bg_quadrature_points: int = 300
    bg_cutoff: float = 10.0

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.grid_points < 10:
            raise ValueError(f"grid_points must be >= 10, got {self.grid_points}")
        if self.grid_extent <= 0:
            raise ValueError(f"grid_extent must be positive, got {self.grid_extent}")
        if self.max_n < 2:
            raise ValueError(f"max_n must be >= 2, got {self.max_n}")

        # Aliasing criterion: N > extent / (0.5 * sqrt(epsilon))
        # The thermal wavelength sqrt(epsilon) must be resolved by the grid spacing dx.
        n_min = self.grid_extent / (0.5 * np.sqrt(self.epsilon))
        if self.grid_points < n_min:
            warnings.warn(
                f"Grid resolution too low: grid_points={self.grid_points} < {n_min:.0f} "
                f"(required for epsilon={self.epsilon}). Thermal aliasing may occur. "
                f"Increase grid_points or increase epsilon.",
                stacklevel=2,
            )

    @property
    def dx(self) -> float:
        return (2 * self.grid_extent) / self.grid_points

    @property
    def x_axis(self) -> np.ndarray:
        return np.linspace(-self.grid_extent, self.grid_extent, self.grid_points)

    def memory_estimate_mb(self) -> float:
        """Estimate peak memory usage in MB for a default scan."""
        from .primes import PrimeStream
        ps = PrimeStream(self.max_n)
        P = ps.num_prime_powers(self.max_n)
        U = self.bg_quadrature_points
        # K_pp (P x P) + K_pb (P x U) + K_bb (U x U), all float64
        bytes_tier1 = (P * P + P * U + U * U) * 8
        # Tier 2: eigenvectors (N x N float64) + working matrices
        N = self.grid_points
        bytes_tier2 = N * N * 8 * 3
        return (bytes_tier1 + bytes_tier2) / (1024 * 1024)

    @classmethod
    def quick(cls) -> "SystemConfig":
        return cls(grid_points=200, grid_extent=12.0, epsilon=0.1, max_n=2_000,
                   bg_quadrature_points=150, bg_cutoff=8.0)

    @classmethod
    def default(cls) -> "SystemConfig":
        return cls()

    @classmethod
    def high_precision(cls) -> "SystemConfig":
        return cls(grid_points=600, grid_extent=15.0, epsilon=0.02, max_n=50_000,
                   bg_quadrature_points=500, bg_cutoff=12.0)
