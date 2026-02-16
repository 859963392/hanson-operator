"""
Prime number stream and von Mangoldt function.

Provides the arithmetic input channel: the von Mangoldt distribution
Lambda(n) for all prime powers n >= 2.
"""

import numpy as np
from typing import Iterator, Tuple


class PrimeStream:
    def __init__(self, capacity: int = 10_000):
        if capacity < 2:
            raise ValueError(f"capacity must be >= 2, got {capacity}")
        self.capacity = capacity
        self._primes = self._sieve(capacity)
        self._lambda_cache = self._precompute_lambdas(capacity)

    def _sieve(self, n: int) -> np.ndarray:
        """Sieve of Eratosthenes returning array of primes up to n."""
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0:2] = False
        for i in range(2, int(np.sqrt(n)) + 1):
            if is_prime[i]:
                is_prime[i * i :: i] = False
        return np.nonzero(is_prime)[0]

    def _precompute_lambdas(self, n: int) -> np.ndarray:
        """
        Precomputes Lambda(k) for k in [0, n].
        Lambda(n) = log(p) if n = p^k for some prime p, else 0.
        """
        lambdas = np.zeros(n + 1)
        for p in self._primes:
            log_p = np.log(float(p))
            k = int(p)
            while k <= n:
                lambdas[k] = log_p
                k *= p
        return lambdas

    @property
    def primes(self) -> np.ndarray:
        return self._primes

    @property
    def lambda_cache(self) -> np.ndarray:
        return self._lambda_cache

    def num_prime_powers(self, max_n: int) -> int:
        """Count of prime powers in [2, max_n]."""
        if max_n > self.capacity:
            raise ValueError(f"max_n={max_n} exceeds capacity={self.capacity}")
        return int(np.count_nonzero(self._lambda_cache[2 : max_n + 1]))

    def get_fluctuations(self, max_n: int) -> Iterator[Tuple[float, float]]:
        """Yields (log(n), Lambda(n)) pairs for all prime powers n in [2, max_n]."""
        if max_n > self.capacity:
            raise ValueError(f"max_n={max_n} exceeds capacity={self.capacity}")
        indices = np.nonzero(self._lambda_cache[: max_n + 1])[0]
        indices = indices[indices >= 2]
        for n in indices:
            yield np.log(float(n)), self._lambda_cache[n]

    def get_arrays(self, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (log_n, lambda_n) as numpy arrays for all prime powers in [2, max_n].
        Suitable for vectorised computation.
        """
        if max_n > self.capacity:
            raise ValueError(f"max_n={max_n} exceeds capacity={self.capacity}")
        indices = np.nonzero(self._lambda_cache[: max_n + 1])[0]
        indices = indices[indices >= 2]
        log_n = np.log(indices.astype(np.float64))
        lambda_n = self._lambda_cache[indices]
        return log_n, lambda_n
