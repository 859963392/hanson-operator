"""
Output formatting and optional matplotlib plotting.
"""

import sys
import numpy as np
from typing import List, Optional

from .engine import ScanResult, KNOWN_ZEROS


def format_invariant_report(results: list) -> str:
    """Format invariant check results for terminal output."""
    lines = []
    lines.append("=" * 60)
    lines.append("  INVARIANT VALIDATION REPORT")
    lines.append("=" * 60)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        marker = " [+]" if r.passed else " [X]"
        lines.append(f"{marker} {r.name}: {status}")
        lines.append(f"      {r.message}")
        if not r.passed:
            all_passed = False

    lines.append("-" * 60)
    if all_passed:
        lines.append("  SYSTEM NOMINAL: All invariants hold.")
    else:
        lines.append("  SYSTEM COMPROMISED: One or more invariants failed.")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_scan_result(result: ScanResult) -> str:
    """Format scan results for terminal output."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  CRITICAL LINE SCAN (eps={result.epsilon}, max_n={result.max_n})")
    lines.append(f"  Range: t = {result.t_values[0]:.2f} to {result.t_values[-1]:.2f}")
    lines.append(f"  Points: {len(result.t_values)}")
    lines.append("=" * 60)

    # Resonance detection via local normalisation
    resonances = result.find_resonances()
    if len(resonances) > 0:
        lines.append(f"\n  Resonances detected: {len(resonances)}")
        lines.append(f"  {'t_resonance':>12s}  {'nearest known':>14s}  {'error':>8s}")
        lines.append(f"  {'-'*12}  {'-'*14}  {'-'*8}")
        for tr in resonances:
            dists = np.abs(KNOWN_ZEROS - tr)
            nearest_idx = np.argmin(dists)
            nearest = KNOWN_ZEROS[nearest_idx]
            err = dists[nearest_idx]
            lines.append(f"  {tr:12.4f}  {nearest:14.6f}  {err:8.4f}")
    else:
        lines.append("\n  No resonances detected in scan range.")

    # Sign changes in Re(Tr(A^2))
    crossings = result.find_sign_changes()
    if len(crossings) > 0:
        lines.append(f"\n  Sign changes in Re(Tr(A^2)): {len(crossings)}")

    # Summary statistics
    lines.append("\n  Trace statistics:")
    lines.append(f"    max |Tr(A^2)|  = {np.max(result.trace_moduli):.6e}")
    lines.append(f"    mean |Tr(A^2)| = {np.mean(result.trace_moduli):.6e}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_convergence_report(results: List[ScanResult]) -> str:
    """Format multi-epsilon convergence results."""
    lines = []
    lines.append("=" * 60)
    lines.append("  CONVERGENCE ANALYSIS (multi-epsilon)")
    lines.append("=" * 60)

    for result in results:
        resonances = result.find_resonances()
        lines.append(f"\n  eps = {result.epsilon}")
        lines.append(f"    Resonances: {len(resonances)}")
        if len(resonances) > 0:
            for tr in resonances[:5]:
                dists = np.abs(KNOWN_ZEROS - tr)
                nearest_idx = np.argmin(dists)
                nearest = KNOWN_ZEROS[nearest_idx]
                lines.append(f"      t = {tr:.4f} (nearest known: {nearest:.6f}, "
                             f"err = {abs(tr - nearest):.4f})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def plot_energy_spectrum(results: List[ScanResult],
                         output_path: Optional[str] = None,
                         show: bool = True) -> None:
    """
    Plot energy spectrum with known zeros overlay.
    Requires matplotlib (optional dependency).
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: "
              "pip install matplotlib", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: |Tr(A^2)| — raw modulus
    ax1 = axes[0]
    for result in results:
        ax1.plot(result.t_values, result.trace_moduli,
                 label=f"eps={result.epsilon}", alpha=0.8)
    ax1.set_ylabel("|Tr(A^2)|")
    ax1.set_title("Anomaly Energy on the Critical Line (s = 1/2 + it)")
    ax1.legend(fontsize=8)

    # Mark known zeros
    t_min = results[0].t_values[0]
    t_max = results[0].t_values[-1]
    for gamma in KNOWN_ZEROS:
        if t_min <= gamma <= t_max:
            ax1.axvline(x=gamma, color="red", linewidth=0.7,
                        linestyle=":", alpha=0.6)

    # Bottom: locally normalised — resonance detection view
    ax2 = axes[1]
    for result in results:
        norm = result.locally_normalized()
        ax2.plot(result.t_values, norm,
                 label=f"eps={result.epsilon}", alpha=0.8)
    ax2.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Locally Normalized |Tr(A^2)|")
    ax2.set_xlabel("t (imaginary part)")
    ax2.legend(fontsize=8)

    for gamma in KNOWN_ZEROS:
        if t_min <= gamma <= t_max:
            ax2.axvline(x=gamma, color="red", linewidth=0.7,
                        linestyle=":", alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
