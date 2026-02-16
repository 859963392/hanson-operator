"""
Command-line interface for The Hanson Operator.

Usage:
    python -m the_hanson_operator validate
    python -m the_hanson_operator scan --t-min 12 --t-max 16
    python -m the_hanson_operator convergence
    python -m the_hanson_operator plot --output spectrum.png
    python -m the_hanson_operator info
"""

import sys
import time
import click
import numpy as np

from .config import SystemConfig
from .engine import AnomalyEngine
from .invariants import run_all_invariants, VacuumBreachError
from .display import (
    format_invariant_report,
    format_scan_result,
    format_convergence_report,
    plot_energy_spectrum,
)


def _build_config(preset, epsilon, max_n, grid_points) -> SystemConfig:
    """Build SystemConfig from CLI options."""
    if preset == "quick":
        cfg = SystemConfig.quick()
    elif preset == "high":
        cfg = SystemConfig.high_precision()
    else:
        cfg = SystemConfig.default()

    # Override with explicit options if given
    if epsilon is not None:
        cfg = SystemConfig(
            grid_points=cfg.grid_points, grid_extent=cfg.grid_extent,
            epsilon=epsilon, max_n=cfg.max_n,
            bg_quadrature_points=cfg.bg_quadrature_points,
            bg_cutoff=cfg.bg_cutoff,
        )
    if max_n is not None:
        cfg = SystemConfig(
            grid_points=cfg.grid_points, grid_extent=cfg.grid_extent,
            epsilon=cfg.epsilon, max_n=max_n,
            bg_quadrature_points=cfg.bg_quadrature_points,
            bg_cutoff=cfg.bg_cutoff,
        )
    if grid_points is not None:
        cfg = SystemConfig(
            grid_points=grid_points, grid_extent=cfg.grid_extent,
            epsilon=cfg.epsilon, max_n=cfg.max_n,
            bg_quadrature_points=cfg.bg_quadrature_points,
            bg_cutoff=cfg.bg_cutoff,
        )

    return cfg


@click.group()
def main():
    """The Hanson Operator -- Spectral Containment Field."""
    pass


@main.command()
@click.option("--preset", type=click.Choice(["quick", "default", "high"]),
              default="default", help="Configuration preset.")
@click.option("--epsilon", type=float, default=None, help="Heat parameter.")
@click.option("--grid-points", type=int, default=None, help="Grid resolution N.")
def validate(preset, epsilon, grid_points):
    """Run all invariant checks."""
    cfg = _build_config(preset, epsilon, None, grid_points)
    click.echo(f"Configuration: N={cfg.grid_points}, L={cfg.grid_extent}, "
               f"eps={cfg.epsilon}, max_n={cfg.max_n}")
    click.echo("Running invariant checks...\n")

    t0 = time.time()
    try:
        results = run_all_invariants(cfg)
    except VacuumBreachError as e:
        click.echo(f"\nFATAL: {e}", err=True)
        sys.exit(1)

    elapsed = time.time() - t0
    click.echo(format_invariant_report(results))
    click.echo(f"\nCompleted in {elapsed:.1f}s")

    if not all(r.passed for r in results):
        sys.exit(1)


@main.command()
@click.option("--t-min", type=float, default=0.0, help="Start of scan range.")
@click.option("--t-max", type=float, default=35.0, help="End of scan range.")
@click.option("--points", type=int, default=500, help="Number of scan points.")
@click.option("--preset", type=click.Choice(["quick", "default", "high"]),
              default="default", help="Configuration preset.")
@click.option("--epsilon", type=float, default=None, help="Heat parameter.")
@click.option("--max-n", type=int, default=None, help="Prime cutoff.")
@click.option("--grid-points", type=int, default=None, help="Grid resolution N.")
def scan(t_min, t_max, points, preset, epsilon, max_n, grid_points):
    """Scan the critical line for Riemann zero resonances."""
    cfg = _build_config(preset, epsilon, max_n, grid_points)
    click.echo(f"Configuration: N={cfg.grid_points}, eps={cfg.epsilon}, "
               f"max_n={cfg.max_n}")
    click.echo(f"Scanning s = 1/2 + it, t in [{t_min}, {t_max}], "
               f"{points} points...\n")

    t0 = time.time()
    engine = AnomalyEngine(cfg)

    # Progress reporting
    result = _scan_with_progress(engine, t_min, t_max, points)

    elapsed = time.time() - t0
    click.echo(format_scan_result(result))
    click.echo(f"\nCompleted in {elapsed:.1f}s")


def _scan_with_progress(engine, t_min, t_max, points):
    """Scan with progress bar using tqdm if available."""
    try:
        from tqdm import tqdm
        t_values = np.linspace(t_min, t_max, points)
        raw_traces = np.zeros(points)
        trace_moduli = np.zeros(points)
        log_energies = np.zeros(points)

        for i in tqdm(range(points), desc="Scanning", unit="pt"):
            s = 0.5 + 1j * t_values[i]
            trace = engine.compute_raw_trace(s)
            raw_traces[i] = trace.real
            trace_moduli[i] = abs(trace)
            log_energies[i] = engine.compute_log_energy(s)

        from .engine import ScanResult
        return ScanResult(
            t_values=t_values,
            raw_traces=raw_traces,
            trace_moduli=trace_moduli,
            log_energies=log_energies,
            epsilon=engine.eps,
            max_n=engine.config.max_n,
        )
    except ImportError:
        return engine.scan_critical_line(t_min, t_max, points)


@main.command()
@click.option("--t-min", type=float, default=10.0, help="Start of scan range.")
@click.option("--t-max", type=float, default=35.0, help="End of scan range.")
@click.option("--points", type=int, default=300, help="Number of scan points.")
@click.option("--preset", type=click.Choice(["quick", "default", "high"]),
              default="default", help="Configuration preset.")
@click.option("--max-n", type=int, default=None, help="Prime cutoff.")
def convergence(t_min, t_max, points, preset, max_n):
    """Run multi-epsilon convergence analysis."""
    cfg = _build_config(preset, None, max_n, None)
    click.echo(f"Convergence analysis: t in [{t_min}, {t_max}], "
               f"{points} points per epsilon")
    click.echo("Epsilon values: 0.5, 0.2, 0.1, 0.05\n")

    t0 = time.time()
    engine = AnomalyEngine(cfg)
    results = engine.multi_epsilon_scan(t_min, t_max, points)

    elapsed = time.time() - t0
    click.echo(format_convergence_report(results))
    click.echo(f"\nCompleted in {elapsed:.1f}s")


@main.command()
@click.option("--t-min", type=float, default=0.0, help="Start of scan range.")
@click.option("--t-max", type=float, default=50.0, help="End of scan range.")
@click.option("--points", type=int, default=500, help="Number of scan points.")
@click.option("--output", "-o", type=str, default=None,
              help="Save plot to file instead of displaying.")
@click.option("--preset", type=click.Choice(["quick", "default", "high"]),
              default="default", help="Configuration preset.")
@click.option("--max-n", type=int, default=None, help="Prime cutoff.")
@click.option("--multi-eps/--single-eps", default=True,
              help="Run multiple epsilon values for convergence overlay.")
def plot(t_min, t_max, points, output, preset, max_n, multi_eps):
    """Generate energy spectrum plot."""
    cfg = _build_config(preset, None, max_n, None)
    click.echo(f"Generating plot: t in [{t_min}, {t_max}]...")

    t0 = time.time()
    engine = AnomalyEngine(cfg)

    if multi_eps:
        results = engine.multi_epsilon_scan(t_min, t_max, points)
    else:
        results = [engine.scan_critical_line(t_min, t_max, points)]

    show = output is None
    plot_energy_spectrum(results, output_path=output, show=show)

    elapsed = time.time() - t0
    click.echo(f"Completed in {elapsed:.1f}s")


@main.command()
@click.option("--preset", type=click.Choice(["quick", "default", "high"]),
              default="default", help="Configuration preset.")
def info(preset):
    """Show system configuration and resource estimates."""
    if preset == "quick":
        cfg = SystemConfig.quick()
    elif preset == "high":
        cfg = SystemConfig.high_precision()
    else:
        cfg = SystemConfig.default()

    click.echo("=" * 50)
    click.echo("  THE HANSON OPERATOR - System Information")
    click.echo("=" * 50)
    click.echo(f"  Grid points (N):       {cfg.grid_points}")
    click.echo(f"  Grid extent (L):       {cfg.grid_extent}")
    click.echo(f"  Heat parameter (eps):  {cfg.epsilon}")
    click.echo(f"  Prime cutoff (max_n):  {cfg.max_n}")
    click.echo(f"  Grid spacing (dx):     {cfg.dx:.6f}")
    click.echo(f"  BG quadrature points:  {cfg.bg_quadrature_points}")
    click.echo(f"  BG cutoff:             {cfg.bg_cutoff}")

    click.echo("\n  Aliasing criterion:")
    n_min = cfg.grid_extent / (0.5 * np.sqrt(cfg.epsilon))
    status = "OK" if cfg.grid_points >= n_min else "WARNING"
    click.echo(f"    N_min = {n_min:.0f}, N = {cfg.grid_points} [{status}]")

    try:
        mem = cfg.memory_estimate_mb()
        click.echo(f"\n  Estimated memory: {mem:.1f} MB")
    except Exception:
        click.echo("\n  Memory estimate: unavailable")

    click.echo("=" * 50)
