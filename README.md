# The Hanson Operator

Proof of the Riemann Hypothesis via Trace-Class Regularisation of the Prime Field and Spectral Stability of the Mehler-Zeta Operator

https://doi.org/10.5281/zenodo.18212540

This repository contains the Python implementation of the anomaly operator Aε(s) described in the paper.

## Requirements

- Python 3.10+
- 8GB RAM (peak usage ~150MB during scans)

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

For plotting support (requires matplotlib):

```bash
pip install -e ".[plot]"
```

For development (tests + plotting):

```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Validate Invariants

Check that the operator satisfies the trace class condition and vacuum dominance (Section 5 of the paper):

```bash
python -m the_hanson_operator validate
```

### Scan the Critical Line

This scans s = 1/2 + it for t in [0, 35] with default parameters (eps=0.05, max_n=10000) and detects resonances in the anomaly energy

```bash
python -m the_hanson_operator scan
```

### Hear the Operator

Run the sonification script to generate the audio triptych (quantum_choir.wav) demonstrating the operator's cooling phase:

```bash
python scripts/sonify_operator.py
```

## Commands

### Validate

Run all four invariant checks: trace class, vacuum dominance, renormalisation rigidity, and Hilbert-Schmidt decay.

```bash
python -m the_hanson_operator validate
python -m the_hanson_operator validate --preset high
python -m the_hanson_operator validate --epsilon 0.03 --grid-points 600
```

### Scan

Scan the critical line and detect resonances via local normalisation.

```bash
python -m the_hanson_operator scan --t-min 12 --t-max 16 --points 500
python -m the_hanson_operator scan --max-n 50000
python -m the_hanson_operator scan --preset quick
```

Options:
- `--t-min`, `--t-max`: scan range (default 0 to 35)
- `--points`: number of sample points (default 500)
- `--max-n`: prime cutoff, higher = more primes = sharper resolution
- `--epsilon`: heat parameter, smaller = finer spectral resolution
- `--preset`: `quick` (fast, lower resolution), `default`, or `high` (slow, highest resolution)

### Convergence

Run the same scan at multiple epsilon values (0.5, 0.2, 0.1, 0.05) to demonstrate that resonance structure sharpens as the regularisation is removed.

```bash
python -m the_hanson_operator convergence
python -m the_hanson_operator convergence --t-min 12 --t-max 20
```

### Plot

Generate an energy spectrum plot (requires matplotlib).

```bash
python -m the_hanson_operator plot
python -m the_hanson_operator plot -o spectrum.png
python -m the_hanson_operator plot --single-eps --t-min 10 --t-max 25
```

### Information

Display current configuration and resource estimates.

```bash
python -m the_hanson_operator info
python -m the_hanson_operator info --preset high
```

## Configuration Presets

| Preset | Grid N | Epsilon | max_n | Use case |
|--------|--------|---------|-------|----------|
| `quick` | 200 | 0.1 | 5000 | Fast checks, ~1s scans |
| `default` | 400 | 0.05 | 10000 | Standard analysis, ~4s scans |
| `high` | 800 | 0.02 | 50000 | Publication quality, ~30s scans |

## Interpreting Results

The scan reports **resonances** -- peaks in the locally normalised trace energy that rise above the background. Each resonance is matched to the nearest known Riemann zero.

Two types of peaks appear:

- **Physical zeros**: peaks that stay fixed as you change `max_n`. These correspond to actual Riemann zeros.
- **Cutoff ringing**: artifact peaks spaced at approximately `2*pi / log(max_n)` that shift when you change `max_n`. These are finite truncation effects and are not physical.

To distinguish them, run the same scan at two different `max_n` values and see which peaks move.

## Architecture

```
the_hanson_operator/
  config.py       # SystemConfig with validation and presets
  primes.py       # Prime sieve and von Mangoldt function
  kernel.py       # Exact scalar kernel trace (Proposition 3.2)
  operators.py    # Eigendecomposition-based Hamiltonian
  engine.py       # AnomalyEngine: scanning and energy computation
  invariants.py   # Four invariant validators
  display.py      # Output formatting and plotting
  cli.py          # Click CLI entry point
  
scripts/
  sonify_operator.py  # Audio generation script
```

The core computation uses the closed-form kernel trace formula from Proposition 3.2 of the manuscript. This reduces all energy computations to scalar quadratic forms over small vectors, keeping memory under 200MB regardless of scan parameters.

## Tests

```bash
python -m pytest tests/ -v
```
