"""
The Sound of Convergence — Phase 1

Renders Re(Tr(A^2(1/2 + it))) as a 30-second audio triptych demonstrating
the physical cooling of the Hanson operator through three epsilon values.

Output: artifacts/quantum_choir.wav

Three movements, cross-faded:
  I:   The Chaos    (eps=0.5)   — warm rushing static
  II:  The Focusing (eps=0.05)  — textures emerge, vacuum gap opens
  III: The Crystal  (eps=0.005) — resonances punch through as sharp beats

Usage:
    python scripts/sonify_operator.py              # full run (~71 min)
    python scripts/sonify_operator.py --smoke       # quick test (~1 min)
"""

import sys
import os
import time
import argparse

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# Allow running as `python scripts/sonify_operator.py` without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from the_hanson_operator.config import SystemConfig
from the_hanson_operator.engine import AnomalyEngine


# --- Constants ---

SAMPLE_RATE = 44100
SCAN_SPEED = 100.0          # t-units per second of audio
T_MIN = 0.0
T_MAX = 1000.0
DURATION = 10.0             # seconds per movement
CROSSFADE_SAMPLES = 22050   # 500ms at 44.1kHz

MOVEMENTS = [
    ("I: The Chaos",     0.5),
    ("II: The Focusing", 0.05),
    ("III: The Crystal", 0.005),
]


def build_engine(eps: float, max_n: int) -> AnomalyEngine:
    """Build an AnomalyEngine for the given epsilon."""
    config = SystemConfig(max_n=max_n, epsilon=eps)
    return AnomalyEngine(config)


def compute_movement(engine: AnomalyEngine, n_samples: int,
                     chunk_size: int, label: str) -> np.ndarray:
    """
    Vectorized trace computation for one movement.

    Accesses engine internals to compute Re(Tr(A^2)) in batches via
    the bilinear form w^T @ K_full @ w, where w = [w_arith; -w_bg].

    Returns float64 array of shape (n_samples,).
    """
    dt = SCAN_SPEED / SAMPLE_RATE
    t_all = T_MIN + np.arange(n_samples) * dt

    # Extract engine internals
    K_full = engine._ktm.K_full
    pp_log_n = engine._pp_log_n
    pp_lambda = engine._pp_lambda
    u_bg = engine._u_bg
    simpson_w = engine._simpson_weights

    signal = np.empty(n_samples, dtype=np.float64)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    with tqdm(total=n_samples, desc=label, unit="smp") as pbar:
        for c in range(n_chunks):
            i0 = c * chunk_size
            i1 = min(i0 + chunk_size, n_samples)
            t_chunk = t_all[i0:i1]
            s_chunk = 0.5 + 1j * t_chunk  # (C,)

            # Vectorized weight computation
            W_a = pp_lambda[:, None] * np.exp(
                -0.5 * s_chunk[None, :] * pp_log_n[:, None]
            )  # (P, C)
            W_b = (
                np.exp((1.0 - 0.5 * s_chunk[None, :]) * u_bg[:, None])
                * simpson_w[:, None]
            )  # (U, C)
            W = np.concatenate([W_a, -W_b], axis=0)  # (D, C)

            # Bilinear form: traces[i] = W[:,i]^T @ K_full @ W[:,i]
            KW = K_full @ W           # (D, C) — the BLAS call
            traces = np.sum(W * KW, axis=0)  # (C,)

            signal[i0:i1] = traces.real
            pbar.update(i1 - i0)

    return signal


def process_audio(signal: np.ndarray) -> np.ndarray:
    """DC removal + normalize to -3dB peak."""
    # Remove DC offset
    signal = signal - np.mean(signal)

    # Normalize to -3dB (peak amplitude = 10^(-3/20) ≈ 0.7079)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= 0.7079 / peak

    return signal


def crossfade(movements: list) -> np.ndarray:
    """
    Equal-power sinusoidal crossfade between adjacent movements.

    sin^2 + cos^2 = 1 preserves perceived loudness through transitions.
    """
    # Clamp crossfade to half the shortest movement
    max_fade = min(len(m) for m in movements) // 2
    F = min(CROSSFADE_SAMPLES, max_fade)
    t_fade = np.arange(F, dtype=np.float64) / F
    fade_out = np.cos(t_fade * np.pi / 2)
    fade_in = np.sin(t_fade * np.pi / 2)

    # Start with first movement
    result = movements[0].copy()

    for i in range(1, len(movements)):
        nxt = movements[i]

        # Apply crossfade to overlap region
        result[-F:] *= fade_out
        blended = result[-F:] + nxt[:F] * fade_in

        # Concatenate: everything before the fade + blended + rest of next
        result = np.concatenate([result[:-F], blended, nxt[F:]])

    return result


def main():
    parser = argparse.ArgumentParser(description="Sonify the Hanson Operator")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test with reduced parameters")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: artifacts/quantum_choir.wav)")
    args = parser.parse_args()

    # Parameters
    if args.smoke:
        max_n = 5000
        n_samples = 1000
        chunk_size = 500
        print("SMOKE TEST MODE: reduced parameters for quick verification\n")
    else:
        max_n = 50000
        n_samples = int(DURATION * SAMPLE_RATE)  # 441,000
        chunk_size = None  # auto-select per movement
        print(f"Full render: {n_samples} samples/movement, max_n={max_n}\n")

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "quantum_choir.wav")

    total_t0 = time.time()
    processed = []

    for label, eps in MOVEMENTS:
        print(f"\n{'='*60}")
        print(f"  {label}  (eps={eps})")
        print(f"{'='*60}")

        # Auto-select chunk size based on dimension
        if chunk_size is None:
            cs = 3000 if eps < 0.01 else 5000
        else:
            cs = chunk_size

        t0 = time.time()
        engine = build_engine(eps, max_n)
        dim = engine._ktm._P + engine._ktm._U
        print(f"  Engine: dim={dim}, K_full={engine._ktm.memory_usage_mb():.0f} MB")

        signal = compute_movement(engine, n_samples, cs, label)
        elapsed = time.time() - t0
        print(f"  Computed in {elapsed:.1f}s "
              f"({elapsed/n_samples*1000:.3f} ms/sample)")

        # Free engine memory before next movement
        del engine

        processed.append(process_audio(signal))
        print(f"  Processed: DC removed, normalized to -3dB")

    # Crossfade
    print(f"\n{'='*60}")
    print(f"  Crossfading movements ({CROSSFADE_SAMPLES/SAMPLE_RATE*1000:.0f}ms overlap)")
    print(f"{'='*60}")

    if len(processed) > 1:
        final = crossfade(processed)
    else:
        final = processed[0]

    duration_s = len(final) / SAMPLE_RATE
    print(f"  Final length: {len(final)} samples ({duration_s:.1f}s)")

    # Write WAV (float32)
    wavfile.write(out_path, SAMPLE_RATE, final.astype(np.float32))
    file_size = os.path.getsize(out_path)
    total_elapsed = time.time() - total_t0

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {out_path}")
    print(f"  Size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Duration: {duration_s:.1f}s")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Total time: {total_elapsed / 60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
