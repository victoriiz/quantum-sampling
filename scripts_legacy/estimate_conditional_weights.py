"""
Estimates P(spike-count = k | failure) directly from the real continuous
model, instead of assuming failure probability mass is spread uniformly
across all 848 discrete failure states regardless of Hamming weight.
 
"Spike" is defined the same way the existing two-point quantization already
does: xi > SPIKE_THRESHOLD (= extreme_load = 4.2), tying this estimate back
to the constant already in use elsewhere rather than introducing a new
arbitrary parameter (see Problem #5 in the audit -- this also gives
extreme_load a concrete operational meaning, independent of resolving its
quantile-vs-conditional-mean ambiguity).
 
Run once, after scripts/classical_env.py (needs GROUND_TRUTH context, though
it recomputes its own MC estimate independently for the conditional shape).
"""
import os
import numpy as np
from math import comb
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
 
NUM_QUBITS = 10
TAU = 65.0
SPIKE_THRESHOLD = 4.2
N_MC = 20_000_000
SEED = 2026
 
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    x = rng.exponential(1.0, size=(N_MC, NUM_QUBITS)).astype(np.float32)
    S = (x ** 2).sum(axis=1)
    is_failure = S > TAU
    spike_count = (x > SPIKE_THRESHOLD).sum(axis=1)
 
    n_failures = int(is_failure.sum())
    print(f"Empirical P(failure) from this run: {is_failure.mean()*100:.4f}%  ({n_failures:,} failures out of {N_MC:,})")
 
    failure_spikes = spike_count[is_failure]
    counts = np.bincount(failure_spikes, minlength=NUM_QUBITS + 1)
    p_k_given_failure = counts / counts.sum()
 
    print("\nEmpirical P(spike-count = k | failure), vs. the uniform assumption it replaces:")
    print(f"{'k':>3} | {'C(10,k)':>8} | {'count':>9} | {'P(k|fail)':>10} | {'uniform-per-state (old)':>24} | {'weight-dependent p(state)':>26}")
    uniform_p_per_state = 1.0 / 848.0  # old assumption, same for every failure state regardless of k
    for k in range(4, NUM_QUBITS + 1):
        c = comb(NUM_QUBITS, k)
        p_state = (p_k_given_failure[k] / c) if c > 0 else 0.0
        print(f"{k:3d} | {c:8d} | {counts[k]:9d} | {p_k_given_failure[k]:10.4f} | {uniform_p_per_state:24.6e} | {p_state:26.6e}")
 
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "weight_conditional_probs.npy")
    np.save(out_path, p_k_given_failure)
    print(f"\nSaved P(k | failure) for k=0..10 to '{out_path}'")