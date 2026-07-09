import numpy as np
from math import comb
 
TAU = 65.0
NUM_VARS = 10
 
def surv(t):
    # P(Y > t) = P(X > sqrt(t)) = exp(-sqrt(t)), exact, no singularity
    return np.exp(-np.sqrt(np.clip(t, 0, None)))
 
def run(c, T_max, n_grid):
    dt = T_max / n_grid
    edges = np.arange(n_grid + 1) * dt
    bin_mass = surv(edges[:-1]) - surv(edges[1:])  # exact P(t_lo < Y <= t_hi), sums to ~surv(0)-surv(T_max)=1-tiny
 
    c2 = c * c
    t_mid = edges[:-1]  # bin assigned "low" or "high" by its left edge vs c^2 (threshold falls exactly on a bin edge below)
    g_low = np.where(edges[1:] <= c2, bin_mass, 0.0)
    g_high = np.where(edges[:-1] >= c2, bin_mass, 0.0)
    # (bins straddling c2 exactly: negligible at this resolution, c2 chosen to land on a grid edge)
 
    mass_check = g_low.sum() + g_high.sum()
 
    n_fft = 4 * n_grid
    G_low = np.fft.rfft(g_low, n=n_fft)
    G_high = np.fft.rfft(g_high, n=n_fft)
    tau_idx = int(TAU / dt)
 
    p_fail_given_k = np.zeros(NUM_VARS + 1)
    for k in range(NUM_VARS + 1):
        spec = (G_high ** k) * (G_low ** (NUM_VARS - k))
        conv = np.fft.irfft(spec, n=n_fft)
        conv = np.clip(conv, 0, None)
        tail_mass = conv[tau_idx:].sum()
        p_fail_given_k[k] = comb(NUM_VARS, k) * tail_mass
 
    return p_fail_given_k.sum(), p_fail_given_k, mass_check
 
 
if __name__ == "__main__":
    for c in [3.2, 4.2, 5.2]:
        c2 = c * c
        print("=" * 78)
        print(f"Spike threshold c = {c}")
        print("=" * 78)
        results = []
        for n_grid in [200_000, 400_000]:
            dt = c2 / round(c2 / (300.0 / n_grid))
            T_max = dt * n_grid
            p_fail_total, p_fail_given_k, mass_check = run(c, T_max, n_grid)
            results.append(p_fail_total)
            print(f"\n  grid={n_grid:>7,}  dt={dt:.6f}  mass conservation check (want 1.0): {mass_check:.8f}")
            print(f"  P(failure) [exact numerical integration] : {p_fail_total*100:.4f}%")
        rel_change = abs(results[-1] - results[-2]) / results[-2]
        print(f"  convergence: relative change between grids = {rel_change*100:.5f}%")
        print(f"  P(failure) [reference, from 20M-sample MC] : 1.4220%")
 
        p_k_given_fail = p_fail_given_k / p_fail_total
        print(f"\n  {'k':>3} | {'C(10,k)':>8} | {'P(K=k | fail)':>14}")
        for k in range(NUM_VARS + 1):
            print(f"  {k:3d} | {comb(NUM_VARS,k):8d} | {p_k_given_fail[k]*100:13.6f}%")
        mass_lt4 = p_k_given_fail[:4].sum()
        mass_ge4 = p_k_given_fail[4:].sum()
        print(f"\n  P(K < 4 | failure)  = {mass_lt4*100:.4f}%")
        print(f"  P(K >= 4 | failure) = {mass_ge4*100:.4f}%   <- what the model scores as 'failure'")