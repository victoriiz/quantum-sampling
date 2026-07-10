"""
(a) Noise: standard exponential-decay model for noisy Grover iteration
    (e.g. Tanaka et al. 2021): with per-application fidelity f, the
    depth-m measurement probability becomes
        P_m = f^m * sin^2((2m+1)theta) + (1 - f^m) * 1/2
    (signal decays toward a maximally-mixed coin). MLE uses the
    noise-aware likelihood with f known -- the BEST case for QAE.
(b) Translatable metric: for each precision eps, report the maximum
    cost ratio R = (time per oracle query)/(time per classical sample)
    at which QAE still wins wall-clock:  R*(eps) = N_classical(eps) /
    N_qae(eps). QAE wins in wall-clock iff its per-query slowdown
    factor is below R*.
"""
import numpy as np
from math import comb, exp, asin, sin, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, KF = 10, 4
P0, PT = exp(-4.2), 0.4
PEX = sum(comb(N, k) * P0**k * (1 - P0)**(N - k) for k in range(KF, N + 1))
LR = lambda k: (P0 / PT)**k * ((1 - P0) / (1 - PT))**(N - k)
ZMAX = LR(KF)
MU = sum(comb(N, k) * PT**k * (1 - PT)**(N - k) * LR(k)
         for k in range(KF, N + 1)) / ZMAX
TH = asin(sqrt(MU))
RNG = np.random.default_rng(11)
GRID = np.linspace(1e-6, np.pi / 2 - 1e-6, 200_000)
REPS = 60


def qae_rmse(f, J, shots=60):
    powers = [0] + [2**j for j in range(J)]
    errs, queries = [], 0
    for _ in range(REPS):
        ll = np.zeros_like(GRID); queries = 0
        for m in powers:
            decay = f ** m
            p1 = decay * sin((2 * m + 1) * TH) ** 2 + (1 - decay) * 0.5
            h = RNG.binomial(shots, p1)
            pg = decay * np.sin((2 * m + 1) * GRID) ** 2 + (1 - decay) * 0.5
            ll += h * np.log(pg + 1e-300) \
                + (shots - h) * np.log(1 - pg + 1e-300)
            queries += shots * (2 * m + 1)
        est = ZMAX * sin(GRID[np.argmax(ll)]) ** 2
        errs.append((est - PEX) / PEX)
    return queries, float(np.sqrt(np.mean(np.square(errs))))


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    print(f"{'fidelity f':>10} {'slope':>8}   (per-Grover-application)")
    curves = {}
    for f, c in [(1.0, "#2a7"), (0.999, "#27b"), (0.99, "#e90"),
                 (0.95, "#c44")]:
        pts = [qae_rmse(f, J) for J in range(2, 8)]
        x, y = zip(*pts)
        s = np.polyfit(np.log(x), np.log(y), 1)[0]
        curves[f] = pts
        ax.loglog(x, y, "o-", color=c,
                  label=f"QAE, f={f} (slope {s:+.2f})")
        print(f"{f:>10} {s:>+8.2f}")
    # classical reference: rmse = sigma_rel / sqrt(N), sigma from theory
    var_w = sum(comb(N, k) * PT**k * (1 - PT)**(N - k) * LR(k)**2
                for k in range(KF, N + 1)) - PEX**2
    ns = np.logspace(2.5, 6.5, 20)
    cl = np.sqrt(var_w / ns) / PEX
    ax.loglog(ns, cl, "--", color="black",
              label="classical tilted MC (slope -0.50, analytic)")
    ax.set_xlabel("budget: oracle queries / samples")
    ax.set_ylabel("relative RMSE (log)")
    ax.set_title("Noise bends the QAE advantage back to classical scaling\n"
                 "(depolarizing Grover model, noise-aware MLE, f known)")
    ax.legend(fontsize=9); ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout(); fig.savefig("fig5_noise_sweep.png", dpi=200)

    # translatable metric: parity ratio R*(eps) = N_cl(eps)/N_qae(eps)
    print("\nwall-clock parity ratio R* (QAE wins iff per-query slowdown "
          "< R*):")
    print(f"{'target eps':>12} {'f=1.0':>10} {'f=0.999':>10} {'f=0.99':>10}")
    for eps in [0.02, 0.01, 0.005, 0.002]:
        n_cl = var_w / (eps * PEX) ** 2
        row = f"{eps:>12}"
        for f in [1.0, 0.999, 0.99]:
            xs, ys = zip(*curves[f])
            ok = [x for x, y in zip(xs, ys) if y <= eps]
            row += f"{(n_cl / min(ok)):>10.1f}" if ok else f"{'--':>10}"
        print(row)