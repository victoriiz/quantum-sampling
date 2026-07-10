"""

"""
import numpy as np
from math import comb, exp, asin, sin, sqrt
import pennylane as qml

N, K_FAIL = 10, 4
P0 = exp(-4.2)
P_TILT = K_FAIL / N 
RNG = np.random.default_rng(7)

P_EXACT = sum(comb(N, k) * P0**k * (1-P0)**(N-k) for k in range(K_FAIL, N+1))

def LR(k):
    return (P0 / P_TILT)**k * ((1-P0) / (1-P_TILT))**(N-k)

ZMAX = LR(K_FAIL)
MU = sum(comb(N, k) * P_TILT**k * (1-P_TILT)**(N-k)*LR(k)
         for k in range(K_FAIL, N+1)) / ZMAX
THETA = asin(sqrt(MU))

def verify_state_prep():
    """Verifty state-prep step against actual circuit"""
    dev = qml.device("default.qubit", wires=N)
    @qml.qnode(dev)
    def probs():
        for i in range(N):
            qml.RY(2 * np.arcsin(np.sqrt(P_TILT)), wires=i)
        return qml.probs(wires=range(N))
    q = np.array(probs())
    hw = np.array([bin(i).count("1") for i in range(2**N)])
    q_expect = P_TILT**hw * (1 - P_TILT)**(N - hw)
    assert np.allclose(q, q_expect, atol=1e-10)
    print("state-prep check: 10x RY reproduces tilted Bern(0.4)")

# MLE-QAE
def qae_run(powers, shots):
    """
    Simulate measurement record, return MLE estimate of p.
    """
    grid = np.linspace(1e-6, np.pi / 2 - 1e-6, 200000)
    loglik = np.zeros_like(grid)
    queries = 0
    for m in powers:
        prob1 = sin((2*m + 1) * THETA) ** 2
        h = RNG.binomial(shots, prob1)
        pg = np.sin((2*m + 1) * grid) ** 2
        loglik += h * np.log(pg + 1e-300) + (shots - h) * np.log(1- pg + 1e-300)
        queries += shots * (2*m + 1)
    th = grid[np.argmax(loglik)]
    return ZMAX * sin(th)**2, queries

def classical_tilt(n_samples):
    k = RNG.binomial(N, P_TILT, size=n_samples)
    lr_tables = np.array([LR(kk) for kk in range(N+1)])
    w = lr_tables[k] * (k >= K_FAIL)
    return w.mean()

# ---- experiment: RMSE vs queries ---------------------------------------
if __name__ == "__main__":
    print(f"exact p = {P_EXACT:.6e}   Zmax = {ZMAX:.4e}   "
          f"mu = {MU:.6f}   theta = {THETA:.6f}")
    verify_state_prep()
    REPS = 60
    print(f"\n{'method':<12}{'budget (queries/samples)':>26}"
          f"{'RMSE/p':>12}")
    results = []
    for J in range(2, 8):                       # schedules of growing depth
        powers = [0] + [2**j for j in range(J)]
        shots = 60
        errs, qs = [], 0
        for _ in range(REPS):
            est, qs = qae_run(powers, shots)
            errs.append((est - P_EXACT) / P_EXACT)
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        results.append(("QAE", qs, rmse))
        print(f"{'QAE':<12}{qs:>26,}{rmse:>12.4%}")
    for n_s in [10**3, 10**4, 10**5, 10**6]:
        errs = [(classical_tilt(n_s) - P_EXACT) / P_EXACT
                for _ in range(REPS)]
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        results.append(("classical", n_s, rmse))
        print(f"{'classical':<12}{n_s:>26,}{rmse:>12.4%}")

    # slope check: fit log(rmse) vs log(budget) for each method
    for tag in ("QAE", "classical"):
        pts = [(b, r) for t, b, r in results if t == tag and r > 0]
        lb, lr_ = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        slope = np.polyfit(lb, lr_, 1)[0]
        print(f"\nfitted error-decay slope, {tag}: {slope:+.2f} "
              f"(theory: {'-1.0 (QAE)' if tag == 'QAE' else '-0.5 (MC)'})")