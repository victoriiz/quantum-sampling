"""
the stacked experiment: structure-matched proposal + amplitude estimation

target relative error eps = 0.5%. Rows = proposal, columns = estimator.
Every cell reports the budget (samples or oracle queries) to reach eps;

proposal:   best single tilt (p' = 0.678) | compiled Dicke (delta=1e-3 amplitude error, realistic compiled-circuit case)
estimators: classical average             | MLE-QAE (noiseless orace)

classical budgets are analytic from exact per-sample variance,
QAE budgets are simulated (40 reps per ladder, smallest ladder achieving eps.)
    Payoff for QAE: Z(b) = p(b)1_fail(b) / q(b)
"""
import numpy as np
from math import asin, sin, sqrt
import correlated_model as cm

N = cm.N
HW = cm.hamming_weights()
MASK = cm.failure_mask()
P_TGT = cm.p_vector() * MASK
PF = cm.exact_p_fail()
P_STAR = cm.p_star()
EPS = 0.005
RNG = np.random.default_rng(3)
GRID = np.linspace(1e-6, np.pi / 2 - 1e-6, 400_000)

# proposals
def q_tilt(pt=0.678):
    return (pt ** HW) * ((1-pt) ** (N-HW))

def q_dicke(delta=1e-3, seed=5):
    rng = np.random.default_rng(seed)
    amp = np.sqrt(P_STAR) * (1 + delta*rng.normal(size=P_STAR.shape))
    return amp ** 2 / np.sum(amp ** 2)

def classical_budget(q):
    var = float(np.sum(np.where(q > 0, P_TGT ** 2 / q, 0.0))) - PF ** 2
    return int(np.ceil(var / (EPS * PF) ** 2)), var

# QAE budgets (simulated) 
def qae_budget(q, reps=40, shots=60):
    sup = q > 1e-15
    Z = np.where(sup, P_TGT / np.where(sup, q, 1.0), 0.0)
    Zmax = Z.max()
    mu = float(np.sum(q * Z)) / Zmax          # = PF / Zmax
    th = asin(sqrt(mu))
    for J in range(1, 11):
        powers = [0] + [2 ** j for j in range(J)]
        errs, queries = [], 0
        for _ in range(reps):
            ll = np.zeros_like(GRID); queries = 0
            for m in powers:
                p1 = sin((2 * m + 1) * th) ** 2
                h = RNG.binomial(shots, p1)
                pg = np.sin((2 * m + 1) * GRID) ** 2
                ll += h * np.log(pg + 1e-300) \
                    + (shots - h) * np.log(1 - pg + 1e-300)
                queries += shots * (2 * m + 1)
            est = Zmax * sin(GRID[np.argmax(ll)]) ** 2
            errs.append((est - PF) / PF)
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        if rmse <= EPS:
            return queries, mu, rmse
    return None, mu, rmse

if __name__ == "__main__":
    print(f"target: relative error {EPS:.1%} on P_fail = {PF:.4e}\n")
    naive = int(np.ceil((1 - PF) / (PF * EPS ** 2)))
    print(f"{'pipeline':<38}{'budget to eps':>16}")
    print("-" * 56)
    print(f"{'naive MC + classical':<38}{naive:>16,}")
    rows = []
    for pname, q in (("best tilt (p'=0.678)", q_tilt()),
                     ("compiled Dicke (delta=1e-3)", q_dicke())):
        n_cl, var = classical_budget(q)
        print(f"{pname + ' + classical':<38}{n_cl:>16,}")
        n_q, mu, rmse = qae_budget(q)
        tag = f"{n_q:,}" if n_q else ">10^ladder"
        print(f"{pname + ' + QAE':<38}{tag:>16}"
              f"    (mu={mu:.4f}, achieved rmse={rmse:.3%})")
        rows.append((pname, n_cl, n_q, mu))
    print("\nattribution (vs naive MC + classical):")
    for pname, n_cl, n_q, mu in rows:
        print(f"  proposal '{pname}': {naive / n_cl:>12,.0f}x from proposal"
              + (f"; further {n_cl / n_q:,.1f}x from QAE "
                 f"(total {naive / n_q:,.0f}x)" if n_q else ""))

