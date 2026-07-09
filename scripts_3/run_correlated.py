"""
train + evaluate on the common-shock model (similar to sum of squares, but with correlations?)

classical baselines:
- BEST single iid tilt
- naive MC

quantum algo: same 10-qubit RY/CNOT + forward KL pipeline as scripts_2.
hypothesis: once p(b) is a two-temperature miixture, the entire product-form family is misspecified, 
and an entangling circuit, which CAN represent mixtures of amplitude patterns, can close that gap.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import correlated_model as cm

N = cm.N
N_LAYERS = 5
STEPS = 300
BUDGET = 200000
TRIALS = 20
SEED = 42

MASK = cm.failure_mask()
P_VEC = cm.p_vector()
P_TGT = P_VEC * MASK
PF = cm.exact_p_fail()
P_STAR = pnp.array(cm.p_star(), requires_grad=False)
HW = cm.hamming_weights()

def train_kl():
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def probs(params):
        params = params.reshape((N_LAYERS, N))
        for L in range(N_LAYERS):
            for i in range(N):
                qml.RY(params[L, i], wires=i)
            for i in range(N-1):
                qml.CNOT(wires=[i, i+1])
        return qml.probs(wires=range(N))
    
    def cost(params):
        return -pnp.sum(P_STAR * pnp.log(probs(params) + 1e-12))
    
    np.random.seed(SEED)
    params = pnp.array(np.random.uniform(0, 2*np.pi, N_LAYERS*N), requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    floor = float(-np.sum(cm.p_star() * np.log(cm.p_star() + 1e-300)))
    for s in range(STEPS):
        params, loss = opt.step_and_cost(cost, params)
        if s % 75 == 0 or s == STEPS-1:
            print(f"  [KL] step {s:3d} loss={float(loss):.4f} "
                  f"(floor {floor:.4f})")
    return np.array(probs(params))

def best_single_tilt():
    """Exact per-sample weight variance of tilt p', minimized by grid.
    var = E_q[w^2] - PF^2 with E_q[w^2] = sum_b p_tgt(b)^2 / q(b)."""
    grid = np.linspace(0.02, 0.95, 1000)
    best, best_v = None, np.inf
    for pt in grid:
        q = (pt ** HW) * ((1 - pt) ** (N - HW))
        ew2 = float(np.sum(np.where(q > 0, P_TGT ** 2 / q, 0.0)))
        v = ew2 - PF ** 2
        if v < best_v:
            best, best_v = pt, v
    return best, best_v


def tilt_trial(rng, pt, budget=BUDGET):
    b = rng.random((budget, N)) < pt
    k = b.sum(axis=1)
    q_of_k = (pt ** k) * ((1 - pt) ** (N - k))
    p_hi = (cm.P_HI ** k) * ((1 - cm.P_HI) ** (N - k))
    p_lo = (cm.P_LO ** k) * ((1 - cm.P_LO) ** (N - k))
    p_of_k = cm.P_SHOCK * p_hi + (1 - cm.P_SHOCK) * p_lo
    return (p_of_k / q_of_k) * (k >= cm.K_FAIL)


def q_trial(rng, q, budget=BUDGET):
    idx = rng.choice(2 ** N, size=budget, p=q)
    return np.where(q[idx] > 0, P_TGT[idx] / q[idx], 0.0)


def naive_trial(rng, budget=BUDGET):
    shock = rng.random(budget) < cm.P_SHOCK
    pr = np.where(shock, cm.P_HI, cm.P_LO)[:, None]
    k = (rng.random((budget, N)) < pr).sum(axis=1)
    return (k >= cm.K_FAIL).astype(float)


def summarize(name, fn):
    ests, vs, esss = [], [], []
    for t in range(TRIALS):
        w = fn(np.random.default_rng(1000 + t))
        ests.append(w.mean()); vs.append(w.var(ddof=1))
        sw, sw2 = w.sum(), (w ** 2).sum()
        esss.append(sw * sw / sw2 if sw2 > 0 else 0.0)
    m = np.mean(ests)
    ci = 1.96 * np.std(ests, ddof=1) / np.sqrt(TRIALS)
    v = np.mean(vs)
    vrf = PF * (1 - PF) / v if v > 0 else float("inf")
    print(f"{name:<30} est={m:.4e} +/-{ci:.1e} "
          f"(bias {100*(m-PF)/PF:+.2f}%)  VRF={vrf:>10,.1f}x  "
          f"ESS={np.mean(esss):>9,.0f}/{BUDGET:,}")


if __name__ == "__main__":
    s, b = cm.component_split()
    print(f"Exact P(fail) = {PF:.6e}  "
          f"(shock {s/PF:.1%} / background {b/PF:.1%})\n")
    pt, v_exact = best_single_tilt()
    vrf_theory = PF * (1 - PF) / v_exact
    print(f"best possible single tilt: p'={pt:.3f}, exact VRF ceiling "
          f"for the ENTIRE product family = {vrf_theory:,.1f}x\n")
    print("training quantum proposal (KL) ...")
    q = train_kl()
    np.save("q_corr_kl.npy", q)
    print()
    summarize("naive MC", naive_trial)
    summarize(f"best single tilt (p'={pt:.2f})",
              lambda r: tilt_trial(r, pt))
    summarize("quantum IS (KL)", lambda r: q_trial(r, q))
