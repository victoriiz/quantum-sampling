"""
Question to answer after first run for a commmon-shock spike model: Does a symmetry-matched ansatz beat the exact product-family ceiling (~418x) on mixture problems?

Three proposals with same KL objective and eval
    1. HEA (baseline)   : 5-layer RY + linear CNOT chain, 50 params.
    2. Shared-ring      : permutation-shared RY angle per layer + ring of CZ entanglers, 5 params, cyclic-symmetric
    3. Dicke-sector     : direct parameterization of the 11 Hamming-sector probabilities via softmax; 
                          sampled dist q(b) = s_k / C(10, k) is exactly what superposition over Dicke states |D^10_l> with amplitudes
                          sqrt(s_k) produces.

success criteria: VRF > 418.1
"""
import numpy as np
from math import comb
import pennylane as qml
from pennylane import numpy as pnp
import correlated_model as cm

N = cm.N
BUDGET = 200000
TRIALS = 20
CEILING = 418.1 # from earlier simulated result, checking up to 0.95

HW = cm.hamming_weights()
MASK = cm.failure_mask()
P_TGT = cm.p_vector() * MASK
PF = cm.exact_p_fail()
P_STAR = cm.p_star()
FLOOR = float(-np.sum(P_STAR * np.log(P_STAR + 1e-300)))
CHOOSE = np.array([comb(N, k) for k in range(N+1)], dtype=float)

# HEA
def train_hea(layers=5, steps=300, seed=42):
    dev = qml.device("default.qubit", wires=N)
    ps = pnp.array(P_STAR, requires_grad=False)

    @qml.qnode(dev)
    def probs(params):
        params = params.reshape((layers, N))
        for L in range(layers):
            for i in range(N):
                qml.RY(params[L, i], wires=i)
            for i in range(N-1):
                qml.CNOT(wires=[i, i+1])
        return qml.probs(wires=range(N))
    
    np.random.seed(seed)
    params = pnp.array(np.random.uniform(0, 2*np.pi, layers*N), requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    for _ in range(steps):
        params, _ = opt.step_and_cost(
            lambda p: -pnp.sum(ps * pnp.log(probs(p) + 1e-12)), params
        )
    q = np.array(probs(params))
    return q, float(-np.sum(P_STAR * np.log(q + 1e-12)))

# shared-angle ring
def train_ring(layers=5, steps=400, seed=42):
    dev = qml.device("default.qubit", wires=N)
    ps = pnp.array(P_STAR, requires_grad=False)

    @qml.qnode(dev)
    def probs(params):
        for L in range(layers):
            for i in range(N):
                qml.RY(params[L], wires=i) # SAME theta on every qubit
            for i in range(N):
                qml.CZ(wires=[i, (i+1) % N])
        return qml.probs(wires=range(N))
    
    np.random.seed(seed)
    params = pnp.array(np.random.uniform(0, 2*np.pi, layers),
                       requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    for _ in range(steps):
        params, _ = opt.step_and_cost(
            lambda p: -pnp.sum(ps * pnp.log(probs(p) + 1e-12)), params
        )
    q = np.array(probs(params))
    return q, float(-np.sum(P_STAR * np.log(q + 1e-12)))

# Dicke-sector model
def train_dicke(steps=3000, lr=0.5, seed=0):
    """
    Softmax over 11 sector probs, exact KL gradient descent.
    q(b) = s_{|b|} / C(N, |b|).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.1, N+1)
    t = np.array([P_STAR[HW ==k].sum() for k in range(N+1)])
    for _ in range(steps):
        s = np.exp(z - z.max()); s /= s.sum()
        grad = s - t # d/dx of KL(t||s), exact
        z -= lr * grad
    s = np.exp(z - z.max()); s /= s.sum()
    q = s[HW] / CHOOSE[HW]
    return q, float(-np.sum(P_STAR * np.log(q + 1e-12)))

# EVAL!
def evaluate(name, q, kl):
    ests, vs, esss = [], [], []
    for t in range(TRIALS):
        r = np.random.default_rng(1000 + t)
        idx = r.choice(2 ** N, size=BUDGET, p=q)
        w = np.where(q[idx] > 0, P_TGT[idx] / q[idx], 0.0)
        ests.append(w.mean()); vs.append(w.var(ddof=1))
        esss.append(w.sum() ** 2 / (w ** 2).sum() if (w ** 2).sum() else 0)
    m, v = np.mean(ests), np.mean(vs)
    ci = 1.96 * np.std(ests, ddof=1) / np.sqrt(TRIALS)
    vrf = PF * (1 - PF) / v if v > 0 else float("inf")
    beat = "BEATS CEILING" if vrf > CEILING else "below ceiling"
    print(f"{name:<22} KL gap={kl - FLOOR:6.3f} nats | "
          f"est={m:.4e}+/-{ci:.1e} (bias {100*(m-PF)/PF:+.2f}%) | "
          f"VRF={vrf:>12,.1f}x  [{beat}] | ESS={np.mean(esss):>9,.0f}")


if __name__ == "__main__":
    print(f"exact P(fail)={PF:.6e}   product-family ceiling={CEILING}x   "
          f"KL floor={FLOOR:.4f}\n")
    for name, fn in (("A: HEA 50-param", train_hea),
                     ("B: shared-ring 5-param", train_ring),
                     ("C: Dicke-sector 11-param", train_dicke)):
        q, kl = fn()
        evaluate(name, q, kl)
