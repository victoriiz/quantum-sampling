"""
run_network.py -- Train (KL) + evaluate on the network reliability model.

Baselines, strongest first:
  - Cross-entropy (CE) IS: iteratively fitted PER-EDGE tilt vector
    (Rubinstein & Kroese, 'The Cross-Entropy Method', 2004; applied to
    network reliability in Hui, Bean, Kraetzl & Kroese 2005). This is
    the honest classical opponent here -- unlike the spike model there
    is no closed-form optimal tilt, but CE with a product family is the
    standard practical method.
  - naive MC.
Quantum: 12-qubit RY/CNOT ansatz trained with forward KL toward p*.
ESS = (sum w)^2 / sum w^2 reported for every method (Owen ch. 9).
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import network_model as nm

N = nm.N_EDGES
N_LAYERS = 5
STEPS = 200
BUDGET = 100_000
TRIALS = 20
SEED = 42

MASK = nm.failure_mask()
P_VEC = nm.p_vector()
P_TGT = P_VEC * MASK
PF = nm.exact_p_fail()
P_STAR = pnp.array(nm.p_star(), requires_grad=False)


# ------------------------- quantum: KL training --------------------------
def train_kl():
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def probs(params):
        params = params.reshape((N_LAYERS, N))
        for L in range(N_LAYERS):
            for i in range(N):
                qml.RY(params[L, i], wires=i)
            for i in range(N - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.probs(wires=range(N))

    def cost(params):
        return -pnp.sum(P_STAR * pnp.log(probs(params) + 1e-12))

    np.random.seed(SEED)
    params = pnp.array(np.random.uniform(0, 2 * np.pi, N_LAYERS * N),
                       requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    floor = float(-np.sum(nm.p_star() * np.log(nm.p_star() + 1e-300)))
    for s in range(STEPS):
        params, loss = opt.step_and_cost(cost, params)
        if s % 50 == 0 or s == STEPS - 1:
            print(f"  [KL] step {s:3d} loss={float(loss):.4f} "
                  f"(floor {floor:.4f})")
    return np.array(probs(params))


# ------------------------- classical: CE tilts ---------------------------
def ce_fit(iters=8, batch=50_000, rho=0.1, seed=0):
    """Fit per-edge tilt vector theta by cross-entropy iterations."""
    rng = np.random.default_rng(seed)
    theta = np.full(N, 0.25)                       # start well-tilted
    for _ in range(iters):
        b = rng.random((batch, N)) < theta
        ints = b @ (1 << np.arange(N))
        fail = MASK[ints].astype(bool)
        if fail.sum() < 50:
            theta = np.minimum(theta * 1.5, 0.5)
            continue
        lr = np.prod(np.where(b, nm.P_EDGE_FAIL / theta,
                              (1 - nm.P_EDGE_FAIL) / (1 - theta)), axis=1)
        w = lr * fail
        theta = np.clip((w[:, None] * b).sum(0) / w.sum(), 1e-4, 0.5)
    return theta


def ce_trial(rng, theta, budget=BUDGET):
    b = rng.random((budget, N)) < theta
    ints = b @ (1 << np.arange(N))
    lr = np.prod(np.where(b, nm.P_EDGE_FAIL / theta,
                          (1 - nm.P_EDGE_FAIL) / (1 - theta)), axis=1)
    w = lr * MASK[ints]
    return w


def q_trial(rng, q, budget=BUDGET):
    idx = rng.choice(2 ** N, size=budget, p=q)
    return np.where(q[idx] > 0, P_TGT[idx] / q[idx], 0.0)


def naive_trial(rng, budget=BUDGET):
    b = rng.random((budget, N)) < nm.P_EDGE_FAIL
    ints = b @ (1 << np.arange(N))
    return MASK[ints].astype(float)


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
    print(f"{name:<26} est={m:.4e} +/-{ci:.1e} "
          f"(exact {PF:.4e}, bias {100*(m-PF)/PF:+.2f}%)  "
          f"VRF={vrf:>10,.1f}x  ESS={np.mean(esss):>9,.0f}/{BUDGET:,}")


if __name__ == "__main__":
    print(f"Exact P(disconnect) = {PF:.6e}\n")
    print("training quantum proposal (KL) ...")
    q = train_kl()
    np.save("q_network_kl.npy", q)
    print("fitting CE tilt vector ...")
    theta = ce_fit()
    print(f"  theta (per-edge, corners should stand out):\n  "
          f"{np.round(theta, 3)}\n")
    summarize("naive MC", naive_trial)
    summarize("quantum IS (KL)", lambda r: q_trial(r, q))
    summarize("classical CE tilts", lambda r: ce_trial(r, theta))