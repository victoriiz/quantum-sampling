"""
Replaces cost = (E_q[mask] - 0.10)^2) with 
forward KL(p* || q) = -sum_b p*(b) log q(b) + const

NOTE: at n=10, statevector gives exact probs, so KL gradient is exact. at scale, replace with sampled REINFORCE-style gradient estimate.
"""
import os
import json
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import spike_model as sm

N_QUBITS = 10
N_LAYERS = 5
STEPS = 300
LR = 0.05
SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

dev = qml.device("default.qubit", wires=N_QUBITS)

MASK = pnp.array(sm.failure_mask(N_QUBITS), requires_grad=False)
P_STAR = pnp.array(sm.p_star(N_QUBITS), requires_grad=False)

def ansatz(params):
    params = params.reshape((N_LAYERS, N_QUBITS))
    for L in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(params[L, i], wires=i)
        for i in range(N_QUBITS-1):
            qml.CNOT(wires=[i, i+1])

@qml.qnode(dev)
def probs(params):
    ansatz(params)
    return qml.probs(wires=range(N_QUBITS))

def kl_cost(params):
    q = probs(params)
    return -pnp.sum(P_STAR* pnp.log(q+1e-12))

def old_cost(params, target=0.10):
    """MSE"""
    q = probs(params)
    return (pnp.sum(q*MASK) - target)**2

def train(cost_fn, tag):
    np.random.seed(SEED)
    params = pnp.array(
        np.random.uniform(0, 2*np.pi, N_LAYERS * N_QUBITS),
        requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=LR)
    hist = []
    for step in range(STEPS):
        params, loss = opt.step_and_cost(cost_fn, params)
        q = probs(params)
        mass = float(pnp.sum(q * MASK))
        hist.append({"step:": step, "loss": float(loss), "fail_mass": mass})
        if step % 50 == 0 or step == STEPS - 1:
            print(f"[{tag}] step {step:3d}  loss={float(loss):.6f}  "
                  f"mass on failure set={mass:.4f}")
    
    np.save(os.path.join(OUT, f"weights_{tag}.npy"), np.array(params))
    with open(os.path.join(OUT, f"history_{tag}.json"), "w") as f:
        json.dump(hist, f)
    return np.array(params)

if __name__ == "__main__":
    print(f"Exact P(fail) reference (validation only): "
          f"{sm.exact_p_fail(N_QUBITS):.6e}")
    kl_entropy_floor = float(-np.sum(sm.p_star(N_QUBITS)
                                     * np.log(sm.p_star(N_QUBITS) + 1e-300)))
    print(f"Entropy of p* (KL loss floor if q == p*): "
          f"{kl_entropy_floor:.6f}\n")
    train(kl_cost, "kl")
    print()
    train(old_cost, "old_mse")