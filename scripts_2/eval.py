"""
* IS estimator: w(b) = p(b) * 1_fail(b) / q(b)
* classical exponential-tilting baseline by Bucklew, 2004
* 30 independent trials of everything
sampling from q done classical from exact statevector -- at n=10 this is simulation of quantum sampler, not quantum hardware
"""
import os
import json
import numpy as np
import pennylane as qml
import spike_model as sm

N_QUBITS = 10
N_LAYERS = 5
BUDGET = 200000
TRIALS = 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

P_VEC = sm.p_vector(N_QUBITS)
MASK = sm.failure_mask(N_QUBITS)
HW = sm.hamming_weights(N_QUBITS)
PF_EXACT = sm.exact_p_fail(N_QUBITS)
P_TARGET = P_VEC * MASK 

def circuit_probs(weights_path):
    dev = qml.device("default.qubit", wires=N_QUBITS)
    w = np.load(weights_path)

    @qml.qnode(dev)
    def _p(params):
        params = params.reshape((N_LAYERS, N_QUBITS))
        for L in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(params[L, i], wires=i)
            for i in range(N_QUBITS-1):
                qml.CNOT(wires=[i, i+1])
        return qml.probs(wires=range(N_QUBITS))
    return np.array(_p(w))

def is_trial(q, rng, budget=BUDGET):
    idx = rng.choice(2 ** N_QUBITS, size=budget, p=q)
    w = np.where(q[idx] > 0, P_TARGET[idx] / q[idx], 0.0)
    return w.mean(), w.var(ddof=1)

def tilt_trial(rng, p_tilt, budget=BUDGET):
    b = (rng.random((budget, N_QUBITS)) < p_tilt)
    k = b.sum(axis=1)
    p = sm.P_SPIKE
    lr = (p/p_tilt) ** k * ((1-p) / (1-p_tilt)) ** (N_QUBITS-k)
    w = lr * (k >= sm.K_FAIL)
    return w.mean(), w.var(ddof=1)

def summarize(name, trials_fn):
    ests, vars_ = [], []
    for t in range(TRIALS):
        rng = np.random.default_rng(1000 + t)
        e, v = trials_fn(rng)
        ests.append(e)
        vars_.append(v)
    ests = np.array(ests)
    mean = ests.mean()
    ci = 1.96 * ests.std(ddof=1) / np.sqrt(TRIALS)
    per_sample_var = float(np.mean(vars_))
    vrf = (PF_EXACT * (1 - PF_EXACT)) / per_sample_var \
        if per_sample_var > 0 else float("inf")
    rel_err = (mean - PF_EXACT) / PF_EXACT
    bias = mean - PF_EXACT
    estimator_variance = per_sample_var / BUDGET
    mse = bias ** 2 + estimator_variance
    classical_mse = (PF_EXACT * (1-PF_EXACT)) / BUDGET
    mse_ratio = mse / classical_mse if classical_mse > 0 else float('inf')


    print(f"{name:<28} est = {mean:.4e} +/- {ci:.1e}  "
          f"(exact {PF_EXACT:.4e}, rel.err {rel_err:+.2%})  "
          f"VRF = {vrf:,.1f}x")
    
    print(f"{'':<28} bias = {bias:+.3e}  bias^2 = {bias**2:.3e}  "
          f"est.var = {estimator_variance:.3e}  MSE = {mse:.3e}  "
          f"MSE vs naive MC = {mse_ratio:.4f}x "
          f"({'better' if mse_ratio < 1 else 'worse'})")
    
    return {"name": name, "mean": float(mean), "ci95": float(ci),
            "per_sample_var": per_sample_var, "vrf": float(vrf),
            "bias": float(bias), "bias_sq": float(bias ** 2),
            "estimator_variance": float(estimator_variance),
            "mse": float(mse), "mse_ratio_vs_naive": float(mse_ratio)}


if __name__ == "__main__":
    print(f"Exact P(fail) = {PF_EXACT:.6e}   "
          f"budget/trial = {BUDGET:,}   trials = {TRIALS}\n")

    q_kl = circuit_probs(os.path.join(OUT, "weights_kl.npy"))
    q_old = circuit_probs(os.path.join(OUT, "weights_old_mse.npy"))
    #np.save(os.path.join(OUT, "q_kl.npy"), q_kl)
    #np.save(os.path.join(OUT, "q_old.npy"), q_old)

    def naive_trial(rng, budget=BUDGET):
        # expected failures per trial = budget * 1e-5 ~= 2 -> hopeless
        k = (rng.random((budget, N_QUBITS)) < sm.P_SPIKE).sum(axis=1)
        hits = (k >= sm.K_FAIL).astype(float)
        return hits.mean(), hits.var(ddof=1)

    results = []
    results.append(summarize("naive MC (same budget)", naive_trial))
    results.append(summarize("quantum IS (KL objective)",
                             lambda rng: is_trial(q_kl, rng)))
    results.append(summarize("quantum IS (old MSE obj.)",
                             lambda rng: is_trial(q_old, rng)))
    p_opt = sm.K_FAIL / N_QUBITS
    results.append(summarize(f"classical tilting (p'={p_opt})",
                             lambda rng: tilt_trial(rng, p_opt)))

    with open(os.path.join(OUT, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

