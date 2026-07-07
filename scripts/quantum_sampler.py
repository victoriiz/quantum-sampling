import os
import sys
import pennylane as qml
from pennylane import numpy as np
from math import comb
import quantum_ansatz as ansatz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

NUM_QUBITS = 10
NUM_LAYERS = 5
TAU = 65.0
NORMAL_LOAD = 1.0
EXTREME_LOAD = 4.2
SAMPLE_BUDGET = 1000000
GROUND_TRUTH = 0.01422
CLIP_C = 5.0

USE_WEIGHT_DEPENDENT = True # otherwise use False for old flat-p_discrete

WEIGHT_PATH = os.path.join(OUTPUTS_DIR, "optimized_vqis_weights.npy")
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError("optimized weights not found.")
weights = np.load(WEIGHT_PATH)

dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, shots=SAMPLE_BUDGET)
def generate_quantum_samples(params):
    """
    Executes the quantum circuit and collects empirical shot samples (bitstrings)
    """
    ansatz.variational_circuit(params, NUM_QUBITS, NUM_LAYERS)
    return qml.sample(wires=range(NUM_QUBITS))

# analytical exact probability for exact q(b) extraction
dev_exact = qml.device("lightning.qubit", wires=NUM_QUBITS)

@qml.qnode(dev_exact)
def get_exact_probs(params):
    ansatz.variational_circuit(params, NUM_QUBITS, NUM_LAYERS)
    return qml.probs(wires=range(NUM_QUBITS))

# -------------------------
@qml.qnode(dev)
def get_exact_statevector(params):
    ansatz.variational_circuit(params, NUM_QUBITS, NUM_LAYERS)
    return qml.state()

# --------------------------
def build_p_lookup():
    """
    Returns an array of length (num_qubits+1) for the probability mass assigned
    to a signle basis state of hamming weight k
    """
    lookup = np.zeros(NUM_QUBITS + 1)
    if not USE_WEIGHT_DEPENDENT:
        for k in range(4, NUM_QUBITS + 1):
            lookup[k] = GROUND_TRUTH / 848.0
        return lookup
    
    cond_path = os.path.join(OUTPUTS_DIR, "weight_conditional_probs.npy")
    if not os.path.exists(cond_path):
        raise FileNotFoundError(
            f"{cond_path} not found"
        )
    p_k_given_failure = np.load(cond_path) # length 11, empirical P(k spikes | failure)
    for k in range(NUM_QUBITS + 1):
        c = comb(NUM_QUBITS, k)
        lookup[k] = (GROUND_TRUTH * p_k_given_failure[k] / c) if c > 0 else 0.0
    
    return lookup

# POST PROCESSING IMPORTANCE SAMPLING!!
if __name__ == "__main__":
    print("="*70)
    print("PHASE 4: COHERENT STATEVECTOR SAMPLING")
    print(f"p_discrete mode: {'weight-dependent' if USE_WEIGHT_DEPENDENT else 'flat/uniform'}")
    print("="*70)

    p_discrete_by_weight = build_p_lookup()
    
    state_vector = get_exact_statevector(weights)
    q_dist = np.abs(state_vector) ** 2

    np.random.seed(42)
    sample_indices = np.random.choice(range(2**NUM_QUBITS), size=SAMPLE_BUDGET, p=q_dist)

    is_weights = np.zeros(SAMPLE_BUDGET, dtype=np.float64)
    hamming_weights_seen = np.zeros(SAMPLE_BUDGET, dtype=np.int64)

    for j in range(SAMPLE_BUDGET):
        state_idx = sample_indices[j]
        q_value = q_dist[state_idx]

        #binary_string = f"{state_idx:0{NUM_QUBITS}b}"
        #bitstring = np.array([int(char) for char in binary_string])
        
        #hamming_weight = np.sum(bitstring)
        hamming_weight = bin(state_idx).count("1")
        hamming_weights_seen[j] = hamming_weight
        
        p_discrete = p_discrete_by_weight[hamming_weight]
        if p_discrete > 0 and q_value > 1e-12:
            raw_weight = p_discrete / q_value
            is_weights[j] = min(raw_weight, CLIP_C)
        else:
            is_weights[j] = 0.0

    p_fail_hat = float(np.mean(is_weights))
    
    if p_fail_hat == 0.0:
        print("\n[FATAL] Output mass collapsed to zero. Verify weight convergence in Phase 3.")
        sys.exit(1)
        
    is_variance = float(np.var(is_weights, ddof=1))
    classical_variance = GROUND_TRUTH * (1.0 - GROUND_TRUTH)
    vrf = classical_variance / is_variance if is_variance > 0 else float('nan')

    bias = p_fail_hat - GROUND_TRUTH
    estimator_variance = is_variance / SAMPLE_BUDGET
    mse = bias**2 + estimator_variance
    classical_est_variance = classical_variance / SAMPLE_BUDGET
    classical_mse = classical_est_variance #unbiased

    print(f"Target Ground Truth : {GROUND_TRUTH * 100:.4f}%")
    print(f"VQIS Unbiased Est.  : {p_fail_hat * 100:.4f}%")
    print(f"Bias                : {bias * 100:+.4f} percentage points ({bias/GROUND_TRUTH*100:+.1f}% relative)")
    print("-" * 70)
    print(f"Per-sample weight variance : {is_variance:.6e}   VRF: {vrf:.2f}x")
    print(f"Estimator variance (/{SAMPLE_BUDGET:,}) : {estimator_variance:.6e}")
    print(f"MSE (bias^2 + est. variance): {mse:.6e}")
    print(f"Classical MC MSE (unbiased) : {classical_mse:.6e}")
    if mse > 0:
        print(f"MSE ratio vs. classical MC  : {mse / classical_mse:.1f}x {'worse' if mse > classical_mse else 'better'}")
    
    print(f"VQIS Sample Variance: {is_variance:.6e}")
    print(f"VRF (Speedup)       : {vrf:.2f}x")
    print("="*70)

    if hamming_weights_seen.max() >= 4:
        frac_in_old_subspace = float(np.mean(hamming_weights_seen >= 4))
        print(f"\nFraction of drawn samples with Hamming weight >= 4 (the ansatz's trained target): {frac_in_old_subspace*100:.2f}%")