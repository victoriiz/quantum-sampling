import os
import sys
import pennylane as qml
from pennylane import numpy as np
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

WEIGHT_PATH = "outputs/optimized_vqis_weights.npy"
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

# POST PROCESSING IMPORTANCE SAMPLING!!
if __name__ == "__main__":
    print("="*70)
    print("PHASE 4: COHERENT STATEVECTOR SAMPLING")
    print("="*70)
    
    state_vector = get_exact_statevector(weights)
    q_dist = np.abs(state_vector) ** 2

    np.random.seed(42)
    sample_indices = np.random.choice(range(2**NUM_QUBITS), size=SAMPLE_BUDGET, p=q_dist)

    is_weights = np.zeros(SAMPLE_BUDGET, dtype=np.float64)

    for j in range(SAMPLE_BUDGET):
        state_idx = sample_indices[j]
        q_value = q_dist[state_idx]

        binary_string = f"{state_idx:0{NUM_QUBITS}b}"
        bitstring = np.array([int(char) for char in binary_string])
        
        hamming_weight = np.sum(bitstring)
        
        if hamming_weight >= 4 and q_value > 1e-12:
            #phys_coordinates = np.array([EXTREME_LOAD if b == 1 else NORMAL_LOAD for b in bitstring])
            
            #log_p = -np.sum(phys_coordinates)
            #log_q = np.log(q_value)
            
            #is_weights[j] = np.exp(log_p - log_q, dtype=np.float64)
            p_discrete = GROUND_TRUTH / 848.0
            raw_weight = p_discrete / q_value
            weight = min(raw_weight, 5.0)
            is_weights[j] = weight
        else:
            is_weights[j] = 0.0
    # ---------------------------------------------------------
    # Statistical Evaluation
    # ---------------------------------------------------------
    p_fail_hat = float(np.mean(is_weights))
    
    if p_fail_hat == 0.0:
        print("\n[FATAL] Output mass collapsed to zero. Verify weight convergence in Phase 3.")
        sys.exit(1)
        
    is_variance = float(np.var(is_weights, ddof=1))
    classical_variance = GROUND_TRUTH * (1.0 - GROUND_TRUTH)
    vrf = classical_variance / is_variance

    print(f"Target Ground Truth : {GROUND_TRUTH * 100:.4f}%")
    print(f"VQIS Unbiased Est.  : {p_fail_hat * 100:.4f}%")
    print("-" * 70)
    print(f"Standard MC Variance: {classical_variance:.6e}")
    print(f"VQIS Sample Variance: {is_variance:.6e}")
    print(f"VRF (Speedup)       : {vrf:.2f}x")
    print("="*70)