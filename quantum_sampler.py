import os
import sys
import pennylane as qml
from pennylane import numpy as np
import quantum_ansatz as ansatz

NUM_QUBITS = 10
NUM_LAYERS = 3
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


# POST PROCESSING IMPORTANCE SAMPLING!!
if __name__ == "__main__":
    print("="*70)
    print("STARTING SAMPLE EVALUATION & VARIANCE VALIDATION")
    print("="*70)
    print(f"Sample Allocation Budget (M): {SAMPLE_BUDGET:,} shots")

    # 1. extract exact analytical q(b) dist from optimized circuit
    print("extracting analytical proposal distribution matrix...")
    q_dist = get_exact_probs(weights)

    # 2. generate M empiraical sample bitstirngs from frozen hardware sim
    print("querying quantum circuit sampler for empirical shots...")
    raw_samples = generate_quantum_samples(weights) # shape: (SAMPLE_BUDGET, NUM_QUBITS)

    # Temporary Diagnostic Block
    print("Running diagnostic on first 50 generated bitstrings...")
    for idx in range(50):
        hw = list(raw_samples[idx]).count(1)
        if hw >= 6:
            print(f"Captured high-weight failure state at shot {idx}: {raw_samples[idx]}")

    # 3. process samples through Likelihood Ratio Evaluator
    is_weights = []
    failure_indicators = []

    print("executing classical Radon-Nikodym weight corrections...")
    for j in range(SAMPLE_BUDGET):
        bitstring = raw_samples[j]
        state_idx = int("".join(str(b) for b in bitstring), 2)
        q_val = q_dist[state_idx]

        phys_coords = np.array([EXTREME_LOAD if bit==1 else NORMAL_LOAD for bit in bitstring])

        # eval true physics model stress: S = sum(x_i^2)
        #stress = np.sum(phys_coords**2)
        #is_failure = 1.0 if stress > TAU else 0.0
        hamming_weight = list(bitstring).count(1)
        is_failure = 1.0 if hamming_weight >= 6 else 0.0
        failure_indicators.append(is_failure)

        # calc true cont. joint probability density: p(x) = exp(-sum(x_i))
        p_val = np.exp(-np.sum(phys_coords))
        
        weight = p_val/q_val if q_val > 1e-12 else 0.0 # likelihood ratio: w(x) = p(x)/q(x)

        is_weights.append(is_failure * weight)
    
    is_weights = np.array(is_weights)
    p_fail_hat = np.mean(is_weights)

    if p_fail_hat == 0.0:
        print("\n[FATAL ERROR] Zero failure states were observed during empirical sampling.")
        print("The VRF calculation is mathematically invalid. Increase SAMPLE_BUDGET or re-evaluate weights.")
        sys.exit()

    # sample variance of VQIS estimator
    is_variance = np.var(is_weights, ddof=1)
    is_cov = np.sqrt(is_variance) / (np.sqrt(SAMPLE_BUDGET) * p_fail_hat)

    # classically, bernoulli trial estimator has variance Var = p * (1-p)
    classical_variance = GROUND_TRUTH * (1.0 - GROUND_TRUTH)
    classical_cov = np.sqrt(classical_variance) / (np.sqrt(SAMPLE_BUDGET) * GROUND_TRUTH)

    vrf = classical_variance / is_variance

    # ------------------------
    # bunch of printing for performance
    # -------------------------
    print("\n" + "="*70)
    print("PHASE 4 PERFORMANCE EVALUATION REPORT")
    print("="*70)
    print(f"True Classical Ground Truth   : {GROUND_TRUTH * 100:.4f}%")
    print(f"VQIS Unbiased Sample Estimate : {p_fail_hat * 100:.4f}%")
    print(f"Absolute Estimate Error       : {abs(p_fail_hat - GROUND_TRUTH):.6f}")
    print("-" * 70)
    print(f"Standard Classical MC Variance: {classical_variance:.6f}  (CoV: {classical_cov:.4f})")
    print(f"Optimized VQIS Sample Variance: {is_variance:.6f}  (CoV: {is_cov:.4f})")
    print("-" * 70)
    print(f"DEFINITIVE VARIANCE REDUCTION FACTOR (VRF): {vrf:.2f}x")
    if vrf > 1.0:
        print(f"[SUCCESS] Quantum circuit achieved a {vrf:.2f}x efficiency acceleration!")
        print(f"This run matched the confidence of {int(SAMPLE_BUDGET * vrf):,} standard classical samples.")
    else:
        print("[WARNING] No variance reduction detected. Circuit optimization requires deeper profiling.")
    print("="*70)