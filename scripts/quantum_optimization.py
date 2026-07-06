import pennylane as qml
from pennylane import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scripts.quantum_ansatz as ansatz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

#TARGET_PROB = 0.01420
#if os.path.exists("outputs/ground_truth_baseline.npy"):
#    TARGET_PROB = float(np.load("outputs/ground_truth_baseline.npy"))
TARGET_PROB = 0.10 # NEW: force aggressively oversample

NUM_QUBITS = 10
NUM_LAYERS = 5
LEARNING_RATE = 0.05
MAX_STEPS = 150
SEED = 42

dev = qml.device("lightning.qubit", wires=NUM_QUBITS)
diagonal_elements = ansatz.hamming_failure_mask(NUM_QUBITS, k_crit=3.5)
failure_operator = qml.Hermitian(np.diag(diagonal_elements), wires=range(NUM_QUBITS))

@qml.qnode(dev, diff_method="adjoint")
def execute_qnode(params):
    """
    Binds the ansatz layour to the lightning simulator device with Adjoint tracking enabled.
    """
    ansatz.variational_circuit(params, NUM_QUBITS, NUM_LAYERS)
    return qml.expval(failure_operator)

def cost(params):
    """
    Cost function for the optimization problem.
    params: matrix of angles (shape: layers x num_qubits)
    returns: cost value based on the probabilities and failure mask
    """
    current_pfail = execute_qnode(params)
    loss = (current_pfail - TARGET_PROB) ** 2
    return loss

if __name__ == "__main__":
    print("="*70)
    print("STARTING VQIS PHASE 3: VARIATIONAL CLASSICAL INVERSION LOOP")
    print("="*70)
    print(f"Target Failure Probability Anchor: {TARGET_PROB * 100:.3f}%")
    print(f"Total System Dimensions/Qubits : {NUM_QUBITS}")
    print(f"Total Optimization Parameters  : {NUM_LAYERS * NUM_QUBITS} (3 Layers)")
    
    np.random.seed(SEED)
    initial_params = np.random.uniform(0, 2 * np.pi, NUM_LAYERS * NUM_QUBITS, requires_grad=True)
    
    initial_p_fail = execute_qnode(initial_params)
    print(f"Initial Arbitrary Interference Failure Mass: {initial_p_fail * 100:.2f}% (Expected ~42.1%)")
    print("-" * 70)
    
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    
    params = initial_params
    loss_history = []
    prob_history = []
    
    print(f"{'Step':<6} | {'MSE Loss':<12} | {'Current P_fail (%)':<20} | {'Delta to Target':<12}")
    print("-" * 70)
    
    for step in range(MAX_STEPS):
        params, loss = opt.step_and_cost(cost, params)
        
        current_p_fail = execute_qnode(params)
        delta = current_p_fail - TARGET_PROB
        
        loss_history.append(loss)
        prob_history.append(current_p_fail)
        
        if step % 10 == 0 or step == MAX_STEPS - 1:
            print(f"{step:<6} | {loss:<12.7f} | {current_p_fail * 100:<19.3f}% | {delta:<+12.5f}")
            
        if loss < 1e-8:
            print(f"\n[INFO] Convergence achieved within tolerance at step {step}!")
            break
    
    #np.save("outputs/optimized_vqis_weights.npy", params)
    #np.save("outputs/loss_history.npy", np.array(loss_history))
    np.save(os.path.join(OUTPUTS_DIR, "optimized_vqis_weights.npy"), params)
    np.save(os.path.join(OUTPUTS_DIR, "loss_history.npy"), np.array(loss_history))
    print("Optimization complete. Optimized weights and loss history saved to 'outputs/' directory.")

    fig, ax1 = plt.subplots(figsize=(10,5))

    color = 'tab:blue'
    ax1.set_xlabel('Optimization Iterations')
    ax1.set_ylabel('Loss (MSE)', color=color)
    ax1.plot(loss_history, color=color, linewidth=2, label="MSE Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Quantum Failure Mass [P_fail]', color=color)
    ax2.plot(prob_history, color=color, linestyle='--', linewidth=2, label="Current P_fail")
    ax2.axhline(y=TARGET_PROB, color='black', linestyle=':', linewidth=1.5, label='Target Anchor')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("VQIS Classical Inversion Loop Convergence Profile")
    fig.tight_layout()
    #plt.savefig("figures/vqis_convergence_profile.png", dpi=300)
    convergence_path = os.path.join(FIGURES_DIR, "vqis_convergence_profile.png")
    plt.savefig(convergence_path, dpi=300)
    print("[SUCCESS] Convergence tracking schematic saved to 'figures/vqis_convergence_profile.png'")


