cat << 'EOF' > quantum_optimization.py
import pennylane as qml
from pennylane import numpy as np
import os
import matplotlib.pyplot as plt

if os.path.exists("outputs/ground_truth_baseline.npy"):
    target = float(np.load("outputs/ground_truth_baseline.npy"))
else:
    target = 0.01420

num_qubits = 10
dev = qml.device("default.qubit", wires=num_qubits)

num_states = 2**num_qubits
failure_mask = np.zeros(num_states)
for state_idx in range(num_states):
    binary_string = f"{state_idx:0{num_qubits}b}"
    # TODO: figure out actually plausible failure representation
    if binary_string.count("1") > 5:
        failure_mask[state_idx] = 1.0

@qml.qnode(dev)
def circuit(params):
    """
    Constructs a Variational Quantum Circuit for a 10-qubit system
    params: matrix of angles (shape: layers x num_qubits)
    returns: probabilities of all 2^10 possible state configs
    """
    num_layers = params.shape[0]

    for layer in range(num_layers):
        for i in range(num_qubits):
            qml.RY(params[layer, i], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires = [i, i+1])
    
    return qml.probs(wires=range(num_qubits))

def cost(params):
    """
    Cost function for the optimization problem.
    params: matrix of angles (shape: layers x num_qubits)
    returns: cost value based on the probabilities and failure mask
    """
    probs = circuit(params)
    qfailure_weight = np.dot(probs, failure_mask)
    loss = (qfailure_weight - target) ** 2
    return loss

if __name__ == "__main__":
    np.random.seed(42)
    num_layers = 3
    weights = np.random.uniform(0, 2*np.pi, (num_layers, num_qubits), requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.1)
    max_iters = 50

    print(f"Starting optimization with {num_layers} layers and {num_qubits} qubits. There are {weights_0.size} trainable parameters.")
    print(f"Target failure probability: {target}")
    print("-"*50)

    loss_history = []
    weight_history = []

    for it in range(max_iters):
        weights, loss = opt.step_and_cost(cost, weights)
        loss_history.append(loss)
        weight_history.append(weights.copy())

        if (it + 1) % 5 == 0 or it == 0:
            current_probs = circuit(weights)
            current_weight = float(np.dot(current_probs, failure_mask))
            print(f"Iteration {it+1:02d} | Loss: {loss:.6f} | Current Failure Weight: {current_weight:.6f}")
    
    np.save("outputs/optimized_weights.npy", np.array(weights))
    np.save("outputs/loss_history.npy", np.array(loss_history))
    np.save("outputs/weight_history.npy", np.array(weight_history))
    print("Optimization complete. Optimized weights and loss history saved to 'outputs/' directory.")

    plt.figure(figsize=(7,4))
    plt.plot(range(1, max_iters + 1), loss_history, marker='o', color='blue', label='Adam Optimizer Loss')
    plt.title("Variational Quantum Parameter Convergence")
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Mean Squared Error Loss")
    plt.grid(True, alpha=0.2)
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/optimization_loss_plot.png", dpi=300, bbox_inches='tight')
    print("Loss convergence plot saved to 'figures/optimization_loss_plot.png'")

EOF

