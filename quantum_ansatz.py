import pennylane as qml
from pennylane import numpy as np

# define 10-qubit quantum device accelerated by GPU
num_qubits = 10
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def variational_circuit(params):
    """
    Constructs a Parameterized Quantum Circuit (PQC) for a 10-qubit system.
    params: matrix of angles (shape: layers x num_qubits)
    returns: probabilities of all 2^10 possible state configs
    """
    num_layers = params.shape[0]

    for layer in range(num_layers):
        for qubit in range(num_qubits):
            qml.RY(params[layer, qubit], wires=qubit)
        
        for qubit in range(num_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
    
    return qml.probs(wires=range(num_qubits))

if __name__ == "__main__":
    np.random.seed(42)
    layers = 3
    initial_weights = np.random.uniform(0, 2 * pi, (layers, num_qubits))

    probs = variational_circuit(initial_weights)
    print(f"Successfully initiated ansatz with {layers} layers and {num_qubits} qubits. There are {initial_weights.size} trainable parameters")
    print(f"Output state space dimension: {len(probs)} distinct quantum amplitudes.")