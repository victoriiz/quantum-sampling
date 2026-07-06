import pennylane as qml
from pennylane import numpy as np

num_qubits = 10
dev = qml.device("default.qubit", wires=num_qubits)

def hamming_failure_mask(num_qubits, k_crit=3.5) -> np.ndarray:
    """
    Generates a diagonal failure projector mask based on the abstract Hamming weight boundary, k_crit.
    States with Hamming weight >= ceil(k_crit) are considered failure states.

    Args:
        num_qubits: Number of qubits in the register.
        k_crit: Critical geometric boundary midpoint. Default is 3.5.
    
    Returns:
        A 1D numpy array of size 2^num_qubits acting as a diagonal filter.
    """
    num_states = 2**num_qubits
    failure_mask = np.zeros(num_states, dtype=np.float64)
    threshold = int(np.ceil(k_crit))

    for idx in range(num_states):
        hamming_weight = bin(idx).count("1")
        if hamming_weight >= threshold:
            failure_mask[idx] = 1.0
    return failure_mask

def phys_failure_mask(num_qubits, tau=65.0, normal_load=1.0, extreme_load=4.2) -> np.ndarray:
    """
    Generates a diagonal failure mask by explicitly evaluating the physical stress equations for every individual basis state config.
    
    Args:
        num_qubits: number of variables/qubits.
        tau: structural capacity ceiling.
        normal_load: physical value mapped to bit 0.
        extreme_load: physical value mapped to bit 1.
    
    Returns:
        A 1D numpy array of size 2^num_qubits representing true micro-physics boundaries.
    """
    num_states = 2**num_qubits
    failure_mask = np.zeros(num_states, dtype = np.float64)

    for idx in range(num_states):
        binary_string = f"{idx:0{num_qubits}b}"
        phys_loads = [extreme_load if bit == '1' else normal_load for bit in binary_string]
        stress = np.sum(np.array(phys_loads, dtype=np.float64) ** 2)

        if stress > tau:
            failure_mask[idx] = 1.0

    return failure_mask

def variational_circuit(params, num_qubits, num_layers):
    """
    Constructs a Parameterized Quantum Circuit (PQC) for a 10-qubit system.
    Args:
        params: Flattened or structured array of trainable gate rotation angles
        num_qubits: Number of wires in device.
        num_layers: Total depth of the variational ansatz.
    """
    params = params.reshape((num_layers, num_qubits))

    for layer in range(num_layers):
        for qubit in range(num_qubits):
            qml.RY(params[layer, qubit], wires=qubit)
        
        for qubit in range(num_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
    
    #return qml.probs(wires=range(num_qubits))

if __name__ == "__main__":
    import os

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

    num_qubits = 10
    dev = qml.device("default.qubit", wires=num_qubits)
    np.random.seed(42)
    layers = 3
    initial_weights = np.random.uniform(0, 2 * np.pi, (layers, num_qubits), requires_grad=True)

    @qml.qnode(dev)
    def _probe(weights):
        variational_circuit(weights, num_qubits, layers)
        return qml.probs(wires=range(num_qubits))
    
    probs = _probe(initial_weights)

    print(f"Successfully initiated ansatz with {layers} layers and {num_qubits} qubits. There are {initial_weights.size} trainable parameters")
    print(f"Output state space dimension: {len(probs)} distinct quantum amplitudes.")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "initial_quantum_probs.npy")
    np.save(out_path, np.array(probs))
    print(f"Intiail quantum probabilities saved to '{out_path}'")
    # os.makedirs("quantum_outputs", exist_ok=True)
    # np.save("quantum_outputs/initial_quantum_probs.npy", np.array(probs))
    # print("Initial quantum probabilities saved to 'quantum_outputs/initial_quantum_probs.npy'")
