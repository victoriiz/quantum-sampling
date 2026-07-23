"""
Isolated unit tests for every component of the Grover operator.
"""
import numpy as np
import pennylane as qml

# ------------------------------------------------------------------ instance
N      = 4
ANC    = N
TOT    = N + 1
P_TRUE = np.exp(-1.6)
P_TILT = 0.55
K_FAIL = 3

def bits_of(b):
    return [(b >> j) & 1 for j in reversed(range(N))]

def lr(bits):
    k = sum(bits)
    return ((P_TRUE / P_TILT) ** k) * (((1 - P_TRUE) / (1 - P_TILT)) ** (N - k))

raw = {}
for b in range(2 ** N):
    bt = bits_of(b)
    raw[b] = lr(bt) if sum(bt) >= K_FAIL else 0.0
ZMAX = max(raw.values())
Z = {b: raw[b] / ZMAX for b in raw}

q_prob = {b: np.prod([P_TILT if x else 1 - P_TILT for x in bits_of(b)])
          for b in range(2 ** N)}
MU = sum(q_prob[b] * Z[b] for b in q_prob)
THETA = np.arcsin(np.sqrt(MU))

print(f"instance: N={N}, p_true={P_TRUE:.4f}, p_tilt={P_TILT}, k>={K_FAIL}")
print(f"mu = {MU:.6f}, theta = {THETA:.6f} rad\n")

# ============================================================= components

def A_op():
    """State preparation + payoff encoding (forward)."""
    for i in range(N):
        qml.RY(2 * np.arcsin(np.sqrt(P_TILT)), wires=i)
    for b in range(2 ** N):
        z = Z[b]
        if z > 0:
            qml.ctrl(qml.RY, control=list(range(N)), control_values=bits_of(b))(
                2 * np.arcsin(np.sqrt(z)), wires=ANC)

def A_op_adjoint_manual():
    """Explicit inverse of A_op."""
    for b in reversed(range(2 ** N)):
        z = Z[b]
        if z > 0:
            qml.ctrl(qml.RY, control=list(range(N)), control_values=bits_of(b))(
                -2 * np.arcsin(np.sqrt(z)), wires=ANC)
    for i in reversed(range(N)):
        qml.RY(-2 * np.arcsin(np.sqrt(P_TILT)), wires=i)

def S_chi():
    """Phase-flip the good subspace (ancilla = 1)."""
    qml.PauliZ(wires=ANC)

def S_zero():
    """Reflection about |0...0>:  I - 2|0..0><0..0|."""
    for w in range(TOT):
        qml.PauliX(wires=w)
    qml.ctrl(qml.PauliZ, control=list(range(TOT - 1)),
             control_values=[1] * (TOT - 1))(wires=TOT - 1)
    for w in range(TOT):
        qml.PauliX(wires=w)

def G_op(use_manual_adjoint=True):
    """Grover iterate G = A S_0 A^dagger S_chi."""
    S_chi()
    if use_manual_adjoint:
        A_op_adjoint_manual()
    else:
        qml.adjoint(A_op)()
    S_zero()
    A_op()

# ================================================================== QNodes
dev = qml.device("default.qubit", wires=TOT)

@qml.qnode(dev)
def qnode_A_dag_A_identity(use_manual=True):
    A_op()
    if use_manual:
        A_op_adjoint_manual()
    else:
        qml.adjoint(A_op)()
    return qml.probs(wires=list(range(TOT)))

@qml.qnode(dev)
def qnode_S_zero_on_zero():
    S_zero()
    return qml.state()

@qml.qnode(dev)
def qnode_S_chi_phase_flip():
    qml.Hadamard(wires=ANC)
    S_chi()
    return qml.state()

@qml.qnode(dev)
def qnode_get_psi_vec():
    """Return |psi> = A|0...0> as a plain numpy vector."""
    A_op()
    return qml.state()

# ================================================================== TESTS

# ------------------------------------------------------------------ TEST 1
print("=" * 60)
print("TEST 1: A^dagger A = I  (fidelity with |0...0>)")
print("=" * 60)
probs_manual = np.array(qnode_A_dag_A_identity(use_manual=True))
fid_manual = float(probs_manual[0])
probs_auto = np.array(qnode_A_dag_A_identity(use_manual=False))
fid_auto = float(probs_auto[0])
print(f"  manual A^dagger : fidelity = {fid_manual:.6f}  (want 1.000000)")
print(f"  qml.adjoint(A)  : fidelity = {fid_auto:.6f}  (want 1.000000)")
assert fid_manual > 1 - 1e-10, "FAIL: manual A^dagger is not unitary inverse"
if fid_auto < 1 - 1e-6:
    print("  >>> WARNING: qml.adjoint(A_op) is BROKEN in this PennyLane version")
else:
    print("  >>> qml.adjoint(A_op) passes")

# ------------------------------------------------------------------ TEST 2
print("\n" + "=" * 60)
print("TEST 2: S_0 |0...0> = -|0...0>")
print("=" * 60)
psi = np.array(qnode_S_zero_on_zero(), dtype=complex)
amp_0 = psi[0]
print(f"  amplitude of |0...0> = {amp_0:.6f}  (want -1.000000)")
assert abs(amp_0 + 1.0) < 1e-10, "FAIL: S_zero does not reflect |0...0> correctly"
print("  PASS")

# ------------------------------------------------------------------ TEST 3
print("\n" + "=" * 60)
print("TEST 3: S_chi phase-flips ancilla = 1")
print("=" * 60)
psi = np.array(qnode_S_chi_phase_flip(), dtype=complex)
amp_0 = psi[0]
amp_1 = psi[1]
print(f"  amplitude |anc=0> = {amp_0:.6f}  (want +0.707107)")
print(f"  amplitude |anc=1> = {amp_1:.6f}  (want -0.707107)")
assert abs(abs(amp_0) - 1/np.sqrt(2)) < 1e-10, "FAIL: S_chi affected |anc=0>"
assert abs(amp_1 + 1/np.sqrt(2)) < 1e-10, "FAIL: S_chi did not flip |anc=1>"
print("  PASS")

# ------------------------------------------------------------------ TEST 4 (CORRECTED)
print("\n" + "=" * 60)
print("TEST 4: G preserves the { |psi>, |psi_good> } plane")
print("=" * 60)

# Build the three states directly from their amplitudes
psi_amps = np.zeros(2**TOT, dtype=complex)
psi_good_amps = np.zeros(2**TOT, dtype=complex)
psi_bad_amps = np.zeros(2**TOT, dtype=complex)

for b in range(2**N):
    idx0 = b * 2       # ancilla = 0
    idx1 = b * 2 + 1   # ancilla = 1
    p = q_prob[b]
    z = Z[b]
    psi_amps[idx0] = np.sqrt(p * (1 - z))
    psi_amps[idx1] = np.sqrt(p * z)
    psi_good_amps[idx1] = np.sqrt(p * z)
    psi_bad_amps[idx0] = np.sqrt(p * (1 - z))

# Normalize
psi_vec = psi_amps / np.linalg.norm(psi_amps)
psi_good = psi_good_amps / np.linalg.norm(psi_good_amps)
psi_bad = psi_bad_amps / np.linalg.norm(psi_bad_amps)

# Verify orthogonality and decomposition
print(f"  <psi|psi_good> = {np.vdot(psi_vec, psi_good):.6f}  (want {np.sqrt(MU):.6f})")
print(f"  <psi|psi_bad>  = {np.vdot(psi_vec, psi_bad):.6f}  (want {np.sqrt(1-MU):.6f})")
print(f"  <psi_good|psi_bad> = {np.vdot(psi_good, psi_bad):.6f}  (want 0.000000)")
assert abs(np.vdot(psi_good, psi_bad)) < 1e-10, "FAIL: |psi_good> and |psi_bad> not orthogonal"

# Gram-Schmidt: e1 = |psi>, e2 = component of |psi_good> orthogonal to |psi>
e1 = psi_vec
e2 = psi_good - np.vdot(e1, psi_good) * e1
e2 = e2 / np.linalg.norm(e2)

print(f"  |<e1|e2>| = {abs(np.vdot(e1, e2)):.2e}  (want ~0)")

# Now test Grover
@qml.qnode(dev)
def qnode_G_state(m, use_manual=True):
    A_op()
    for _ in range(m):
        G_op(use_manual_adjoint=use_manual)
    return qml.state()

for m in [1, 2, 3, 5]:
    psi = np.array(qnode_G_state(m, use_manual=True), dtype=complex)
    c1 = np.vdot(e1, psi)
    c2 = np.vdot(e2, psi)
    residual = psi - c1 * e1 - c2 * e2
    leakage = float(np.linalg.norm(residual) ** 2)
    print(f"  m={m}:  manual A^dagger leakage = {leakage:.2e}  (want < 1e-10)")
    assert leakage < 1e-8, f"FAIL: manual G leaks at m={m}"
print("  PASS (manual)")

for m in [1, 2, 3]:
    psi = np.array(qnode_G_state(m, use_manual=False), dtype=complex)
    c1 = np.vdot(e1, psi)
    c2 = np.vdot(e2, psi)
    residual = psi - c1 * e1 - c2 * e2
    leakage = float(np.linalg.norm(residual) ** 2)
    print(f"  m={m}:  qml.adjoint leakage = {leakage:.2e}")

# ------------------------------------------------------------------ TEST 5
print("\n" + "=" * 60)
print("TEST 5: P(anc=1) after G^m  vs  sin^2((2m+1)theta)")
print("=" * 60)

@qml.qnode(dev)
def qnode_circuit_prob(m, use_manual=True):
    A_op()
    for _ in range(m):
        G_op(use_manual_adjoint=use_manual)
    return qml.probs(wires=ANC)

ms = list(range(0, 13))
for use_manual, label in [(True, "manual A^dagger"), (False, "qml.adjoint")]:
    p_circ = np.array([float(qnode_circuit_prob(m, use_manual=use_manual)[1]) for m in ms])
    p_model = np.sin((2 * np.array(ms) + 1) * THETA) ** 2
    err = np.max(np.abs(p_circ - p_model))
    print(f"  {label:20s}: max deviation = {err:.6f}  (want < 1e-6)")
    if err < 1e-6:
        print(f"    >>> PASS")
    else:
        print(f"    >>> FAIL — do not use this version for QAE")

# ------------------------------------------------------------------ summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If manual A^dagger passes all tests, replace qml.adjoint(A_op)()")
print("with A_op_adjoint_manual() in your production QAE code.")
print("If manual also fails, the bug is in A_op, S_chi, or S_zero.")