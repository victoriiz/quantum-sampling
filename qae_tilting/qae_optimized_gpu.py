"""
N=10 gate-level QAE validation with a QFT-based Hamming-weight oracle.
Replaces the 1024 multi-controlled-gate naive oracle with O(N log N) gates.

Oracle complexity:
  naive:   2^N multi-controlled RY  ≈ 1024 gates (each decomposes to ~20 gates)
  QFT:     QFT + N*W phase shifts + IQFT + (N+1) controlled RY on W qubits
           ≈ 50 gates total for N=10, W=4

This makes N=10 validation feasible on CPU in minutes, and instantaneous on GPU.
"""
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ instance
N      = 10
W      = int(np.ceil(np.log2(N + 1)))   # weight register size = 4
ANC    = N + W                          # payoff ancilla wire
TOT    = N + W + 1
P_TRUE = np.exp(-4.2)                   # ~0.014996
P_TILT = 0.40
K_FAIL = 4

def bits_of(b):
    return [(b >> j) & 1 for j in reversed(range(N))]

def lr(bits):
    k = sum(bits)
    return ((P_TRUE / P_TILT) ** k) * (((1 - P_TRUE) / (1 - P_TILT)) ** (N - k))

# Payoff by Hamming weight (precomputed)
raw = {}
for b in range(2 ** N):
    bt = bits_of(b)
    raw[b] = lr(bt) if sum(bt) >= K_FAIL else 0.0
ZMAX = max(raw.values())

Z_by_weight = np.zeros(N + 1)
for b in range(2 ** N):
    bt = bits_of(b)
    k = sum(bt)
    if k >= K_FAIL:
        Z_by_weight[k] = raw[b] / ZMAX

# Exact quantities
q_prob = {b: np.prod([P_TILT if x else 1 - P_TILT for x in bits_of(b)])
          for b in range(2 ** N)}
MU = sum(q_prob[b] * Z_by_weight[sum(bits_of(b))] for b in q_prob)
THETA = np.arcsin(np.sqrt(MU))
P_FAIL = sum(np.prod([P_TRUE if x else 1 - P_TRUE for x in bits_of(b)])
             for b in range(2 ** N) if sum(bits_of(b)) >= K_FAIL)

print(f"instance: N={N}, p_true={P_TRUE:.6f}, p_tilt={P_TILT}, k>={K_FAIL}")
print(f"exact P_fail = {P_FAIL:.6e}")
print(f"mu = {MU:.6f}, theta = {THETA:.6f} rad")
print(f"check: Zmax * mu = {ZMAX * MU:.6e} (should equal P_fail)\n")
assert abs(ZMAX * MU - P_FAIL) < 1e-12

# ------------------------------------------------------------------ qubits
DATA    = list(range(N))           # [0, 1, ..., 9]
WEIGHT  = list(range(N, N + W))    # [10, 11, 12, 13]  (w0=LSB, w3=MSB)
ANCILLA = N + W                    # 14

# ============================================================= components

def data_prep():
    """Prepare tilted product state on DATA qubits."""
    for i in DATA:
        qml.RY(2 * np.arcsin(np.sqrt(P_TILT)), wires=i)

def data_prep_adjoint():
    """Inverse of data_prep."""
    for i in reversed(DATA):
        qml.RY(-2 * np.arcsin(np.sqrt(P_TILT)), wires=i)

def add_hamming_weight():
    """
    Compute Hamming weight of DATA qubits into WEIGHT register.
    WEIGHT must start in |0...0>.
    Uses QFT adder: QFT, then controlled phase shifts, then IQFT.
    """
    # QFT with MSB first: wires [w3, w2, w1, w0]
    qml.QFT(wires=list(reversed(WEIGHT)))
    # For each data qubit, add 1 to the register if the qubit is |1>
    for i in DATA:
        for j, w in enumerate(WEIGHT):
            # j=0 -> LSB (w0) -> angle = 2*pi/2 = pi
            # j=1 -> w1 -> angle = 2*pi/4 = pi/2
            # etc.
            angle = 2 * np.pi / (2 ** (j + 1))
            qml.ctrl(qml.PhaseShift, control=i)(angle, wires=w)
    # Inverse QFT
    qml.adjoint(qml.QFT)(wires=list(reversed(WEIGHT)))

def uncompute_hamming_weight():
    """Inverse of add_hamming_weight."""
    qml.QFT(wires=list(reversed(WEIGHT)))
    for i in reversed(DATA):
        for j, w in enumerate(WEIGHT):
            angle = -2 * np.pi / (2 ** (j + 1))
            qml.ctrl(qml.PhaseShift, control=i)(angle, wires=w)
    qml.adjoint(qml.QFT)(wires=list(reversed(WEIGHT)))

def payoff_rotation():
    """
    Apply RY on ancilla conditioned on Hamming weight k.
    For each k with Z[k] > 0, rotate ancilla by 2*arcsin(sqrt(Z[k])).
    """
    for k in range(N + 1):
        if Z_by_weight[k] > 0:
            # Binary representation of k with W bits
            binary = format(k, f'0{W}b')           # e.g., k=3 -> '0011'
            # WEIGHT = [w0, w1, w2, w3] where w0 is LSB
            # binary[0] is MSB, binary[-1] is LSB
            # So we reverse binary to match WEIGHT wire order
            cv = [int(c) for c in reversed(binary)]
            qml.ctrl(qml.RY, control=WEIGHT, control_values=cv)(
                2 * np.arcsin(np.sqrt(Z_by_weight[k])), wires=ANCILLA)

def payoff_rotation_adjoint():
    """Inverse of payoff_rotation."""
    for k in reversed(range(N + 1)):
        if Z_by_weight[k] > 0:
            binary = format(k, f'0{W}b')
            cv = [int(c) for c in reversed(binary)]
            qml.ctrl(qml.RY, control=WEIGHT, control_values=cv)(
                -2 * np.arcsin(np.sqrt(Z_by_weight[k])), wires=ANCILLA)

def A_op():
    """State preparation + payoff encoding (forward)."""
    data_prep()
    add_hamming_weight()
    payoff_rotation()

def A_op_adjoint_manual():
    """Explicit inverse of A_op."""
    payoff_rotation_adjoint()
    uncompute_hamming_weight()
    data_prep_adjoint()

def S_chi():
    """Phase-flip the good subspace (ancilla = 1)."""
    qml.PauliZ(wires=ANCILLA)

def S_zero():
    """Reflection about |0...0>: I - 2|0..0><0..0|."""
    for w in range(TOT):
        qml.PauliX(wires=w)
    qml.ctrl(qml.PauliZ, control=list(range(TOT - 1)),
             control_values=[1] * (TOT - 1))(wires=TOT - 1)
    for w in range(TOT):
        qml.PauliX(wires=w)

def G_op():
    """Grover iterate G = A S_0 A^dagger S_chi."""
    S_chi()
    A_op_adjoint_manual()
    S_zero()
    A_op()

# ================================================================== VERIFY
print("=" * 60)
print("VERIFY: Hamming-weight adder correctness")
print("=" * 60)

dev_verify = qml.device("default.qubit", wires=TOT)

@qml.qnode(dev_verify)
def test_adder(data_bits):
    """Test adder on a specific basis state."""
    for i, b in enumerate(data_bits):
        if b:
            qml.PauliX(wires=i)
    add_hamming_weight()
    return qml.probs(wires=WEIGHT)

# Test a few basis states
test_cases = [
    ([0]*10, 0),
    ([1,0,0,0,0,0,0,0,0,0], 1),
    ([1,1,0,0,0,0,0,0,0,0], 2),
    ([1,1,1,1,0,0,0,0,0,0], 4),
    ([1]*10, 10),
]
for bits, expected_k in test_cases:
    probs = test_adder(bits)
    measured_k = np.argmax(probs)
    status = "PASS" if measured_k == expected_k else "FAIL"
    print(f"  weight {sum(bits):2d}: measured register state = {measured_k:2d}  {status}")
    if status == "FAIL":
        print(f"    -> full probs: {probs}")
        raise AssertionError("Adder verification failed — check bit ordering")

print("  All adder tests passed.\n")

# ================================================================== EXP 1
print("=" * 60)
print("EXP 1: gate-level circuit vs analytic sin^2((2m+1)theta)")
print("=" * 60)

dev = qml.device("default.qubit", wires=TOT)
# To use GPU on Delta, replace with:
# dev = qml.device("lightning.gpu", wires=TOT)

@qml.qnode(dev)
def circuit_prob(m):
    A_op()
    for _ in range(m):
        G_op()
    return qml.probs(wires=ANCILLA)

ms = list(range(0, 13))
p_circ = np.array([float(circuit_prob(m)[1]) for m in ms])
p_model = np.sin((2 * np.array(ms) + 1) * THETA) ** 2
err = np.max(np.abs(p_circ - p_model))
print(f"  max |P_circuit - sin^2((2m+1)theta)| over m=0..12 : {err:.2e}")
assert err < 1e-6, f"FAIL: circuit deviates by {err} from analytic model"
print("  PASS — circuit reproduces the analytic model\n")

# ================================================================== EXP 2
print("=" * 60)
print("EXP 2: shot-based MLE-QAE from the real circuit")
print("=" * 60)

GRID = np.linspace(1e-6, np.pi / 2 - 1e-6, 40000)

def mle_from_circuit(powers, shots, rng):
    """Draw shots from the REAL circuit at each Grover power, then MLE."""
    ll = np.zeros_like(GRID)
    queries = 0
    for m in powers:
        p1 = float(circuit_prob(m)[1])          # exact Born prob from circuit
        h = rng.binomial(shots, p1)             # finite-shot measurement record
        pg = np.sin((2 * m + 1) * GRID) ** 2
        ll += h * np.log(pg + 1e-300) + (shots - h) * np.log(1 - pg + 1e-300)
        queries += shots * (2 * m + 1)
    return np.sin(GRID[np.argmax(ll)]) ** 2, queries

rng = np.random.default_rng(7)
REPS, SHOTS = 80, 60
qx, qy = [], []
for J in range(1, 7):
    powers = [0] + [2 ** j for j in range(J)]
    ests, nq = [], 0
    for _ in range(REPS):
        e, nq = mle_from_circuit(powers, SHOTS, rng)
        ests.append(e)
    ests = np.array(ests)
    qx.append(nq)
    qy.append(np.sqrt(np.mean((ests - MU) ** 2)) / MU)
    print(f"  J={J}, powers={powers}, {nq:,} queries -> {100*qy[-1]:.2f}% rel. RMSE")

qx, qy = np.array(qx), np.array(qy)
slope_q = np.polyfit(np.log(qx), np.log(qy), 1)[0]
print(f"\n  measured slope {slope_q:.2f} (theory -1) vs classical -0.50")

# Classical reference line
cx = np.logspace(np.log10(qx.min()), np.log10(qx.max()), 12)
cy = np.sqrt((1 - MU) / (cx * MU))

# ================================================================== EXP 3
print("\n" + "=" * 60)
print("EXP 3: gate-level depolarizing noise vs f^m decay model")
print("=" * 60)

dev_n = qml.device("default.mixed", wires=TOT)
# NOTE: default.mixed is CPU-only. For noise, we stay on CPU.

@qml.qnode(dev_n)
def circuit_prob_noisy(m, pdep):
    A_op()
    for _ in range(m):
        G_op()
        for w in range(TOT):
            qml.DepolarizingChannel(pdep, wires=w)
    return qml.probs(wires=ANCILLA)

ms_n = list(range(0, 9))
noise_levels = [0.002, 0.01, 0.03]
results = {}
for pdep in noise_levels:
    pn = np.array([float(circuit_prob_noisy(m, pdep)[1]) for m in ms_n])
    fs = np.linspace(0.50, 1.0, 2001)
    best_f, best_res = None, np.inf
    for f in fs:
        pred = f ** np.array(ms_n) * np.sin((2 * np.array(ms_n) + 1) * THETA) ** 2 \
               + (1 - f ** np.array(ms_n)) / 2
        r = np.sqrt(np.mean((pred - pn) ** 2))
        if r < best_res:
            best_res, best_f = r, f
    results[pdep] = (pn, best_f, best_res)
    print(f"  gate depolarizing p={pdep:<6}: best-fit f = {best_f:.4f}, "
          f"model RMS residual = {best_res:.4f}")

# ================================================================== figure
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

ax[0].plot(ms, p_model, "-", lw=3, color="#1F7A4D", label=r"analytic $\sin^2((2m{+}1)	heta)$")
ax[0].plot(ms, p_circ, "o", ms=11, mfc="none", mew=2.5, color="#13294B",
           label="gate-level circuit")
ax[0].set_xlabel("Grover power $m$")
ax[0].set_ylabel(r"$P(\mathrm{ancilla}=1)$")
ax[0].set_title(f"EXP 1: circuit reproduces model\nmax deviation {err:.1e}")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].loglog(cx, cy, "k--", lw=2.5, label="classical MC: slope $-0.50$")
ax[1].loglog(qx, qy, "o-", lw=2.5, ms=9, color="#13294B",
             label=f"circuit MLE-QAE: slope {slope_q:.2f}")
ax[1].set_xlabel("oracle queries")
ax[1].set_ylabel(r"relative RMSE of $\mu$")
ax[1].set_title("EXP 2: quadratic separation\nfrom the real circuit")
ax[1].legend()
ax[1].grid(alpha=0.3, which="both")

cols = ["#13294B", "#E87722", "#A02B2B"]
for (pdep, (pn, bf, res)), c in zip(results.items(), cols):
    ax[2].plot(ms_n, pn, "o", ms=9, color=c, label=f"circuit, $p_{{dep}}$={pdep}")
    pred = bf ** np.array(ms_n) * np.sin((2 * np.array(ms_n) + 1) * THETA) ** 2 \
           + (1 - bf ** np.array(ms_n)) / 2
    ax[2].plot(ms_n, pred, "--", lw=2, color=c, alpha=0.75,
               label=f"  $f^m$ model, $f$={bf:.3f}")
ax[2].set_xlabel("Grover power $m$")
ax[2].set_ylabel(r"$P(\mathrm{ancilla}=1)$")
ax[2].set_title("EXP 3: gate-level noise vs\nthe published $f^m$ decay model")
ax[2].legend(fontsize=8, ncol=2)
ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("qae_circuit_validation_optimized.png", dpi=150)
print("\nfigure -> qae_circuit_validation_optimized.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
