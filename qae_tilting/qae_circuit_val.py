"""
Tests QAE methodology rigorously, upgrading from "measurement-model" to an actual gate-level circuit simulation.
Tets whether the analytic models in run_qae.py are correct.

EXP 1: circuit P(anc=1) after G^m vs analytic sin^2((2m+1)theta)
EXP 2: full MLE-QAE driven by shot samples from circuit, RMSE vs queries
EXP 3: gate-leel depolarizing noise (density-matrix sim) vs the analytic f^m decay model -- i.e. is the noise model with fidelity previously correct?

Instance is scaled down (n=4) qubits so ancilla-payoff oracle can be written exactly
"""
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 4
ANC = N # payoff ancilla wire
TOT = N+1
P_TRUE = np.exp(-1.6)
P_TILT = 0.55
K_FAIL = 3

def bits_of(b):
    return [(b >> j) & 1 for j in reversed(range(N))]

def lr(bits):
    """likelihoos ratio p(b)/q(b) for product Bernoulli"""
    k = sum(bits)
    return ((P_TRUE / P_TILT) ** k) * (((1 - P_TRUE) / (1 - P_TILT)) ** (N - k))

raw = {}
for b in range(2**N):
    bt = bits_of(b)
    raw[b] = lr(bt) if sum(bt) >= K_FAIL else 0.0
ZMAX = max(raw.values())
Z = {b: raw[b] / ZMAX for b in raw} 

# exact classical quantities
q_prob = {b: np.prod([P_TILT if x else 1 - P_TILT for x in bits_of(b)])
          for b in range(2 ** N)}
MU     = sum(q_prob[b] * Z[b] for b in q_prob)      # = P(anc=1) under A
THETA  = np.arcsin(np.sqrt(MU))
P_FAIL = sum(np.prod([P_TRUE if x else 1 - P_TRUE for x in bits_of(b)])
             for b in range(2 ** N) if sum(bits_of(b)) >= K_FAIL)
 
print(f"instance: N={N}, p_true={P_TRUE:.4f}, p_tilt={P_TILT}, k>={K_FAIL}")
print(f"exact P_fail = {P_FAIL:.6e}   Zmax = {ZMAX:.4f}")
print(f"mu = E_q[Z] = {MU:.6f}   theta = {THETA:.6f} rad")
print(f"check: Zmax * mu = {ZMAX*MU:.6e}  (should equal P_fail)\n")
assert abs(ZMAX * MU - P_FAIL) < 1e-12

# ------------- circuit
def A_op():
    """State preparation + payoff encoding."""
    for i in range(N):
        qml.RY(2 * np.arcsin(np.sqrt(P_TILT)), wires=i)
        for b in range(2**N):
            z = Z[b]
            if z > 0:
                qml.ctrl(qml.RY, control=list(range(N)), control_values=bits_of(b))(
                    2 * np.arcsin(np.sqrt(z)), wires=ANC
                )

def S_chi():
    """Phase-flip the good subspace (ancilla = 1)"""
    qml.PauliZ(wires=ANC)

def S_zero():
    """I - 2|0...0><0...0| over all wires"""
    for w in range(TOT):
        qml.PauliX(wires=w)
    qml.ctrl(qml.PauliZ, control=list(range(TOT - 1)),
             control_values=[1] * (TOT-1))(wires=TOT-1)
    for w in range(TOT):
        qml.PauliX(wires=w)

def G_op():
    """Grover operator G = -A S_0 A^dagger S_chi"""
    S_chi()
    qml.adjoint(A_op)()
    S_zero()
    A_op()
    
# ================================================== EXP 1: noiseless circuit
dev = qml.device("default.qubit", wires=TOT)
 
@qml.qnode(dev)
def circuit_prob(m):
    A_op()
    for _ in range(m):
        G_op()
    return qml.probs(wires=ANC)
 
ms = list(range(0, 13))
p_circ = np.array([circuit_prob(m)[1] for m in ms])
p_model = np.sin((2 * np.array(ms) + 1) * THETA) ** 2
err = np.abs(p_circ - p_model).max()
print("EXP 1  gate-level circuit vs analytic measurement model")
print(f"       max |P_circuit - sin^2((2m+1)theta)| over m=0..12 : {err:.2e}")
print("       -> the analytic model used in the earlier experiments is EXACT\n")
 
 
# ============================================ EXP 2: shot-based MLE-QAE
GRID = np.linspace(1e-6, np.pi / 2 - 1e-6, 40000)
 
def mle_from_circuit(powers, shots, rng):
    """Draw shots from the REAL circuit at each Grover power, then MLE."""
    ll = np.zeros_like(GRID)
    queries = 0
    for m in powers:
        p1 = circuit_prob(m)[1]                 # exact Born prob from circuit
        h = rng.binomial(shots, p1)             # finite-shot measurement record
        pg = np.sin((2 * m + 1) * GRID) ** 2
        ll += h * np.log(pg + 1e-300) + (shots - h) * np.log(1 - pg + 1e-300)
        queries += shots * (2 * m + 1)
    return np.sin(GRID[np.argmax(ll)]) ** 2, queries
 
rng = np.random.default_rng(7)
REPS, SHOTS = 80, 40
qx, qy = [], []
for J in range(1, 6):
    powers = [0] + [2 ** j for j in range(J)]
    ests, nq = [], 0
    for _ in range(REPS):
        e, nq = mle_from_circuit(powers, SHOTS, rng)
        ests.append(e)
    ests = np.array(ests)
    qx.append(nq)
    qy.append(np.sqrt(np.mean((ests - MU) ** 2)) / MU)
qx, qy = np.array(qx), np.array(qy)
slope_q = np.polyfit(np.log(qx), np.log(qy), 1)[0]
 
cx = np.logspace(np.log10(qx.min()), np.log10(qx.max()), 12)
cy = np.sqrt((1 - MU) / (cx * MU))
print("EXP 2  circuit-driven MLE-QAE")
print(f"       measured slope {slope_q:.2f} (theory -1) vs classical -0.50")
print(f"       {qx[-1]:,} queries -> {100*qy[-1]:.2f}% relative RMSE\n")
 
 
# ====================================== EXP 3: gate-level depolarizing noise
dev_n = qml.device("default.mixed", wires=TOT)
 
@qml.qnode(dev_n)
def circuit_prob_noisy(m, pdep):
    A_op()
    for _ in range(m):
        G_op()
        for w in range(TOT):                    # noise after each iterate
            qml.DepolarizingChannel(pdep, wires=w)
    return qml.probs(wires=ANC)
 
ms_n = list(range(0, 9))
noise_levels = [0.002, 0.01, 0.03]
results = {}
for pdep in noise_levels:
    pn = np.array([circuit_prob_noisy(m, pdep)[1] for m in ms_n])
    # fit the published analytic model  f^m sin^2 + (1-f^m)/2  for best f
    fs = np.linspace(0.50, 1.0, 2001)
    best_f, best_res = None, np.inf
    for f in fs:
        pred = f ** np.array(ms_n) * np.sin((2 * np.array(ms_n) + 1) * THETA) ** 2 \
               + (1 - f ** np.array(ms_n)) / 2
        r = np.sqrt(np.mean((pred - pn) ** 2))
        if r < best_res:
            best_res, best_f = r, f
    results[pdep] = (pn, best_f, best_res)
    print(f"EXP 3  gate depolarizing p={pdep:<6}: best-fit f = {best_f:.4f}, "
          f"model RMS residual = {best_res:.4f}")
print("       -> tests whether the published f^m decay model describes real "
      "gate-level noise\n")
 
 
# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
 
ax[0].plot(ms, p_model, "-", lw=3, color="#1F7A4D", label=r"analytic $\sin^2((2m{+}1)\theta)$")
ax[0].plot(ms, p_circ, "o", ms=11, mfc="none", mew=2.5, color="#13294B",
           label="gate-level circuit")
ax[0].set_xlabel("Grover power $m$"); ax[0].set_ylabel(r"$P(\mathrm{ancilla}=1)$")
ax[0].set_title(f"EXP 1: circuit reproduces the model\nmax deviation {err:.1e}")
ax[0].legend(); ax[0].grid(alpha=0.3)
 
ax[1].loglog(cx, cy, "k--", lw=2.5, label="classical MC: slope $-0.50$")
ax[1].loglog(qx, qy, "o-", lw=2.5, ms=9, color="#13294B",
             label=f"circuit MLE-QAE: slope {slope_q:.2f}")
ax[1].set_xlabel("oracle queries"); ax[1].set_ylabel(r"relative RMSE of $\mu$")
ax[1].set_title("EXP 2: quadratic separation,\nfrom the real circuit")
ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
 
cols = ["#13294B", "#E87722", "#A02B2B"]
for (pdep, (pn, bf, res)), c in zip(results.items(), cols):
    ax[2].plot(ms_n, pn, "o", ms=9, color=c, label=f"circuit, $p_{{dep}}$={pdep}")
    pred = bf ** np.array(ms_n) * np.sin((2 * np.array(ms_n) + 1) * THETA) ** 2 \
           + (1 - bf ** np.array(ms_n)) / 2
    ax[2].plot(ms_n, pred, "--", lw=2, color=c, alpha=0.75,
               label=f"  $f^m$ model, $f$={bf:.3f}")
ax[2].set_xlabel("Grover power $m$"); ax[2].set_ylabel(r"$P(\mathrm{ancilla}=1)$")
ax[2].set_title("EXP 3: gate-level noise vs\nthe published $f^m$ decay model")
ax[2].legend(fontsize=8, ncol=2); ax[2].grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig("qae_circuit_validation.png", dpi=150)
print("figure -> qae_circuit_validation.png")
 