import numpy as np
from math import comb
import pennylane as qml
import correlated_model as cm

N = cm.N
HW = cm.hamming_weights()
MASK = cm.failure_mask()
P_TGT = cm.p_vector() * MASK
PF = cm.exact_p_fail()
P_STAR = cm.p_star()
CHOOSE = np.array([comb(N, k) for k in range(N+1)], dtype=float)

# trained sector probs
def sector_target():
    return np.array([P_STAR[HW == k].sum() for k in range(N+1)])

S = sector_target()

# circuit pieces
def staircase(s):
    """Prepare sum_k sqrt(s_k)|0^(n-k) 1^k>: ones on the LAST k qubits
    (BE input convention). Qubit n-1-i is 1 iff k > i."""
    tail = np.cumsum(s[::-1])[::-1]          # tail[i] = P(k >= i)
    qml.RY(2 * np.arcsin(np.sqrt(tail[1] / tail[0])), wires=N - 1)
    for i in range(1, N):
        cond = tail[i + 1] / tail[i] if tail[i] > 0 else 0.0
        qml.CRY(2 * np.arcsin(np.sqrt(min(cond, 1.0))),
                wires=[N - i, N - 1 - i])

def be_unitary(n):
    """Baertschi-Eidenbenz U_n: |1^k 0^(n-k)> -> |D^n_k> for all k.
    Composition of Split & Cyclic Shift blocks; angles arccos(sqrt(m/l))."""
    for l in range(n, 1, -1):
        # two-qubit block on (l-2, l-1)
        qml.CNOT(wires=[l - 2, l - 1])
        qml.CRY(2 * np.arccos(np.sqrt(1.0 / l)), wires=[l - 1, l - 2])
        qml.CNOT(wires=[l - 2, l - 1])
        # three-qubit blocks
        for m in range(2, l):
            a = l - m - 1                      # target
            qml.CNOT(wires=[a, l - 1])
            ang = 2 * np.arccos(np.sqrt(m / l))
            qml.ctrl(qml.RY, control=[l - 1, l - m])(ang, wires=a)
            qml.CNOT(wires=[a, l - 1])

dev = qml.device("default.qubit", wires=N)

@qml.qnode(dev)
def compiled_state(s):
    staircase(s)
    be_unitary(N)
    return qml.state()

# ---------------- verification -------------------------------------------
def exact_target_state(s):
    """sum_k sqrt(s_k)/sqrt(C(n,k)) on every weight-k basis state."""
    amp = np.sqrt(s[HW] / CHOOSE[HW])
    return amp / np.linalg.norm(amp)

def verify():
    psi = np.array(compiled_state(S)).real
    tgt = exact_target_state(S)
    # global phase/sign fix
    if np.dot(psi, tgt) < 0:
        psi = -psi
    err = np.max(np.abs(np.abs(psi) - tgt))
    fid = np.dot(np.abs(psi), tgt) ** 2
    print(f"verification: max |amplitude error| = {err:.2e}, "
          f"fidelity = {fid:.12f}")
    return err < 1e-8

# ---------------- characterization ---------------------------------------
def gate_counts():
    specs = qml.specs(compiled_state)(S)
    res = specs["resources"]
    print(f"compiled circuit: {res.num_gates} gates, depth {res.depth}, "
          f"gate types: {dict(res.gate_types)}")

def error_to_variance(deltas=(0, 1e-4, 1e-3, 1e-2, 3e-2),
                      trials=10, budget=200_000, seed=5):
    print(f"\nstate-prep error -> variance transfer "
          f"(Gaussian amplitude perturbation, relative size delta):")
    print(f"{'delta':>8} {'VRF':>14} {'bias':>9}")
    tgt = exact_target_state(S)
    for d in deltas:
        rng = np.random.default_rng(seed)
        amp = tgt * (1 + d * rng.normal(size=tgt.shape))
        q = amp ** 2 / np.sum(amp ** 2)
        vs, es = [], []
        for t in range(trials):
            r = np.random.default_rng(2000 + t)
            idx = r.choice(2 ** N, size=budget, p=q)
            w = np.where(q[idx] > 0, P_TGT[idx] / q[idx], 0.0)
            vs.append(w.var(ddof=1)); es.append(w.mean())
        vrf = PF * (1 - PF) / np.mean(vs)
        print(f"{d:>8} {vrf:>14,.0f}x {100*(np.mean(es)-PF)/PF:>+8.2f}%")

if __name__ == "__main__":
    ok = verify()
    if not ok:
        raise SystemExit("compiled state does NOT match target -- "
                         "do not use downstream")
    gate_counts()
    error_to_variance()