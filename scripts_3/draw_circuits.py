"""
draw_circuits.py -- Renders every circuit used in the project.

Outputs (PNG):
  circ1_hea.png            5-layer HEA, 10 qubits (Phases 3-4 / Dir-1 A)
  circ2_shared_ring.png    shared-angle RY + CZ ring (Dir-1 B)
  circ3_dicke_n4.png       staircase + Baertschi-Eidenbenz, n=4 (legible)
  circ3_dicke_n10.png      the full compiled n=10 proposal (dense; for
                           gate-count illustration, not close reading)
  circ4_tilt_prep.png      tilted-Bernoulli state prep (Dir-2)
  circ5_qae_aop.png        QAE A-operator: tilt prep + payoff oracle
                           (black box) + ancilla (Dir-2 / merged)
Angles shown are the actual trained/derived values where applicable.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml

SEED = 42


def save(fig, name):
    fig.savefig(name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------- 1. HEA: 5-layer RY + linear CNOT chain, n=10 ----------------
def hea(n=10, layers=5):
    dev = qml.device("default.qubit", wires=n)
    np.random.seed(SEED)
    params = np.random.uniform(0, 2 * np.pi, (layers, n))

    @qml.qnode(dev)
    def c():
        for L in range(layers):
            for i in range(n):
                qml.RY(params[L, i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.probs(wires=range(n))

    fig, _ = qml.draw_mpl(c, decimals=None, style="pennylane")()
    fig.suptitle("Circuit 1: generic HEA (50 params) -- Dir-1 A / Phases 3-4",
                 y=1.02)
    save(fig, "circ1_hea.png")


# ---------- 2. shared-angle ring: RY(theta_L) on all + CZ ring ----------
def ring(n=10, layers=5):
    dev = qml.device("default.qubit", wires=n)
    np.random.seed(SEED)
    params = np.random.uniform(0, 2 * np.pi, layers)

    @qml.qnode(dev)
    def c():
        for L in range(layers):
            for i in range(n):
                qml.RY(params[L], wires=i)      # SAME angle per layer
            for i in range(n):
                qml.CZ(wires=[i, (i + 1) % n])
        return qml.probs(wires=range(n))

    fig, _ = qml.draw_mpl(c, decimals=None, style="pennylane")()
    fig.suptitle("Circuit 2: shared-angle ring (5 params) -- Dir-1 B\n"
                 "(one theta per layer on every qubit; cyclic CZ ring)",
                 y=1.04)
    save(fig, "circ2_shared_ring.png")


# ---------- 3. Dicke: staircase + BE unitary -----------------------------
def be_unitary(n):
    for l in range(n, 1, -1):
        qml.CNOT(wires=[l - 2, l - 1])
        qml.CRY(2 * np.arccos(np.sqrt(1.0 / l)), wires=[l - 1, l - 2])
        qml.CNOT(wires=[l - 2, l - 1])
        for m in range(2, l):
            a = l - m - 1
            qml.CNOT(wires=[a, l - 1])
            qml.ctrl(qml.RY, control=[l - 1, l - m])(
                2 * np.arccos(np.sqrt(m / l)), wires=a)
            qml.CNOT(wires=[a, l - 1])


def staircase(s, n):
    tail = np.cumsum(s[::-1])[::-1]
    qml.RY(2 * np.arcsin(np.sqrt(tail[1] / tail[0])), wires=n - 1)
    for i in range(1, n):
        cond = tail[i + 1] / tail[i] if tail[i] > 0 else 0.0
        qml.CRY(2 * np.arcsin(np.sqrt(min(cond, 1.0))),
                wires=[n - i, n - 1 - i])


def dicke(n, fname, title):
    dev = qml.device("default.qubit", wires=n)
    s = np.ones(n + 1) / (n + 1)                 # illustrative sectors

    @qml.qnode(dev)
    def c():
        staircase(s, n)
        be_unitary(n)
        return qml.probs(wires=range(n))

    fig, _ = qml.draw_mpl(c, decimals=None, style="pennylane")()
    fig.suptitle(title, y=1.02)
    save(fig, fname)


# ---------- 4. tilted state prep -----------------------------------------
def tilt_prep(n=10, pt=0.4):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def c():
        for i in range(n):
            qml.RY(2 * np.arcsin(np.sqrt(pt)), wires=i)
        return qml.probs(wires=range(n))

    fig, _ = qml.draw_mpl(c, decimals=2, style="pennylane")()
    fig.suptitle("Circuit 4: tilted-Bernoulli state prep, p'=0.4 -- Dir-2\n"
                 "(RY(2 arcsin sqrt(p')) on every qubit)", y=1.04)
    save(fig, "circ4_tilt_prep.png")


# ---------- 5. QAE A-operator schematic ----------------------------------
class PayoffOracle(qml.operation.Operation):
    """Idealized black box: |b>|0> -> |b>(sqrt(1-Z/Zmax)|0>+sqrt(Z/Zmax)|1>)."""
    num_wires = None
    def __init__(self, wires):
        super().__init__(wires=wires)
    @property
    def name(self):
        return "Payoff(Z/Zmax)"
    def decomposition(self):
        return []
    def matrix(self):
        d = 2 ** len(self.wires)
        return np.eye(d)


def qae_aop(n=6, pt=0.4):                        # n=6 for legibility
    dev = qml.device("default.qubit", wires=n + 1)

    @qml.qnode(dev)
    def c():
        for i in range(n):
            qml.RY(2 * np.arcsin(np.sqrt(pt)), wires=i)
        PayoffOracle(wires=list(range(n + 1)))
        return qml.probs(wires=n)

    fig, _ = qml.draw_mpl(c, decimals=2, style="pennylane")()
    fig.suptitle("Circuit 5: QAE A-operator (schematic, ancilla = last wire)\n"
                 "tilted prep + idealized payoff oracle; Grover iterate "
                 "Q = A S0 A^dag S_chi applies this block repeatedly", y=1.06)
    save(fig, "circ5_qae_aop.png")


if __name__ == "__main__":
    hea()
    ring()
    dicke(4, "circ3_dicke_n4.png",
          "Circuit 3a: staircase + Baertschi-Eidenbenz, n=4 (structure)")
    dicke(10, "circ3_dicke_n10.png",
          "Circuit 3b: full compiled Dicke proposal, n=10 "
          "(145 gates, depth 68)")
    tilt_prep()
    qae_aop()