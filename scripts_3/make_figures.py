import numpy as np
from math import comb, exp, asin, sin, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import correlated_model as cm
 
N = cm.N
HW = cm.hamming_weights()
MASK = cm.failure_mask()
P_TGT = cm.p_vector() * MASK
PF = cm.exact_p_fail()
P_STAR = cm.p_star()
CHOOSE = np.array([comb(N, k) for k in range(N + 1)], dtype=float)
 
# ---------- retrain ring + dicke (HEA loaded from disk) -----------------
def train_ring(layers=5, steps=400, seed=42):
    dev = qml.device("default.qubit", wires=N)
    ps = pnp.array(P_STAR, requires_grad=False)
    @qml.qnode(dev)
    def probs(params):
        for L in range(layers):
            for i in range(N):
                qml.RY(params[L], wires=i)
            for i in range(N):
                qml.CZ(wires=[i, (i + 1) % N])
        return qml.probs(wires=range(N))
    np.random.seed(seed)
    params = pnp.array(np.random.uniform(0, 2 * np.pi, layers),
                       requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    for _ in range(steps):
        params, _ = opt.step_and_cost(
            lambda p: -pnp.sum(ps * pnp.log(probs(p) + 1e-12)), params)
    return np.array(probs(params))
 
def train_dicke(steps=3000, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.1, N + 1)
    t = np.array([P_STAR[HW == k].sum() for k in range(N + 1)])
    for _ in range(steps):
        s = np.exp(z - z.max()); s /= s.sum()
        z -= lr * (s - t)
    s = np.exp(z - z.max()); s /= s.sum()
    return s[HW] / CHOOSE[HW]
 
q_hea = np.load("q_corr_kl.npy")
print("training ring ..."); q_ring = train_ring()
q_dicke = train_dicke()
 
def vrf_of(q, trials=20, budget=200_000):
    vs = []
    for t in range(trials):
        r = np.random.default_rng(1000 + t)
        idx = r.choice(2 ** N, size=budget, p=q)
        w = np.where(q[idx] > 0, P_TGT[idx] / q[idx], 0.0)
        vs.append(w.var(ddof=1))
    return PF * (1 - PF) / np.mean(vs)
 
vrfs = {"generic\n(50 par.)": vrf_of(q_hea),
        "shared ring\n(5 par.)": vrf_of(q_ring),
        "Dicke-sector\n(11 par.)": vrf_of(q_dicke)}
print({k.split()[0]: f"{v:,.0f}x" for k, v in vrfs.items()})
 
# ---------------- Fig 1: symmetry chain ---------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
names = list(vrfs); vals = [vrfs[k] for k in names]
bars = ax.bar(names, vals, color=["#c44", "#e90", "#2a7"],
              edgecolor="black", width=0.55)
ax.axhline(418.1, color="black", ls="--", lw=2,
           label="exact classical product-family ceiling (418.1x)")
ax.axhline(1, color="gray", ls=":", label="naive Monte Carlo (1x)")
ax.set_yscale("log"); ax.set_ylim(0.5, 3e7)
ax.set_ylabel("variance reduction factor (log)")
ax.set_title("Proposal symmetry vs performance on the common-shock "
             "mixture\n(same objective, same budget; symmetry increases "
             "left to right)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.4,
            f"{v:,.0f}x" if v < 1e5 else f"{v:.1e}x",
            ha="center", fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, axis="y", which="both", ls="--", alpha=0.3)
fig.tight_layout(); fig.savefig("fig1_symmetry_chain.png", dpi=200)
plt.close(fig)
 
# ---------------- Fig 2: sector-mass profiles ---------------------------
def sector_mass(q):
    return np.array([q[(HW == k) & (MASK > 0)].sum() /
                     max(q[MASK > 0].sum(), 1e-300)
                     for k in range(N + 1)])
 
t_star = np.array([P_STAR[HW == k].sum() for k in range(N + 1)])
fig, ax = plt.subplots(figsize=(8, 5))
ks = np.arange(N + 1)
ax.bar(ks - 0.3, t_star, 0.2, label="optimal target $p^*$ (exact)",
       color="#2a7", edgecolor="black")
ax.bar(ks - 0.1, sector_mass(q_dicke), 0.2, label="Dicke-sector (trained)",
       color="#8d5", edgecolor="black")
ax.bar(ks + 0.1, sector_mass(q_ring), 0.2, label="shared ring (trained)",
       color="#e90", edgecolor="black")
ax.bar(ks + 0.3, sector_mass(q_hea), 0.2, label="generic 50-par.\n(trained)",
       color="#c44", edgecolor="black")
ax.set_yscale("log"); ax.set_ylim(1e-6, 1.5)
ax.set_xticks(ks)
ax.set_xlabel("spike count $k$ (Hamming weight)")
ax.set_ylabel("share of proposal mass on weight-$k$ failure states (log)")
ax.set_title("Why symmetry wins: the target is bimodal in spike count\n"
             "(background failures at k=4; shock failures at k$\\approx$9-10)")
ax.legend(fontsize=9)
ax.grid(True, axis="y", which="both", ls="--", alpha=0.3)
fig.tight_layout(); fig.savefig("fig2_sector_profiles.png", dpi=200)
plt.close(fig)
 
# ---------------- Fig 3: exact ceiling curve ----------------------------
grid = np.linspace(0.02, 0.95, 1000)
vr = np.empty_like(grid)
for i, pt in enumerate(grid):
    qv = (pt ** HW) * ((1 - pt) ** (N - HW))
    vr[i] = PF * (1 - PF) / (np.sum(P_TGT ** 2 / qv) - PF ** 2)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(grid, vr, lw=2)
b = np.argmax(vr)
ax.scatter([grid[b]], [vr[b]], zorder=5, color="#2a7", s=60,
           label=f"true ceiling: {vr[b]:.1f}x at $p'$={grid[b]:.3f}")
e = np.argmin(abs(grid - 0.60))
ax.scatter([0.60], [vr[e]], zorder=5, color="#c44", s=60, marker="s",
           label=f"old grid edge (0.60): {vr[e]:.1f}x")
ax.set_yscale("log")
ax.set_xlabel("tilt parameter $p'$")
ax.set_ylabel("exact VRF of the i.i.d. tilt (log)")
ax.set_title("Exact performance of every classical single tilt\n"
             "(enumerated; the maximum is a hard ceiling for the family)")
ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.3)
fig.tight_layout(); fig.savefig("fig3_ceiling_curve.png", dpi=200)
plt.close(fig)
 
# ---------------- Fig 4: QAE vs classical slopes ------------------------
P0, PT, KF = exp(-4.2), 0.4, 4
PEX = sum(comb(N, k) * P0**k * (1 - P0)**(N - k) for k in range(KF, N + 1))
LR_ = lambda k: (P0 / PT)**k * ((1 - P0) / (1 - PT))**(N - k)
ZMAX = LR_(KF)
MU = sum(comb(N, k) * PT**k * (1 - PT)**(N - k) * LR_(k)
         for k in range(KF, N + 1)) / ZMAX
TH = asin(sqrt(MU))
rng = np.random.default_rng(7)
qae_pts, cl_pts = [], []
gridt = np.linspace(1e-6, np.pi / 2 - 1e-6, 200_000)
for J in range(2, 8):
    powers = [0] + [2**j for j in range(J)]; shots = 60
    errs, qs = [], 0
    for _ in range(60):
        ll = np.zeros_like(gridt); qs = 0
        for m in powers:
            h = rng.binomial(shots, sin((2 * m + 1) * TH) ** 2)
            pg = np.sin((2 * m + 1) * gridt) ** 2
            ll += h * np.log(pg + 1e-300) + (shots - h) * np.log(1 - pg + 1e-300)
            qs += shots * (2 * m + 1)
        est = ZMAX * sin(gridt[np.argmax(ll)]) ** 2
        errs.append((est - PEX) / PEX)
    qae_pts.append((qs, float(np.sqrt(np.mean(np.square(errs))))))
lr_table = np.array([LR_(k) for k in range(N + 1)])
for ns in [10**3, 10**4, 10**5, 10**6]:
    errs = []
    for _ in range(60):
        k = rng.binomial(N, PT, size=ns)
        errs.append((np.mean(lr_table[k] * (k >= KF)) - PEX) / PEX)
    cl_pts.append((ns, float(np.sqrt(np.mean(np.square(errs))))))
 
fig, ax = plt.subplots(figsize=(8, 5))
for pts, lab, c in ((qae_pts, "amplitude estimation", "#2a7"),
                    (cl_pts, "classical tilted MC", "#c44")):
    x, y = zip(*pts)
    s = np.polyfit(np.log(x), np.log(y), 1)[0]
    ax.loglog(x, y, "o-", color=c, label=f"{lab} (fitted slope {s:+.2f})")
ax.set_xlabel("budget: oracle queries (QAE) / samples (classical)")
ax.set_ylabel("relative RMSE (log)")
ax.set_title("Estimation cost of precision under the same tilted proposal\n"
             "(slope -1 vs -0.5: magnification beats repetition)")
ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.3)
fig.tight_layout(); fig.savefig("fig4_qae_slopes.png", dpi=200)
plt.close(fig)
print("figures written")