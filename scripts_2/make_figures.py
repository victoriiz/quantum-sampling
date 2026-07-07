import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spike_model as sm

N_QUBITS = 10
BUDGET = 200_000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

MASK = sm.failure_mask(N_QUBITS).astype(bool)
P_TARGET = sm.p_vector(N_QUBITS) * MASK
P_STAR = sm.p_star(N_QUBITS)
PF = sm.exact_p_fail(N_QUBITS)

q_kl = np.load(os.path.join(OUT, "q_kl.npy"))
q_old = np.load(os.path.join(OUT, "q_old.npy"))

# ---------------- Exp1: concentration over failure states --------------
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.logspace(-12, 0, 48)
ax.hist(np.clip(q_old[MASK], 1e-12, None), bins=bins, alpha=0.55,
        label="q, old MSE objective", edgecolor="black")
ax.hist(np.clip(q_kl[MASK], 1e-12, None), bins=bins, alpha=0.55,
        label="q, KL objective", edgecolor="black")
ax.hist(P_STAR[MASK], bins=bins, histtype="step", linewidth=2,
        color="green", label="optimal target p* (exact)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("probability mass on failure state  $q_i$")
ax.set_ylabel("state count")
ax.set_title("Proposal concentration over the 848 failure states\n"
             "(trained circuits vs exact IS-optimal target)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "exp1_concentration.png"), dpi=200)
plt.close(fig)

# ---------------- Exp2: clipping sweep, exact weights -------------------
rng = np.random.default_rng(1337)
caps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, np.inf]
results = {}
for tag, q in (("KL", q_kl), ("old MSE", q_old)):
    idx = rng.choice(2 ** N_QUBITS, size=BUDGET, p=q)
    raw = np.where(q[idx] > 0, P_TARGET[idx] / q[idx], 0.0)
    rows = []
    for c in caps:
        w = np.minimum(raw, c)
        rows.append((c, w.mean(), w.var(ddof=1)))
    results[tag] = rows

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
colors = {"KL": "tab:blue", "old MSE": "tab:red"}
for tag, rows in results.items():
    cs = [r[0] if np.isfinite(r[0]) else 10.0 for r in rows]
    ests = [r[1] for r in rows]
    vrfs = [(PF * (1 - PF)) / r[2] if r[2] > 0 else np.nan for r in rows]
    ax1.plot(cs, ests, marker="o", color=colors[tag],
             label=f"estimate ({tag})")
    ax2.plot(cs, vrfs, marker="s", ls="--", color=colors[tag],
             alpha=0.6, label=f"VRF ({tag})")
ax1.axhline(PF, color="black", ls=":", label="exact P(fail)")
ax1.set_xscale("log")
ax2.set_yscale("log")
ax1.set_xlabel("clipping threshold  $C$  (rightmost point = no clipping)")
ax1.set_ylabel("IS estimate")
ax2.set_ylabel("variance reduction factor (log)")
ax1.set_title("Clipping bias-variance sweep with exact $p(b)$ weights")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
ax1.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "exp2_clipping_sweep.png"), dpi=200)
plt.close(fig)

# ---------------- Exp3: real sorted profiles ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for label, vec, style in (
        ("optimal target p* (exact)", P_STAR[MASK], "-"),
        ("trained q, KL objective", q_kl[MASK] / q_kl[MASK].sum(), "-"),
        ("trained q, old MSE objective",
         q_old[MASK] / q_old[MASK].sum(), "--")):
    ax.plot(np.sort(vec)[::-1], style, linewidth=2, label=label)
ax.set_yscale("log")
ax.set_ylim(1e-12, 1)
ax.set_xlabel("failure states, sorted by probability")
ax.set_ylabel("normalized probability within failure set (log)")
ax.set_title("Trained proposal shape vs optimal target\n"
             "(real circuits -- replaces synthetic Dirichlet profiles)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "exp3_trained_profiles.png"), dpi=200)
plt.close(fig)

print("figures written:",
      sorted(os.listdir(FIG)))

