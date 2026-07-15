"""
network_model.py -- Network reliability as the rare-event problem.

Graph: 3x3 grid, 9 nodes, 12 edges. One qubit per EDGE; bit=1 means the
edge has FAILED. Failure event: source s (corner 0) disconnected from
terminal t (corner 8) through surviving edges.

Why this beats the Bernoulli-spike model scientifically:
  - exact reliability is #P-complete in general (Valiant 1979; Provan &
    Ball 1983), so classical exactness fails at scale for a structural
    reason, not mere rarity
  - the event is NON-EXCHANGEABLE: which edges fail matters (a 2-edge
    cut at a corner disconnects; 4 scattered failures may not), so the
    closed-form i.i.d. tilt that dominated the spike model no longer
    matches the target's structure.

At 12 edges (4096 states) everything is exactly enumerable: ground
truth, p*, and all diagnostics are exact -- same validation philosophy
as spike_model.py.
"""
import numpy as np
from functools import lru_cache

# --- 3x3 grid: nodes 0..8, edges listed once ----------------------------
EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),   # horizontal
         (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]   # vertical
N_EDGES = len(EDGES)          # 12
N_NODES = 9
S, T = 0, 8                   # opposite corners
P_EDGE_FAIL = 0.01            # i.i.d. edge failure probability


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def disconnected(b_int):
    """1 if s-t disconnected when edges in bitmask b_int have failed."""
    parent = list(range(N_NODES))
    for e, (u, v) in enumerate(EDGES):
        if not (b_int >> e) & 1:                 # edge survives
            ru, rv = _find(parent, u), _find(parent, v)
            if ru != rv:
                parent[ru] = rv
    return int(_find(parent, S) != _find(parent, T))


@lru_cache(maxsize=1)
def failure_mask():
    return np.array([disconnected(b) for b in range(2 ** N_EDGES)],
                    dtype=np.float64)


def hamming(n=N_EDGES):
    return np.array([bin(i).count("1") for i in range(2 ** n)])


def p_vector(p=P_EDGE_FAIL):
    k = hamming()
    return (p ** k) * ((1 - p) ** (N_EDGES - k))


def exact_p_fail(p=P_EDGE_FAIL):
    """Exact by enumeration -- validation only, never in the estimator."""
    return float(np.dot(p_vector(p), failure_mask()))


def p_star(p=P_EDGE_FAIL):
    v = p_vector(p) * failure_mask()
    return v / v.sum()


if __name__ == "__main__":
    m = failure_mask()
    pf = exact_p_fail()
    k = hamming()
    print(f"edges={N_EDGES}  failure states={int(m.sum())}/{2**N_EDGES}")
    print(f"exact P(s-t disconnect) = {pf:.6e}  (p_edge={P_EDGE_FAIL})")
    # non-exchangeability exhibit: same Hamming weight, different outcome
    for kk in range(2, 5):
        idx = np.where(k == kk)[0]
        fr = m[idx].mean()
        print(f"  among weight-{kk} states: {fr:.4%} disconnect "
              f"(exchangeable model would force 0% or 100%)")