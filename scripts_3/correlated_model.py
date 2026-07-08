"""
Common-shock spike model.

latent shock Z ~ Bern(p_shock) hits whole system:
    Z=1 (shock): every bit spikes independently with p_hi
    Z=0 (no shock): every bit spikes independently with p_lo
so p(b) = p_shock * p_hi^k (1-p_hi)^(n-k)
        + (1-p_shock)(p_lo^k)(1-p_lo)^(n-k)

everything else is exact: p(b) closed-form per state, P(fail) a mixture of two binomial tails.
Failure = Hamming weight >= K_FAIL
"""
import numpy as np
from math import comb

N = 10
K_FAIL = 4
P_SHOCK = 1.4e-4
P_HI = 0.95
P_LO = 0.03 # background spike rate (failures at k = 4)

def tail(p):
        return sum(comb(N, k) * p**k * (1-p)**(N-k) for k in range(K_FAIL, N+1))

def hamming_weights(n=N):
    return np.array([bin(i).count("1") for i in range(2**n)])

def failure_mask(n=N, k_fail=K_FAIL):
    return (hamming_weights(n) >= k_fail).astype(np.float64)

def p_vector(n=N):
    k = hamming_weights(n)
    hi = (P_HI**k) * ((1-P_HI)**(n-k))
    lo = (P_LO**k) * ((1-P_LO)**(n-k))
    return P_SHOCK*hi + (1-P_SHOCK)*lo

def exact_p_fail(n=N, k_fail=K_FAIL):
    return P_SHOCK * tail(P_HI) + (1- P_SHOCK)*tail(P_LO)

def p_star(n=N, k_fail=K_FAIL):
    v = p_vector(n) * failure_mask(n, k_fail)
    return v / v.sum()

def component_split(n=N, k_fail=K_FAIL):
    """How much failure probability each component carries"""
    shock = P_SHOCK * tail(P_HI)
    bg = (1 - P_SHOCK) * tail(P_LO)
    return shock, bg

if __name__ == "__main__":
    pf = exact_p_fail()
    s, b = component_split()
    print(f"exact P(fail) = {pf:.6e}")
    print(f"  shock-driven: {s:.3e} ({s/pf:.1%})   "
          f"background:   {b:.3e} ({b/pf:.1%})")
    print("failure is a MIXTURE of two mechanisms -> single tilt must "
          "pick one and mis-weight the other")