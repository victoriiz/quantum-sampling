"""
Reframe the problem to kill the circularity.

Model: n independent Bernoulli 'spike' indicators, b_i ~ Bern(p_spike)
p_spike = P(x > c) = exp(-c) for x ~ Exp(1), c = 4.2
Failure event: Hamming weight(b) >= K_FAIL

everything below is computable without Monte Carlo and without knowing the p.d.f in advance
- p(b) per bitstring: p^k (1-p)^(n-k)
- exact P(fail)
- p*(b) = p(b) 1_fail(b) / P(fail): IS-optimal target shape
"""
import numpy as np
from math import comb, exp

C_SPIKE =4.2
P_SPIKE = exp(-C_SPIKE)
N = 10
K_FAIL = 4

def hamming_weights(n):
    return np.array([bin(i).count("1") for i in range(2**n)])

def failure_mask(n, k_fail=K_FAIL):
    return (hamming_weights(n) >= k_fail).astype(np.float64)

def p_vector(n, p=P_SPIKE):
    k = hamming_weights(n)
    return (p ** k) * ((1.0 - p) ** (n-k))

def exact_p_fail(n, p=P_SPIKE, k_fail=K_FAIL):
    """computes exact binomial tail; for validation only."""
    return sum(comb(n, k) * p**k * (1-p)**(n-k) for k in range(k_fail, n+1))

def p_star(n, p=P_SPIKE, k_fail=K_FAIL):
    """IS-optimal target, p(b) restricted to failure set, normalized."""
    pv = p_vector(n, p) * failure_mask(n, k_fail)
    return pv / pv.sum()

if __name__ == "__main__":
    for n in (10, 12, 14):
        pf = exact_p_fail(n)
        nf = int(failure_mask(n).sum())
        print(f"n={n:2d}  |failure set|={nf:5d}/{2**n:5d}  "
              f"exact P(fail)={pf:.6e}")