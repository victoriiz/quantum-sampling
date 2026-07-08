# quantum-sampling

Rare-event probability estimation using a variational quantum circuit as an
importance-sampling proposal (VQIS), trained with a classical Adam optimizer.
Estimators are validated against exact, enumerable ground truth at every
stage -- nothing about the answer enters the estimator.

The project has three generations of models, kept side by side:

| dir | model | status |
|---|---|---|
| `scripts/`   | v1: Exp(1) loads quantized onto 10 qubits, Hamming mask k>=4 | legacy -- kept as a case study (see below) |
| `scripts_2/` | v2: independent Bernoulli spikes, p = exp(-4.2), fail = k>=4 | current baseline pipeline |
| `scripts_3/` | v3: common-shock correlated spikes (two-component mixture) | active -- where product-form classical IS is provably capped |

## v1 (legacy) -- why it was replaced

The v1 estimator built its target distribution from the Monte-Carlo ground
truth it claimed to estimate (circular), and its discrete failure region
(k>=4 spikes) holds only ~0.07% of the true continuous failure mass --
real failures are 1-2 large-spike events. It is retained because it is a
clean demonstration that a large VRF is meaningless without a bias audit
(its weight-dependent mode posts a 43,000x VRF while biased -99.9%).

## v2 (`scripts_2/`) -- current baseline

Bits are i.i.d. Bernoulli spike indicators; p(b) is exact per state, and
P(fail) = P(Bin(10, 0.015) >= 4) = 9.878e-6 is an exact binomial tail used
only for validation. Run order:

```
python spike_model.py    # problem definition + exact constants
python train_kl.py       # trains KL objective AND old MSE objective (same seed)
python eval.py           # 30-trial comparison: bias, variance, MSE, VRF, CI
python make_figures.py   # regenerates all figures from the real runs
```

Headline (n=10, 200k samples/trial): KL objective improves the quantum
proposal ~118x over the old MSE objective (VRF 23x -> 2,711x), but classical
exponential tilting still leads at 35,594x -- expected, since the problem is
product-form and tilting is near-optimal there. Sampling is a classical
simulation from the exact statevector (stated openly; no hardware claim).

## v3 (`scripts_3/`) -- correlated spikes

A latent shock makes p(b) a two-component mixture (shock: p_hi = 0.95;
background: p_lo = 0.03; both mechanisms carry ~half the failure mass).
This provably caps the ENTIRE product family of classical proposals
(exact ceiling ~420x VRF, computed by enumeration) -- the first setting
where beating every classical tilt is mathematically possible.

```
python correlated_model.py   # mixture definition + exact P(fail)
python run_correlated.py     # KL training + naive MC / best-tilt / quantum
```

Current status: best single tilt ~220x, quantum-KL ~56x. The gap between
56x and the ~420x ceiling is the open problem.

## Layout

```
scripts/     v1 legacy pipeline (see case study above)
scripts_2/   v2 baseline pipeline
scripts_3/   v3 correlated-mixture pipeline
slurm/       Delta / SLURM job scripts
notebooks/   exploratory analysis
figures/     generated figures
outputs/     generated .npy/.json (gitignored; regenerate via scripts)
```

## Validation philosophy

Every model is small enough to enumerate (<= 2^12 states): p(b) is exact
and closed-form, ground truth is an exact sum, and every reported estimator
carries bias, variance, MSE, VRF, and ESS -- because a VRF without a bias
audit is how v1 fooled us.
