# quantum-sampling

Rare-event probability estimation using quantum circuits as
importance-sampling proposals and estimators (VQIS), trained with a
classical Adam optimizer. Estimators are validated against exact,
enumerable ground truth at every stage.

The project has three generations of models plus two estimation
studies.

| dir | model / study | status |
|---|---|---|
| `scripts_legacy/` | v1: Exp(1) loads quantized onto 10 qubits, Hamming mask k>=4 | retired -- kept as a case study (see below) |
| `scripts_2/` | v2 / Benchmark A: independent Bernoulli spikes, p = exp(-4.2), fail = k>=4 | complete -- current baseline pipeline|
| `scripts_3/` | v3 / Benchmark B: common-shock correlated spikes; symmetry ladder; compiled Dicke circuit; merged pipeline | complete -- the headline results |
| `qnet/` | network reliability: 3x3 grid, s-t disconnection | complete -- separate model sidequest |
| `qae_tilting/` | amplitude estimation on the tilted estimator | complete -- the quantum amplitude estimation study |
| `circuits/` | circuit diagrams (full + condensed) and generators | -- |
| `figures/` | cross-experiment / report figures ONLY | -- |
| `slurm/` | Delta cluster job files | -- |

**Figure rule:** figures live with the experiment that generates them
(e.g. `scripts_2/figures/`); root `figures/` holds only cross-cutting
report figures.

## v1 (`scripts_legacy/`) -- why it was replaced

The v1 estimator built its target distribution from the Monte-Carlo
ground truth it claimed to estimate (circular), and its discrete
failure region (k>=4 spikes) holds only ~0.07% of the true continuous
failure mass -- exact FFT analysis shows real failures are 1-2
large-spike events. It is retained because it is a clean demonstration
that a large VRF is meaningless without a bias audit: its
weight-dependent mode posts a 43,000x VRF while biased -99.9%.

- `classical_env.py` -- the continuous Exp(1) load model and MC ground truth
- `test_failure_subspace.py` -- exact FFT conditional analysis (the
  computation that killed v1's failure region)
- `quantum_ansatz.py` / `quantum_optimization.py` / `quantum_sampler.py`
  -- the retired circuit, training loop, and sampler
- `estimate_conditional_weights.py`, `run_benchmarks.py` -- the circular
  estimator and its benchmark harness (kept as the cautionary artifact)
- `plot_domain_mapping.py`, `plot_mask_classification.py` -- v1 figures
- `analysis.ipynb` -- early exploratory notebook (fossil, predates v1 naming)
- `figures/` -- v1-era plots

## v2 / Benchmark A (`scripts_2/`) -- the fair fight

Bit i encodes the event "load i spiked" (x_i > 4.2), so p = exp(-4.2)
~ 0.015 and every state probability is closed-form: p(b) = p^k
(1-p)^(n-k). The exact answer P_fail = 9.878e-6 (binomial tail)./ Two findings: swapping the training
objective from mass-matching to forward KL against the ideal proposal
p* improves the same 50-parameter circuit 23x -> 2,711x (the objective,
not the ansatz, was the binding constraint); and classical exponential
tilting wins the benchmark at 35,594x. Product-form
problems are conceded to classical tilting.

- `spike_model.py` -- model constants, exact tail, p*, tilting baseline
- `train_kl.py` -- circuit + both objectives (mass-target and forward KL)
- `eval.py` -- the standard audit: bias / variance / VRF / MSE / ESS
  over seeded trials
- `make_figures.py` -- generates `figures/exp1..exp3`
- `figures/` -- concentration, clipping-sweep, and trained-profile plots

## v3 / Benchmark B (`scripts_3/`) -- correlated shock, ceiling, symmetry

Add a hidden common shock (Z ~ Bern(1.4e-4); bits Bern(0.95) under shock,
Bern(0.03) otherwise) makes p(b) a two-component mixture with exact
P_fail = 2.871e-4 and failure mass split ~50/50 between two humps in
spike count (k=4 and k~9.5). Any i.i.d. tilt has a single geometric
decay rate, so no tilt fits two humps -- and with 1,024 states the
entire tilt family is provably capped: max VRF = 418.1x at p' = 0.678,
by exact enumeration. The symmetry ladder then shows structure beats
capacity: generic 50-param circuit 56x; shared-angle ring (5 params)
182x; Dicke-sector (11 params) >1e6x at 0.002 nats from the optimum,
bias +0.00%, ESS ~ full budget -- past the cap, at the zero-variance
optimum. The merged study shows proposal quality and QAE are
SUBSTITUTES: stacking QAE on the near-perfect proposal is worse than
classical counting (240 queries vs a handful of samples).

- `correlated_model.py` -- mixture model, exact quantities, p*
- `run_correlated.py` -- KL training on the mixture + the exact tilt
  ceiling (the 418.1x computation)
- `train_symmetric.py` -- the symmetry ladder: HEA vs shared ring vs
  Dicke-sector, one experiment
- `ladder_on_A.py` -- CONTROL: the ladder applied to Benchmark A.
  Finding: A's tilt ceiling is 35,584x at p'=0.400 (the textbook tilt
  IS optimal there), and the Dicke sampler beats it hollowly (A's p*
  is a one-line conditioned binomial -- a classical lookup ties it)
- `compile_dicke.py` -- Baertschi-Eidenbenz preparation circuit,
  compiled and verified to fidelity 1-1e-16 (145 gates, depth 68),
  plus the error->variance transfer law VRF ~ 1.2/delta^2
- `run_merged.py` -- the proposal x estimator 2x2 attribution (the
  anti-synergy result)
- `draw_circuits.py`, `make_figures.py` -- diagram and figure generators

## Network reliability (`qnet/`) -- the non-exchangeable sidequest

3x3 grid, 12 edges failing independently at 1%, failure = corner-to-
corner disconnection (exact P = 2.08e-4; #P-complete at scale). Here
which edges fail matters, not how many. Classical cross-entropy
per-edge tilts win 269x vs the quantum circuit's 16.8x. It is this repo's own counterexample against claiming
"quantum helps whenever variables are correlated."

## Amplitude estimation (`qae_tilting/`) -- the quantum estimation study

Freezes the proposal at Benchmark A's tilt and races only the
estimator. Runs MLE-QAE over a ladder of Grover powers measures error-decay
exponents -0.89 (theory -1) vs classical -0.50 (theory -0.5): 64x
fewer queries at 0.17% relative error, widening with precision -- the
one component of the project whose gain is a physically quantum
mechanism (interference). The noise sweep prices it: the slope
survives per-round fidelity 0.999, bends at 0.99, inverts below
classical at 0.95; and the wall-clock parity ratio R* (classical
samples per quantum query at equal precision) is ~7 at 1% precision,
~45 at 0.2% -- the advantage is real in query scaling, not yet in
seconds.

- `run_qae.py` -- tilted state prep (verified), payoff encoding,
  MLE-QAE ladder, slope measurement
- `noise_sweep.py` -- depolarizing-Grover noise model + the parity table
- `fig5_noise_sweep.png`

## Scope of claims

Wins are against the classical product/tilt family (what practical
classical methods actually search), never "classical" per se -- the
exploited symmetry is classically imitable. All exact validation is at
<= 2^12 states; scaling claims are not licensed (training uses exact
statevector gradients, a crutch that dies near n=20). QAE numbers are
idealized query counts under a noiseless oracle; no wall-clock
advantage is claimed. All models are stylized surrogates motivated by
GPU reliability, not calibrated models of it.

## Reproducing

`pip install -r requirements.txt` (PennyLane >= 0.45 required -- older
versions have a breaking API difference). Each directory is
self-contained; run order is in the script headers. Exact ground
truths are computed by enumeration; every sampling experiment reports
bias, variance, VRF, MSE, and ESS over >= 20 seeded trials. No GPU
required; everything runs in minutes on a laptop except
`train_symmetric.py` (~20 min CPU).
