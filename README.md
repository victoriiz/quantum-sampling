Research workspace for rare-event simulation using a variational quantum
circuit as an importance-sampling proposal distribution, tuned with a
classical Adam optimizer (VQIS). Modeled on a synthetic structural-reliability
benchmark: N=10 i.i.d. load variables x_i ~ Exp(1), failure defined as
sum(x_i^2) > tau (tau=65.0), currently classified via a Hamming-weight
boundary of k >= 4 (848 of 1024 discrete basis states).
 
## Layout
 
```
scripts/
  classical_env.py       Phase 1: classical MC ground truth
  quantum_ansatz.py       Phase 2: defines the variational quantum circuit and failure mask
  quantum_optimization.py Phase 3: train the HEA as an importance sampler
  quantum_sampler.py       Phase 4: statevector sampling + weight clipping
  run_benchmarks.py        Porter-Thomas / clipping-sweep / squeezing figures experiments
  plot_domain_mapping.py   Continuous-vs-discrete illustration figure
  plot_mask_classification.py  Mask classification illustration figure
 
slurm/                 Cluster job scripts (Delta-AI / SLURM)
notebooks/analysis.ipynb  Exploratory analysis and comparison plots, miscellaneous figures
outputs/                Generated .npy/.json (gitignored, regenerate via scripts/)
figures/                Generated + final figures used in the paper/poster
```
