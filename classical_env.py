# classical_env.py
import numpy as np
import time
import multiprocessing as mp
import sys

def evaluate_system_failure(seed_and_samples):
    """ Runs a batch of Monte Carlo samples to look for a rare failure. """
    seed, num_samples = seed_and_samples
    np.random.seed(seed)
    
    # 10 independent random load variables
    num_variables = 10
    loads = np.random.exponential(scale=1.0, size=(num_samples, num_variables))
    
    # System stresses out and fails if cumulative square loads surpass a threshold
    threshold = 65.0
    system_stress = np.sum(loads**2, axis=1)
    failures = np.sum(system_stress > threshold)
    
    return failures

if __name__ == "__main__":
    TOTAL_SAMPLES = 10_000_000
    
    # Check how many CPU cores Slurm allocated to us (default to 4 for local tests)
    num_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    samples_per_core = TOTAL_SAMPLES // num_cores
    
    print(f"[HPC] Distributing {TOTAL_SAMPLES:,} samples across {num_cores} parallel CPU cores...")
    start_time = time.time()
    
    # Create distinct tasks for each core
    tasks = [(core_id, samples_per_core) for core_id in range(num_cores)]
    
    # Run the parallel processing worker pool
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(evaluate_system_failure, tasks)
        
    total_failures = sum(results)
    true_probability = total_failures / TOTAL_SAMPLES
    elapsed_time = time.time() - start_time
    
    print("\n================== NCSA RUN RESULTS ==================")
    print(f"Execution Time   : {elapsed_time:.2f} seconds")
    print(f"Total Failures   : {total_failures:,}")
    print(f"True Probability : {true_probability:.8e}")
    print("======================================================")
    
    np.save("ground_truth_baseline.npy", true_probability)