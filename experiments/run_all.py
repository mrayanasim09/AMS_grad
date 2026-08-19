"""
Run All Experiments: Complete end-to-end research reproduction and extension pipeline.
"""

import os
import sys
import subprocess
import time

def run_script(script_path):
    print("\n" + "=" * 80)
    print(f"RUNNING: {os.path.basename(script_path)}")
    print("=" * 80)
    start = time.time()
    res = subprocess.run([sys.executable, script_path], check=True)
    dur = time.time() - start
    print(f"COMPLETED in {dur:.2f}s (Exit code: {res.returncode})")

def main():
    exp_dir = os.path.dirname(__file__)
    
    # 1. Pilot calibration
    run_script(os.path.join(exp_dir, "00_pilot_calibration.py"))
    
    # 2. Phase 2 Toy benchmarks
    run_script(os.path.join(exp_dir, "01_toy_benchmarks.py"))
    
    # 3. Phase 3 Counterexample reproduction
    run_script(os.path.join(exp_dir, "02_counterexample_reproduction.py"))
    
    # 4. Phase 4 Phase boundary sweep
    run_script(os.path.join(exp_dir, "03_beta2_phase_boundary.py"))
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("All figures saved in: report/figures/")
    print("All data saved in:    results/")
    print("=" * 80)

if __name__ == "__main__":
    main()
