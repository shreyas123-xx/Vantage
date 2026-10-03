"""
Trains the impact model. Supports two subcommands so simulation (the slow
part - it trains a real classifier and runs a cross-validated drift test per
scenario) can be run in safe, resumable chunks instead of one long-running
process:

    python3 train_impact_model.py simulate --count 150 --start 0
    python3 train_impact_model.py simulate --count 150 --start 150
    python3 train_impact_model.py simulate --count 80  --start 300
    python3 train_impact_model.py fit

Each `simulate` call appends to artifacts/training_scenarios.csv. `fit` reads
the accumulated CSV, trains the regressors, and saves artifacts/impact_models.joblib.

Or just run with no arguments for a single-shot run (fine on a machine
without a wall-clock limit on long-running processes):

    python3 train_impact_model.py
"""
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

from app.core.impact_model import generate_training_dataset, train_impact_models, save_models

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SCENARIOS_CSV = ARTIFACT_DIR / "training_scenarios.csv"


def cmd_simulate(count: int, start: int, seed: int):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Simulating scenarios [{start}, {start + count}) "
          f"(train real classifiers, measure real performance drop, compute real drift metrics)...")
    t0 = time.time()
    df = generate_training_dataset(n_scenarios=count, seed=seed + start, start_index=start, verbose=True)
    print(f"Chunk finished in {time.time() - t0:.1f}s.")

    if SCENARIOS_CSV.exists():
        existing = pd.read_csv(SCENARIOS_CSV)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(SCENARIOS_CSV, index=False)
    print(f"Total scenarios accumulated: {len(df)}  ->  {SCENARIOS_CSV}")


def cmd_fit():
    if not SCENARIOS_CSV.exists():
        print(f"No scenarios found at {SCENARIOS_CSV}. Run `simulate` first.")
        sys.exit(1)
    df = pd.read_csv(SCENARIOS_CSV)
    print(f"Fitting impact models on {len(df)} accumulated scenarios...")
    bundles = train_impact_models(df)
    out_path = save_models(bundles)

    print(f"\nSaved trained impact models to: {out_path}\n")
    print(f"{'metric':<10} {'R^2':>8} {'MAE':>8} {'n_train':>9} {'n_test':>8}")
    for metric, bundle in bundles.items():
        v = bundle.validation
        print(f"{metric:<10} {v['r2']:>8.3f} {v['mae']:>8.4f} {v['n_train']:>9d} {v['n_test']:>8d}")
        print(f"           mae by drift type: {v['mae_by_drift_type']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--count", type=int, default=320)
    p_sim.add_argument("--start", type=int, default=0)
    p_sim.add_argument("--seed", type=int, default=7)

    sub.add_parser("fit")

    args = parser.parse_args()

    if args.cmd == "simulate":
        cmd_simulate(args.count, args.start, args.seed)
    elif args.cmd == "fit":
        cmd_fit()
    else:
        # single-shot convenience path
        n_scenarios = int(sys.argv[1]) if len(sys.argv) > 1 else 320
        cmd_simulate(n_scenarios, 0, 7)
        cmd_fit()
