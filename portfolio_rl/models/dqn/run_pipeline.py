"""
Complete pipeline script for DQN Reinforcement Learning Agent.

This script runs the complete pipeline for training and testing a DQN agent:
1. Trains a DQN agent on historical data
2. Tests the trained agent on test data
3. Saves and displays the results

Usage:
    python run_pipeline.py [--train_episodes EPISODES] [--test_steps STEPS] [--seed SEED]

Arguments:
    --train_episodes: Number of episodes to train for (default: 100)
    --test_steps: Maximum number of steps to run during testing (default: 252)
    --seed: Random seed for reproducibility (default: None)
"""

import os
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline(train_episodes=100, test_steps=252, seed=None):
    """
    Run the complete pipeline for training and testing a DQN agent.

    Args:
        train_episodes: Number of episodes to train for
        test_steps: Maximum number of steps to run during testing
        seed: Random seed for reproducibility
    """
    # Create timestamp for unique model and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"dqn_agent_{timestamp}.pth")

    print("=" * 80)
    print(
        f"Starting DQN Portfolio Optimization Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 80)

    # Step 1: Train the agent
    print("\n\n" + "=" * 30 + " TRAINING " + "=" * 30 + "\n")
    train_cmd = [
        "python",
        "train_dqn.py",
        "--episodes",
        str(train_episodes),
        "--save_path",
        model_path,
    ]

    if seed is not None:
        train_cmd.extend(["--seed", str(seed)])

    print(f"Running command: {' '.join(train_cmd)}")
    train_start = time.time()
    train_result = subprocess.run(
        train_cmd, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    train_end = time.time()

    if train_result.returncode != 0:
        print(f"Training failed with return code {train_result.returncode}")
        return

    print(f"\nTraining completed in {(train_end - train_start)/60:.2f} minutes")

    # Step 2: Test the agent
    print("\n\n" + "=" * 30 + " TESTING " + "=" * 30 + "\n")
    test_cmd = [
        "python",
        "test_dqn.py",
        "--model_path",
        model_path,
        "--steps",
        str(test_steps),
    ]

    if seed is not None:
        test_cmd.extend(["--seed", str(seed)])

    print(f"Running command: {' '.join(test_cmd)}")
    test_start = time.time()
    test_result = subprocess.run(
        test_cmd, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    test_end = time.time()

    if test_result.returncode != 0:
        print(f"Testing failed with return code {test_result.returncode}")
        return

    print(f"\nTesting completed in {(test_end - test_start)/60:.2f} minutes")

    # Step 3: Summary
    print("\n\n" + "=" * 30 + " SUMMARY " + "=" * 30 + "\n")
    print(
        f"Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Total time: {(test_end - train_start)/60:.2f} minutes")
    print(f"Trained model saved to: {model_path}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(
        "\nTo view the results, check the CSV files and PNG plots in the results directory."
    )
    print("=" * 80)


def main():
    """Main function to run the pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the complete DQN portfolio optimization pipeline"
    )
    parser.add_argument(
        "--train_episodes",
        type=int,
        default=100,
        help="Number of episodes to train for",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=252,
        help="Maximum number of steps to run during testing",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            train_episodes=args.train_episodes,
            test_steps=args.test_steps,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
