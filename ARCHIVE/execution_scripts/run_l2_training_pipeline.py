#!/usr/bin/env python3
"""
Train Model with L2 Features
Automated script to train model once L2 data collection is complete
"""

import time
import subprocess
from datetime import datetime


def wait_for_l2_completion():
    """Wait for L2 collection to complete"""
    print("⏳ Waiting for L2 collection to complete...")

    # Check if collection is still running
    while True:
        # Check if run_l2_collector.py process is still running
        try:
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                capture_output=True,
                text=True
            )
            if 'run_l2_collector.py' not in result.stdout:
                print("✅ L2 collection appears to be complete!")
                break
        except subprocess.SubprocessError:
            pass

        time.sleep(60)  # Check every minute


def run_training_pipeline():
    """Run the complete training pipeline with L2 features"""

    steps = [
        (
            "🚀 Training model with L2 and HHT features",
            "python train_model_robust.py --features all --trials 50"
        ),
    ]

    for step_name, command in steps:
        print(f"\n{step_name}...")
        print(f"Running: {command}")

        try:
            # The output will be streamed to the console in real-time
            subprocess.run(
                command.split(),
                check=True
            )
            print(f"\n✅ {step_name} completed successfully!")

        except subprocess.CalledProcessError as e:
            print(
                f"\n❌ Error in {step_name}: "
                f"Process exited with code {e.returncode}"
            )
            return False
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            return False

    return True


def main():
    """Main entry point"""
    print("="*60)
    print("L2 TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now()}")

    # Wait for L2 collection to complete
    wait_for_l2_completion()

    # Run training pipeline
    success = run_training_pipeline()

    if success:
        print("\n🎉 Training pipeline completed successfully!")
        print("Check the following files:")
        print("  - lgbm_model_l2_hht_*.txt (trained model)")
        print("  - prepared_data_l2_only_*.csv (features)")
    else:
        print("\n❌ Training pipeline failed. Check errors above.")


if __name__ == "__main__":
    main()
