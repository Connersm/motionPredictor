#!/usr/bin/env python3
"""
Train supervised LSTM model for motion prediction.

This script:
1. Creates prediction data for cells that don't have it
2. Loads actual and predicted motion data from PostgreSQL database
3. Cleans data (removes NaNs, outliers, duplicates)
4. Creates input sequences from historical motion data
5. Trains LSTM model to predict next motion state
6. Saves trained model weights
7. Logs all retraining events to a timestamped log file

Usage:
    python train_supervised.py [--input-steps 20] [--epochs 50] [--batch-size 32] [--data-limit 5000] [--clear-db]
    
    --clear-db: Clear all predictions from database before training (for fresh start)
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from learning.model import (
    train_path_model, 
    create_prediction_data_for_cells,
    clear_database_predictions,
    count_database_entries,
    log_retrain_event
)


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised LSTM model for motion prediction"
    )
    parser.add_argument(
        "--input-steps",
        type=int,
        default=20,
        help="Number of historical timesteps to use as input (default: 20)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--data-limit",
        type=int,
        default=5000,
        help="Maximum number of rows to load from database (default: 5000)"
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear all prediction data from database before training"
    )
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=50.0,
        help="Distance threshold for detecting motion jumps (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SUPERVISED LSTM MOTION PREDICTION MODEL - TRAINING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input Steps:         {args.input_steps}")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Batch Size:          {args.batch_size}")
    print(f"  Learning Rate:       {args.learning_rate}")
    print(f"  Data Limit:          {args.data_limit}")
    print(f"  Position Threshold:  {args.position_threshold}")
    print(f"  Clear DB First:      {args.clear_db}")
    print("=" * 80)
    
    try:
        # Log the training session start
        log_retrain_event("\n" + "=" * 80)
        log_retrain_event("MANUAL TRAINING SESSION INITIATED")
        log_retrain_event(f"Configuration: input_steps={args.input_steps}, epochs={args.epochs}, "
                         f"batch_size={args.batch_size}, lr={args.learning_rate}")
        
        # Optionally clear database predictions
        if args.clear_db:
            print("\n[ACTION] Clearing all predictions from database...")
            log_retrain_event("User requested database clearing")
            clear_database_predictions()
            print("[INFO] Database predictions cleared")
        
        # Show current state
        db_count = count_database_entries()
        log_retrain_event(f"Database has {db_count} total entries")
        
        # Train the model
        print("\n[ACTION] Starting model training...")
        model, scaler = train_path_model(
            input_steps=args.input_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            data_limit=args.data_limit,
            auto_retrain=False,  # Manual training, don't auto-trigger
            detect_jumps=True,
            position_threshold=args.position_threshold
        )
        
        if model is not None:
            print("\n" + "=" * 80)
            print("✓ TRAINING COMPLETE")
            print("=" * 80)
            print(f"\nModel saved to: learning/path_model.pt")
            print(f"Log file:       learning/logs/retraining_log.txt")
            print(f"Ready for predictions!")
            log_retrain_event("Manual training session completed successfully")
        else:
            log_retrain_event("Training failed - model is None")
            print("\n[ERROR] Training failed - model is None")
            return 1
            
    except Exception as e:
        error_msg = f"Training failed with exception: {e}"
        print(f"\n[ERROR] {error_msg}")
        log_retrain_event(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        log_retrain_event(f"Traceback:\n{traceback.format_exc()}")
        return 1
    
    return 0



if __name__ == "__main__":
    sys.exit(main())
