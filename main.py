"""Main entry point for Sine AI - Neural Network Sine Function Approximation.

This script provides three modes of operation:
1. Train: Train a new model and save the weights
2. Evaluate: Load trained model and evaluate with visualization
3. Predict: Interactive mode for making predictions with trained model
"""

import argparse
import sys

from sinus_ai import (
    SineNet,
    generate_training_data,
    train_model,
    save_model,
    load_model,
    predict_interactive,
)
from sinus_ai.train import evaluate_and_plot
from sinus_ai.config import N_SAMPLES_TEST, DEVICE


def mode_train():
    """Train a new model and save the weights."""
    print("=== Training Mode ===\n")
    print(f"Using device: {DEVICE}\n")
    
    # Generate data
    print("Generating training data...")
    x_train, y_train = generate_training_data()
    
    # Initialize model
    print("Initializing model...")
    model = SineNet()
    print(f"Model architecture:\n{model}\n")
    
    # Training
    print("Starting training...")
    model = train_model(model, x_train, y_train)
    
    # Save model
    print("\nSaving model...")
    save_model(model)
    
    # Evaluation
    print("\nEvaluating model...")
    x_test, y_test = generate_training_data(n_samples=N_SAMPLES_TEST)
    evaluate_and_plot(model, x_test, y_test)
    
    print("\nTraining complete!")


def mode_evaluate():
    """Load trained model and evaluate with visualization."""
    print("=== Evaluation Mode ===\n")
    
    # Load model
    print("Loading trained model...")
    try:
        model = load_model()
    except FileNotFoundError:
        print("Error: No trained model found. Please train the model first.")
        return
    
    # Generate test data
    print("Generating test data...")
    x_test, y_test = generate_training_data(n_samples=N_SAMPLES_TEST)
    
    # Evaluate
    print("Evaluating model...")
    evaluate_and_plot(model, x_test, y_test)


def mode_predict():
    """Interactive prediction mode."""
    predict_interactive()


def main():
    parser = argparse.ArgumentParser(
        description="Sine AI - Neural Network Sine Function Approximation"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["train", "evaluate", "predict"],
        help="Mode of operation: train, evaluate, or predict",
    )
    
    args = parser.parse_args()
    
    # If no mode specified, show interactive menu
    if args.mode is None:
        print("=== Sine Approximation with PyTorch ===\n")
        print("Select mode:")
        print("1. Train - Train a new model and save weights")
        print("2. Evaluate - Load and evaluate trained model")
        print("3. Predict - Interactive prediction mode")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == "1":
                    mode_train()
                    break
                elif choice == "2":
                    mode_evaluate()
                    break
                elif choice == "3":
                    mode_predict()
                    break
                elif choice == "4":
                    print("Exiting.")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n\nExiting.")
                sys.exit(0)
    else:
        # Execute selected mode
        if args.mode == "train":
            mode_train()
        elif args.mode == "evaluate":
            mode_evaluate()
        elif args.mode == "predict":
            mode_predict()


if __name__ == "__main__":
    main()
