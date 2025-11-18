"""Inference utilities for making predictions with trained models."""

import torch
from .model import SineNet
from . import config


def load_model(filepath=None):
    """Loads a trained model from disk.
    
    Args:
        filepath: Path to the saved model (default from config)
        
    Returns:
        Loaded model in evaluation mode
    """
    if filepath is None:
        filepath = config.MODEL_PATH
    
    model = SineNet()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    print(f'Model loaded from {filepath}')
    return model


def predict(model, x_value):
    """Makes a prediction for a single input value.
    
    Args:
        model: The trained model
        x_value: Input value (float or int)
        
    Returns:
        Predicted sine value
    """
    with torch.no_grad():
        x_tensor = torch.FloatTensor([[x_value]])
        prediction = model(x_tensor)
        return prediction.item()


def predict_interactive(model=None):
    """Interactive prediction loop.
    
    Loads the model if not provided and continuously prompts for input
    values, displaying predictions until the user exits.
    
    Args:
        model: Optional pre-loaded model (loads from disk if None)
    """
    if model is None:
        try:
            model = load_model()
        except FileNotFoundError:
            print(f"Error: Model file not found at {config.MODEL_PATH}")
            print("Please train the model first using the train mode.")
            return
    
    print("\n=== Interactive Sine Prediction ===")
    print("Enter a number to get the sine approximation.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("Enter x value: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting prediction mode.")
                break
            
            x_value = float(user_input)
            prediction = predict(model, x_value)
            actual = torch.sin(torch.tensor(x_value)).item()
            
            print(f"  Predicted: {prediction:.6f}")
            print(f"  Actual:    {actual:.6f}")
            print(f"  Error:     {abs(prediction - actual):.6f}\n")
            
        except ValueError:
            print("Invalid input. Please enter a valid number.\n")
        except KeyboardInterrupt:
            print("\n\nExiting prediction mode.")
            break
        except Exception as e:
            print(f"Error: {e}\n")
