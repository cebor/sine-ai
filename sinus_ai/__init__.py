"""Sinus AI - Neural Network Sine Function Approximation"""

from .model import SineNet
from .data import generate_training_data
from .train import train_model, save_model
from .inference import load_model, predict_interactive

__version__ = "0.1.0"
__all__ = [
    "SineNet",
    "generate_training_data",
    "train_model",
    "save_model",
    "load_model",
    "predict_interactive",
]
