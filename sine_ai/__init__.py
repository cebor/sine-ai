"""Sine AI - Neural Network Sine Function Approximation"""

from .data import generate_training_data
from .inference import load_model, predict_interactive
from .model import SineNet
from .train import save_model, train_model, evaluate_and_plot

__version__ = "0.1.0"
__all__ = [
    "SineNet",
    "generate_training_data",
    "train_model",
    "save_model",
    "load_model",
    "evaluate_and_plot",
    "predict_interactive",
]
