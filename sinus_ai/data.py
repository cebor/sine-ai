"""Data generation utilities for training and testing."""

import numpy as np
import torch

from . import config


def generate_training_data(
    n_samples: int = config.N_SAMPLES_TRAIN, x_min: float = config.X_MIN, x_max: float = config.X_MAX
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates training and test data for the sine function.

    Args:
        n_samples: Number of samples to generate (default from config)
        x_min: Minimum x value (default from config)
        x_max: Maximum x value (default from config)

    Returns:
        Tuple of (x_tensor, y_tensor) as PyTorch tensors
    """

    x = np.linspace(x_min, x_max, n_samples)
    y = np.sin(x)

    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).reshape(-1, 1).to(config.DEVICE)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(config.DEVICE)

    return x_tensor, y_tensor
