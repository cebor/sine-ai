"""Training utilities for the sine approximation model."""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from . import config


def train_model(model, x_train, y_train, epochs=None, lr=None):
    """Trains the model.
    
    Args:
        model: The neural network model to train
        x_train: Training input data
        y_train: Training target data
        epochs: Number of training epochs (default from config)
        lr: Learning rate (default from config)
        
    Returns:
        Trained model
    """
    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        # Forward pass
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return model


def save_model(model, filepath=None):
    """Saves the model weights to disk.
    
    Args:
        model: The model to save
        filepath: Path to save the model (default from config)
    """
    if filepath is None:
        filepath = config.MODEL_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def evaluate_and_plot(model, x_test, y_test, save_plot=True):
    """Evaluates the model and displays the results.
    
    Args:
        model: The trained model
        x_test: Test input data
        y_test: Test target data
        save_plot: Whether to save the plot to disk
    """
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    
    # Calculate MSE
    mse = nn.MSELoss()(predictions, y_test)
    print(f'\nFinal MSE: {mse.item():.6f}')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), y_test.numpy(), label='Actual Sine Function', linewidth=2)
    plt.plot(x_test.numpy(), predictions.numpy(), label='NN Prediction', linestyle='--', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Sine Approximation with Neural Network')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig(config.PLOT_FILENAME, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f'Plot saved as {config.PLOT_FILENAME}')
    
    plt.show()
