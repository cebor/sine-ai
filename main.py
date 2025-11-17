import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class SineNet(nn.Module):
    """Simple neural network to approximate the sine function."""
    
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def generate_training_data(n_samples=1000):
    """Generates training and test data for the sine function."""
    x = np.linspace(0, 2 * np.pi, n_samples)
    y = np.sin(x)
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    return x_tensor, y_tensor


def train_model(model, x_train, y_train, epochs=5000, lr=0.001):
    """Trains the model."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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


def evaluate_and_plot(model, x_test, y_test):
    """Evaluates the model and displays the results."""
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
    plt.savefig('sine_approximation.png', dpi=150, bbox_inches='tight')
    print('Plot saved as sine_approximation.png')
    plt.show()


def main():
    print("=== Sine Approximation with PyTorch ===\n")
    
    # Generate data
    print("Generating training data...")
    x_train, y_train = generate_training_data(n_samples=1000)
    
    # Initialize model
    print("Initializing model...")
    model = SineNet()
    print(f"Model architecture:\n{model}\n")
    
    # Training
    print("Starting training...")
    model = train_model(model, x_train, y_train, epochs=5000, lr=0.001)
    
    # Evaluation
    print("\nEvaluating model...")
    x_test, y_test = generate_training_data(n_samples=200)
    evaluate_and_plot(model, x_test, y_test)


if __name__ == "__main__":
    main()
