# Sine AI

A neural network that learns to approximate the sine function using PyTorch.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for model, data, training, and inference
- **Model Persistence**: Save and load trained model weights
- **Three Operation Modes**:
  - **Train**: Train a new model and save weights
  - **Evaluate**: Load trained model and visualize performance
  - **Predict**: Interactive mode for real-time predictions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sine-ai

# Install dependencies (using uv)
uv sync
```

## Usage

### Interactive Menu

Run without arguments to get an interactive menu:

```bash
uv run python main.py
```

This will display:
```
=== Sine Approximation with PyTorch ===

Select mode:
1. Train - Train a new model and save weights
2. Evaluate - Load and evaluate trained model
3. Predict - Interactive prediction mode
4. Exit

Enter choice (1-4):
```

### Command-Line Interface

Use command-line arguments for direct mode selection:

```bash
# Train a new model
uv run python main.py train

# Evaluate existing model
uv run python main.py evaluate

# Interactive prediction mode
uv run python main.py predict
```

**Alternatively**, activate the virtual environment:

```bash
source .venv/bin/activate
python main.py [train|evaluate|predict]
```

### Interactive Prediction Mode

In predict mode, you can continuously enter values and get predictions:

```
=== Interactive Sine Prediction ===
Enter a number to get the sine approximation.
Type 'quit' or 'exit' to stop.

Enter x value: 1.5
  Predicted: 0.997495
  Actual:    0.997495
  Error:     0.000000

Enter x value: 3.14159
  Predicted: 0.000026
  Actual:    0.000000
  Error:     0.000026

Enter x value: quit
Exiting prediction mode.
```

## Project Structure

```
sine-ai/
├── main.py                      # Entry point with CLI
├── sine_ai/                    # Main package
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration settings
│   ├── model.py                # SineNet architecture
│   ├── data.py                 # Data generation
│   ├── train.py                # Training and evaluation
│   └── inference.py            # Model loading and prediction
├── models/                      # Saved model weights (created automatically)
│   └── sine_model.pth
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## Configuration

Default hyperparameters can be modified in `sine_ai/config.py`:

- **Model Architecture**: Hidden layer sizes (64, 64, 32)
- **Training**: 5000 epochs, learning rate 0.001
- **Data**: 1000 training samples, 200 test samples
- **Model Path**: `models/sine_model.pth`

## Model Architecture

The `SineNet` model uses a simple feedforward architecture:

- Input: 1 neuron (x value)
- Hidden Layer 1: 64 neurons + ReLU
- Hidden Layer 2: 64 neurons + ReLU
- Hidden Layer 3: 32 neurons + ReLU
- Output: 1 neuron (sin(x) approximation)

## Dependencies

- Python >= 3.14
- PyTorch >= 2.9.1
- NumPy >= 2.3.5
- Matplotlib >= 3.10.7
- Jupyter >= 1.1.1 (for notebook experiments)

## Development

### Keeping Notebooks Clean in Git

This project uses `nbstripout` to prevent notebook outputs and metadata from cluttering git commits. It's already configured in the dev dependencies.

To set it up for your local repository:

```bash
# nbstripout is included in dev dependencies via uv sync
uv run nbstripout --install
```

This adds a git filter that automatically strips:
- Cell outputs
- Execution counts
- Metadata

The notebook (`sine_approximation.ipynb`) will remain functional with its outputs when you work locally, but commits will only include the code and markdown cells.

**Manual setup** (if not using the dev dependencies):

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

**Verify it's working:**

```bash
git diff --cached sine_approximation.ipynb
```

After making changes to the notebook, the diff should only show code/markdown changes, not execution metadata.

## License

This project is open source and available under the MIT License.
