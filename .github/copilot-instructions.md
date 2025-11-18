# Sinus AI - Copilot Instructions

## Project Overview
A PyTorch-based neural network that learns to approximate the sine function. The project demonstrates a clean ML workflow with modular architecture for training, evaluation, and inference.

## Architecture & Module Responsibilities

- **`sinus_ai/model.py`**: Defines `SineNet` - a 4-layer feedforward network (1→64→64→32→1) with ReLU activations
- **`sinus_ai/data.py`**: Generates sine function training data as PyTorch tensors
- **`sinus_ai/train.py`**: Training loop with MSE loss + Adam optimizer; model persistence; evaluation with matplotlib visualization
- **`sinus_ai/inference.py`**: Model loading and prediction interface (single value + interactive mode)
- **`sinus_ai/config.py`**: Central configuration (hyperparameters, paths, constants)
- **`main.py`**: CLI entry point with interactive menu and argument-based mode selection

## Key Patterns & Conventions

**Configuration Management**: All hyperparameters, paths, and constants live in `config.py`. Import and use these values rather than hardcoding - e.g., `config.EPOCHS`, `config.MODEL_PATH`, `config.HIDDEN_SIZE_1`.

**Model Persistence**: Models are saved as state dicts to `models/sine_model.pth`. Use `torch.save(model.state_dict(), filepath)` to save and `model.load_state_dict(torch.load(filepath, weights_only=True))` to load.

**Data Format**: All data is PyTorch tensors with shape `(-1, 1)`. Generate with `generate_training_data(n_samples)` which returns `(x_tensor, y_tensor)` tuples.

**Import Style**: Package exports are managed through `__init__.py`. Import from package root: `from sinus_ai import SineNet, train_model, load_model`.

## Developer Workflows

**Dependencies**: Install with `uv sync` (uses `pyproject.toml`). Python 3.14+ required.

**Running the CLI**:
- Interactive menu: `python main.py`
- Direct mode: `python main.py train|evaluate|predict`

**Jupyter Notebook**: `sine_approximation.ipynb` contains the exploratory prototype with identical architecture and training approach. It includes additional visualizations (training loss curves, error analysis) not in the CLI version.

**Dev Tooling**: `nbstripout` configured in dev dependencies to prevent committing notebook outputs.

## Testing & Validation

Model quality is evaluated using:
- MSE loss on test data (generated with `generate_training_data`)
- Visual comparison plots (actual vs predicted)
- Interactive predictions show per-sample error: `abs(prediction - actual)`

No formal test suite exists. Validation is visual/interactive through evaluation mode.

## Common Tasks

**Adding a hyperparameter**: Add to `config.py`, then update usage sites. Model architecture params should default to config values in `SineNet.__init__()`.

**Modifying model architecture**: Update `SineNet` in both `model.py` and the notebook to keep them synchronized.

**Changing training behavior**: Edit `train_model()` in `train.py`. Progress logging happens every 500 epochs by default.

**New inference mode**: Add to `inference.py` and expose via a new mode function in `main.py` with corresponding menu option.
