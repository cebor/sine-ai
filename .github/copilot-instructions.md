# Sine AI - Copilot Instructions

## Project Overview
A PyTorch-based neural network that learns to approximate the sine function. The project demonstrates clean ML workflow with modular architecture for training, evaluation, and inference. Python 3.14+, PyTorch 2.9+, managed via `uv`.

## Architecture & Module Responsibilities

- **`sine_ai/model.py`**: Defines `SineNet` - a 4-layer feedforward network (1→64→64→32→1) with ReLU activations. Model automatically moves to `config.DEVICE` (CUDA if available, else CPU) in `__init__`.
- **`sine_ai/data.py`**: Generates sine function training data as PyTorch tensors with shape `(-1, 1)`. Returns `(x_tensor, y_tensor)` tuples on `config.DEVICE`.
- **`sine_ai/train.py`**: Training loop (MSE loss + Adam optimizer), model persistence (`save_model`), evaluation with matplotlib visualization (`evaluate_and_plot`). Prints progress every 500 epochs.
- **`sine_ai/inference.py`**: Model loading (`load_model`) and prediction interface (`predict` for single values, `predict_interactive` for CLI loop).
- **`sine_ai/config.py`**: Central configuration for hyperparameters (EPOCHS=5000, LEARNING_RATE=0.001, hidden layer sizes), paths (MODEL_PATH="models/sine_model.pth"), data ranges (0 to 2π), and device selection.
- **`main.py`**: CLI entry point with two modes: interactive menu (no args) or direct mode selection (`train`, `evaluate`, `predict` as args).
- **`sine_approximation.ipynb`**: Exploratory prototype with identical architecture. Includes additional visualizations (training loss curves, error analysis) not in CLI version. Keep synchronized with `model.py`.

## Key Patterns & Conventions

**Configuration Management**: All hyperparameters, paths, and constants are in `config.py`. Always import and use these values: `config.EPOCHS`, `config.MODEL_PATH`, `config.HIDDEN_SIZE_1`, etc. Never hardcode values.

**Model Persistence**: Save models as state dicts: `torch.save(model.state_dict(), filepath)`. Load with `model.load_state_dict(torch.load(filepath, weights_only=True))`. Default path is `models/sine_model.pth` (auto-created on first save).

**Data Format**: All data are PyTorch tensors with shape `(-1, 1)` on `config.DEVICE`. Use `generate_training_data(n_samples)` which returns `(x_tensor, y_tensor)` tuples. Data range defaults to [0, 2π].

**Device Handling**: All tensors and models automatically move to `config.DEVICE`. Don't manually specify device in model or data creation - rely on config.

**Import Style**: Package exports are centralized in `__init__.py`. Import from package root: `from sine_ai import SineNet, train_model, load_model, generate_training_data, evaluate_and_plot, predict_interactive, save_model`.

**Training Output**: Model training prints loss every 500 epochs. Evaluation displays MSE, saves plot to `sine_approximation.png` (DPI=150), and shows matplotlib figure.

## Developer Workflows

**Setup**: `uv sync` installs all dependencies from `pyproject.toml`. Python 3.14+ required.

**Running the CLI**:
- Interactive menu: `python main.py` (displays 4 options: train, evaluate, predict, exit)
- Direct mode: `python main.py [train|evaluate|predict]`

**Training Flow**: Train mode generates data → initializes model → trains → saves to `models/sine_model.pth` → evaluates → displays plot.

**Evaluation Flow**: Evaluate mode loads existing model from `models/sine_model.pth` (errors if not found) → generates test data → calculates MSE → displays plot.

**Prediction Flow**: Interactive loop that loads model, prompts for x values, displays prediction + actual + error. Exit with "quit" or "exit".

**Jupyter Notebook**: `sine_approximation.ipynb` is the exploratory prototype. When modifying model architecture, update both `model.py` and the notebook to maintain consistency. Notebook includes visualizations not in CLI.

**Git Cleanliness**: `nbstripout` is in dev dependencies. Run `nbstripout --install` after `uv sync` to strip notebook outputs/metadata from commits automatically.

## Testing & Validation

No formal test suite. Validation is visual/interactive:
- MSE loss on test data (200 samples by default)
- Visual comparison plots (actual vs predicted sine curves)
- Interactive prediction mode shows per-sample error: `|prediction - actual|`

## Common Tasks

**Adding hyperparameter**: 1) Add to `config.py`, 2) Update usage sites (e.g., function signatures), 3) Ensure `SineNet.__init__()` defaults to config values.

**Modifying architecture**: 1) Update `SineNet` in `model.py`, 2) Update hidden size constants in `config.py`, 3) Sync changes to notebook, 4) Retrain model (old weights won't match new architecture).

**Changing training behavior**: Edit `train_model()` in `train.py`. Logging frequency controlled by `if (epoch + 1) % 500 == 0`.

**New inference mode**: 1) Add prediction function to `inference.py`, 2) Create mode function in `main.py` (e.g., `mode_batch_predict()`), 3) Add CLI arg to parser choices, 4) Add menu option to interactive loop.

**Adjusting data range**: Modify `X_MIN` and `X_MAX` in `config.py`. Current range is [0, 2π] which covers one complete sine period.
