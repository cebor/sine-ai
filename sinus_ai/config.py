"""Configuration settings for the Sine AI model."""

import os

# Model architecture
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32

# Training parameters
EPOCHS = 5000
LEARNING_RATE = 0.001
N_SAMPLES_TRAIN = 1000
N_SAMPLES_TEST = 200

# Model persistence
MODEL_DIR = "models"
MODEL_FILENAME = "sine_model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Plotting
PLOT_FILENAME = "sine_approximation.png"
PLOT_DPI = 150
