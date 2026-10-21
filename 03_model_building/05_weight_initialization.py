
import torch
import torch.nn as nn

# ==================================================
# Model Building: Weight Initialization
# ==================================================

# Weight Initialization: Sets starting weights before training.
# Purpose: Help neural networks learn effectively.
#
# Default Initialization: PyTorch automatically sets initial weights.
# Purpose: Useful for simple neural networks.
#
# Constant Initialization: Sets weights or biases to a fixed value.
# Purpose: Useful for setting specific values, such as zero biases.
#
# Xavier Initialization: Sets random weights based on layer sizes.
# Purpose: Often useful for networks with Tanh or Sigmoid.
#
# Kaiming Initialization: Sets random weights for ReLU-based networks.
# Purpose: Often useful for CNNs and deep neural networks.


# ==================================================
# 1. Default Initialization
# ==================================================

# Create a Linear layer with 3 inputs and 2 outputs.
layer = nn.Linear(3, 2)

# PyTorch automatically initializes weights and biases.
print("=== Default Initialization ===")
print("Weights:\n", layer.weight)
print("Bias:\n", layer.bias)


# ==================================================
# 2. Constant Initialization
# ==================================================

# Create a new Linear layer.
layer = nn.Linear(3, 2)

# Set all weights to 0.5 and biases to 0.
nn.init.constant_(layer.weight, 0.5)
nn.init.constant_(layer.bias, 0.0)

print("\n=== Constant Initialization ===")
print("Weights:\n", layer.weight)
print("Bias:\n", layer.bias)


# ==================================================
# 3. Xavier Initialization
# ==================================================

# Create a new Linear layer.
layer = nn.Linear(3, 2)

# Initialize weights using Xavier uniform.
nn.init.xavier_uniform_(layer.weight)

print("\n=== Xavier Initialization ===")
print("Weights:\n", layer.weight)


# ==================================================
# 4. Kaiming Initialization
# ==================================================

# Create a new Linear layer.
layer = nn.Linear(3, 2)

# Initialize weights using Kaiming normal for ReLU.
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

print("\n=== Kaiming Initialization ===")
print("Weights:\n", layer.weight)
