
import torch
import torch.nn as nn

# ==================================================
# Model Building: Parameters (Weights and Biases)
# ==================================================

# Parameters: Learnable values inside neural networks.
# Purpose: Help models learn patterns during training.
#
# Weight: Controls how strongly inputs affect outputs.
# Bias: An extra value added to adjust the output.
#
# model.parameters(): Returns learnable model parameters.
# model.named_parameters(): Returns parameter names and values.


# ==================================================
# 1. Understand Weights and Biases
# ==================================================

# Create a Linear layer with 1 input and 1 output.
linear = nn.Linear(1, 1)

# Set example weight = 10 and bias = 20.
with torch.no_grad():
    linear.weight.fill_(10)
    linear.bias.fill_(20)

# Student studied for 5 hours.
hours = torch.tensor([[5.0]])

# Calculate prediction: (5 * 10) + 20 = 70.
prediction = linear(hours)

print("=== Weights and Biases ===")
print("Weight:", linear.weight.item())
print("Bias:", linear.bias.item())
print("Prediction:", prediction.item())  # 70


# ==================================================
# 2. Access Model Parameters
# ==================================================

# Create a Linear layer with 3 input and 2 output features.
model = nn.Linear(3, 2)

print("\n=== Model Parameters ===")

# Display the model structure.
print("Model:", model)

# Display the parameter iterator object.
print("Named Parameters:", model.named_parameters())

# Display each parameter's name, value, and shape.
for parameter_name, parameter_value in model.named_parameters():
    print("Name:", parameter_name)
    print("Value:", parameter_value)
    print("Shape:", parameter_value.shape)
    print()

# Count all learnable parameters.
total = sum(param.numel() for param in model.parameters())

print("Total Parameters:", total)  # 8
