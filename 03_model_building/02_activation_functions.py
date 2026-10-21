
import torch
import torch.nn as nn

# ==================================================
# Model Building: Activation Functions
# ==================================================

# Activation Functions: Transform values inside neural networks.
# Purpose: Help neural networks learn complex patterns.
#
# nn.ReLU(): Changes negative values to 0.
# Purpose: Commonly used in hidden layers.
#
# nn.Sigmoid(): Converts values to between 0 and 1.
# Purpose: Useful for binary classification.
#
# nn.Softmax(): Converts scores into probabilities that sum to 1.
# Purpose: Useful for multiclass classification.


# ==================================================
# 1. nn.ReLU()
# ==================================================

# Create input values.
x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])

# Apply ReLU.
relu = nn.ReLU()
result = relu(x)

print("=== nn.ReLU ===")
print("Input:", x)
print("Output:", result)  # [0, 0, 0, 2, 5]


# ==================================================
# 2. nn.Sigmoid()
# ==================================================

# Create input values.
x = torch.tensor([-2.0, 0.0, 2.0])

# Apply Sigmoid.
sigmoid = nn.Sigmoid()
result = sigmoid(x)

print("\n=== nn.Sigmoid ===")
print("Input:", x)
print("Output:", result)  # [0.1192, 0.5000, 0.8808]


# ==================================================
# 3. nn.Softmax()
# ==================================================

# Example scores for 3 skin lesion classes.
scores = torch.tensor([[2.0, 1.0, 0.1]])

# Apply Softmax across the 3 class scores.
softmax = nn.Softmax(dim=1)
probabilities = softmax(scores)

print("\n=== nn.Softmax ===")
print("Scores:", scores)
print("Probabilities:", probabilities)  # [0.6590, 0.2424, 0.0986]
print("Total:", probabilities.sum().item())  # Approximately 1.0
