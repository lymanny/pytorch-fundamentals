
import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================
# Training Loop: Loss Functions and Optimizers
# ==================================================

# Loss Function: Measures the difference between predictions and actual values.
# Purpose: Calculate prediction errors.
#
# Optimizer: Updates model weights and biases using gradients.
# Purpose: Help the model reduce prediction errors.
#
# ==================================================
# Common Loss Functions
# ==================================================
#
# nn.MSELoss(): Used for regression, such as predicting exam scores.
# nn.BCEWithLogitsLoss(): Used for binary classification (2 classes).
# nn.CrossEntropyLoss(): Used for multiclass classification (3+ classes).
#
# ==================================================
# Common Optimizers
# ==================================================
#
# optim.SGD(): Updates weights using gradient descent.
# optim.Adam(): Adapts learning rates for individual parameters.
#
# ==================================================
# Important Training Functions
# ==================================================
#
# optimizer.zero_grad(): Clears previous gradients.
# loss.backward(): Calculates gradients using Autograd.
# optimizer.step(): Updates model weights and biases.


# ==================================================
# 1. Create a Model and Student Data
# ==================================================

# Create a Linear model with 1 input and 1 output.
model = nn.Linear(1, 1)

# Set starting weight = 10 and bias = 10.
with torch.no_grad():
    model.weight.fill_(10.0)
    model.bias.fill_(10.0)

# A student studied 5 hours and received a score of 70.
hours = torch.tensor([[5.0]])
actual_score = torch.tensor([[70.0]])


# ==================================================
# 2. Create Loss Function and Optimizer
# ==================================================

# Use MSELoss because we are predicting a numerical score.
loss_fn = nn.MSELoss()

# Use SGD to update the model's weights and bias.
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==================================================
# 3. Calculate Loss
# ==================================================

# Make a prediction using the model.
prediction = model(hours)

# Compare the prediction with the actual score.
loss = loss_fn(prediction, actual_score)

print("=== Before Training ===")
print("Prediction:", prediction.item())     # 60.0
print("Actual Score:", actual_score.item()) # 70.0
print("Loss:", loss.item())                 # 100.0


# ==================================================
# 4. Update Model Parameters
# ==================================================

# Clear previous gradients.
optimizer.zero_grad()

# Calculate gradients.
loss.backward()

# Update weights and bias using SGD.
optimizer.step()


# ==================================================
# 5. Check the Updated Prediction
# ==================================================

# Disable gradient tracking during prediction.
with torch.no_grad():
    new_prediction = model(hours)
    new_loss = loss_fn(new_prediction, actual_score)

print("\n=== After One Training Step ===")
print("New Prediction:", new_prediction.item())  # Approximately 65.2
print("New Loss:", new_loss.item())                # Approximately 23.04
