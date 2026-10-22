
import torch
import torch.nn as nn

# ==================================================
# Training Loop: model.train() and model.eval()
# ==================================================

# model.train(): Sets the model to training mode.
# Purpose: Automatically enables training behavior such as Dropout.
#
# model.eval(): Sets the model to evaluation mode.
# Purpose: Automatically disables Dropout for consistent predictions.
#
# torch.no_grad(): Disables gradient tracking.
# Purpose: Save memory and computation during prediction.
#
# nn.Dropout(): Randomly sets some neuron outputs to 0 during training.
# Purpose: Help reduce overfitting.
#
# p=0.5: Each neuron output has a 50% chance of being turned off.


# ==================================================
# 1. Create a Simple Model
# ==================================================

# Create a model with a Dropout layer.
model = nn.Sequential(
    nn.Linear(3, 4),   # Transform 3 input features into 4 features.
    nn.ReLU(),          # Change negative values to 0.
    nn.Dropout(p=0.5),  # Randomly drop outputs during training.
    nn.Linear(4, 2)    # Transform 4 features into 2 output features.
)

# Create one student with 3 features.
student = torch.tensor([[5.0, 8.0, 9.0]])


# ==================================================
# 2. Training Mode
# ==================================================

# Set training mode and automatically activate Dropout.
model.train()

print("=== Training Mode ===")
print("Training:", model.training)  # True
print("Dropout Active:", model[2].training)  # True

# Run the same student data twice.
output1 = model(student)
output2 = model(student)

print("Output 1:", output1)
print("Output 2:", output2)


# ==================================================
# 3. Evaluation Mode
# ==================================================

# Set evaluation mode and automatically disable Dropout.
model.eval()

print("\n=== Evaluation Mode ===")
print("Training:", model.training)  # False
print("Dropout Active:", model[2].training)  # False

# Disable gradients because we are only making predictions.
with torch.no_grad():
    output1 = model(student)
    output2 = model(student)

print("Output 1:", output1)
print("Output 2:", output2)
