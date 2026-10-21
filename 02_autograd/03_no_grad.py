
import torch

# ==================================================
# Autograd: torch.no_grad()
# ==================================================

# torch.no_grad(): Temporarily disables gradient tracking.
#
# Purpose: Save memory and computation when gradients are not needed, such as during model prediction.
#
# Training: Needs gradients to help the model learn.
# Prediction: Usually does not need gradients because the trained model only produces results.
#
# Note: Gradient tracking resumes outside the block.


# ==================================================
# 1. Create a Tensor
# ==================================================

# Enable gradient tracking.
x = torch.tensor(3.0, requires_grad=True)

print("Original:", x.requires_grad)  # True


# ==================================================
# 2. Without torch.no_grad() - Training
# ==================================================

# Training: Track gradients to help the model learn.
y = x * x

print("\nTraining:")
print("Value:", y.item())             # 9.0
print("Requires Grad:", y.requires_grad)  # True


# ==================================================
# 3. With torch.no_grad() - Prediction
# ==================================================

# Prediction: No need to track gradients.
with torch.no_grad():
    z = x * x

    print("\nPrediction:")
    print("Value:", z.item())             # 9.0
    print("Requires Grad:", z.requires_grad)  # False


# ==================================================
# 4. Outside torch.no_grad()
# ==================================================

# Gradient tracking resumes outside the block.
a = x * 2

print("\nOutside Block:")
print("Value:", a.item())             # 6.0
print("Requires Grad:", a.requires_grad)  # True

# Original tensor remains unchanged.
print("Original:", x.requires_grad)  # True
