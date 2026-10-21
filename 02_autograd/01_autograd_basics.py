
import torch

# ==================================================
# Autograd: requires_grad, backward(), .grad
# ==================================================

# Autograd: Automatically calculates gradients.
#
# requires_grad=True: Enable gradient tracking.
# backward(): Calculate gradients.
# .grad: Store and access calculated gradients.
#
# Note: Autograd only calculates gradients.
# It does not update model weights.
# The optimizer uses gradients to update weights.


# ==================================================
# 1. Example: Simple Price Prediction
# ==================================================

# Input: Number of products.
x = torch.tensor(2.0)

# Actual total price.
actual = torch.tensor(10.0)

# Model weight: Estimated price per product.
# Enable gradient tracking for this weight.
weight = torch.tensor(3.0, requires_grad=True)


# ==================================================
# 2. Make a Prediction
# ==================================================

# Prediction = quantity * estimated price
# 2 * 3 = 6
prediction = x * weight

print("Prediction:", prediction.item())  # 6.0


# ==================================================
# 3. Calculate Loss (Error)
# ==================================================

# Loss measures how wrong the prediction is.
# Squared error = (prediction - actual)²
# (6 - 10)² = 16
loss = (prediction - actual) ** 2

print("Loss:", loss.item())  # 16.0


# ==================================================
# 4. Calculate Gradient (Autograd)
# ==================================================

# Autograd only calculates gradients.
# It does not update model weights.
loss.backward()

# Gradient = -16
# Negative gradient means increasing the weight
# slightly would reduce the loss.
print("Gradient:", weight.grad.item())  # -16.0


# ==================================================
# 5. Check Model Weight
# ==================================================

# The weight remains unchanged after backward().
# Updating the weight is the optimizer's job.
print("Weight:", weight.item())  # 3.0


# ==================================================
# 6. Summary
# ==================================================

print("\n=== Autograd Summary ===")
print("requires_grad=True: Enable gradient tracking")
print("backward(): Calculate gradients")
print(".grad: Store and access gradients")
print("Autograd: Does not update model weights")
