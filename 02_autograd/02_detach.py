
import torch

# ==================================================
# Autograd: detach()
# ==================================================

# detach(): Disconnects a tensor from the gradient graph.
# Purpose: Use tensor values without tracking gradients.
# Useful for displaying predictions and converting to NumPy.
# Note: The original tensor remains unchanged.


# ==================================================
# 1. Create a Tensor
# ==================================================

x = torch.tensor(3.0, requires_grad=True)

# Calculate y = 3 * 3 = 9.
y = x * x


# ==================================================
# 2. Detach the Tensor
# ==================================================

# Disconnect y from the gradient graph.
z = y.detach()

# Compare original and detached tensors.
print("Original Value:", y.item())           # 9.0
print("Original Grad:", y.requires_grad)      # True

print("Detached Value:", z.item())           # 9.0
print("Detached Grad:", z.requires_grad)      # False


# ==================================================
# 3. Convert to NumPy
# ==================================================

# y.numpy()  # ERROR: Cannot convert a tensor
#            # that requires gradients directly to NumPy.

# Detached tensor can be converted to NumPy.
array = z.numpy()

print("NumPy Array:", array)  # 9.0
