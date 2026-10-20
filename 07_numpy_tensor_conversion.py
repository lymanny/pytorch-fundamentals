
import numpy as np
import torch

# ==================================================
# Lesson 07: NumPy and Tensor Conversion
# ==================================================

# NumPy -> Tensor: Converts a NumPy array to a tensor.
# Tensor -> NumPy: Converts a tensor to a NumPy array.


# ==================================================
# 1. NumPy -> Tensor
# ==================================================

array = np.array([1, 2, 3])

# from_numpy(): Shares memory with the NumPy array.
a = torch.from_numpy(array)

# tensor(): Creates an independent copy.
b = torch.tensor(array)

print("NumPy:", array)
print("from_numpy:", a)
print("tensor:", b)

# Change the original NumPy array.
array[0] = 100

print("\nAfter changing NumPy:")
print("NumPy:", array)       # [100, 2, 3]
print("from_numpy:", a)      # [100, 2, 3]
print("tensor:", b)          # [1, 2, 3]


# ==================================================
# 2. Tensor -> NumPy
# ==================================================

tensor = torch.tensor([4, 5, 6])

# Convert tensor to NumPy array.
array = tensor.numpy()

print("\nPyTorch Tensor:", tensor)
print("NumPy Array:", array)


# ==================================================
# 3. Data Types
# ==================================================

array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(array)

print("\nNumPy Dtype:", array.dtype)   # float64
print("Tensor Dtype:", tensor.dtype)  # torch.float64


# ==================================================
# 4. Summary
# ==================================================

print("\n=== Lesson Summary ===")
print("from_numpy(): NumPy -> Tensor (shared memory)")
print("torch.tensor(): NumPy -> Tensor (copy)")
print("tensor.numpy(): Tensor -> NumPy")
