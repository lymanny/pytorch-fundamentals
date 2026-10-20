
import torch

# ==================================================
# Tensor Reshaping
# ==================================================

# Tensor Reshaping:
# Changes the shape or arrangement of tensor dimensions.
#
# 1. reshape(): Changes tensor shape while keeping the
#    same total number of elements.
#
# 2. view(): Changes tensor shape without copying data,
#    but requires a compatible memory layout.
#
# 3. squeeze(): Removes dimensions whose size is 1.
#    Example: (1, 3, 1, 4) -> (3, 4)
#
# 4. unsqueeze(): Adds one new dimension of size 1.
#    Example: (3,) -> (1, 3) or (3, 1)
#
# 5. permute(): Rearranges the order of dimensions.
#    Example: (H, W, C) -> (C, H, W)
#
# 6. transpose(): Swaps two dimensions of a tensor.
#    Example: (2, 3) -> (3, 2)


# ==================================================
# 1. reshape()
# ==================================================

print("=== 1. reshape() ===")

x = torch.tensor([1, 2, 3, 4, 5, 6])

print("Original:", x)
print("Original Shape:", x.shape)

# Change 1D into 2D: 2 rows and 3 columns.
a = x.reshape(2, 3)

print("Reshape (2, 3):\n", a)
print("New Shape:", a.shape)

# Change into 3 rows and 2 columns.
b = x.reshape(3, 2)

print("Reshape (3, 2):\n", b)

# -1 means PyTorch automatically calculates the size.
# Choose 2 rows; PyTorch calculates 3 columns.
c = x.reshape(2, -1)

print("Reshape (2, -1):\n", c)

# Choose 2 columns; PyTorch calculates 3 rows.
d = x.reshape(-1, 2)

print("Reshape (-1, 2):\n", d)


# ==================================================
# 2. view()
# ==================================================

print("\n=== 2. view() ===")

x = torch.tensor([1, 2, 3, 4, 5, 6])

# Another way to change tensor shape.
# Requires a compatible memory layout.
a = x.view(2, 3)

print("Original:", x)
print("View (2, 3):\n", a)
print("New Shape:", a.shape)


# ==================================================
# 3. squeeze()
# ==================================================

print("\n=== 3. squeeze() ===")

# Remove dimensions whose size is 1.
# It does not remove actual data values.
x = torch.zeros(1, 3, 1, 4)

print("Original Shape:", x.shape)

# Remove all dimensions of size 1.
# (1, 3, 1, 4) -> (3, 4)
a = x.squeeze()

print("After squeeze:", a.shape)

# Remove only dimension 0.
# (1, 3, 1, 4) -> (3, 1, 4)
b = x.squeeze(dim=0)

print("Squeeze dim=0:", b.shape)

# Change 2D into 1D.
y = torch.tensor([[10, 20, 30]])

print("Before squeeze:", y.shape)
print("After squeeze:", y.squeeze().shape)


# ==================================================
# 4. unsqueeze()
# ==================================================

print("\n=== 4. unsqueeze() ===")

# Add one new dimension whose size is 1.
x = torch.tensor([10, 20, 30])

print("Original:", x)
print("Original Shape:", x.shape)

# Add dimension at position 0.
# 1D -> 2D: 1 row and 3 columns.
a = x.unsqueeze(0)

print("Unsqueeze dim=0:\n", a)
print("Shape:", a.shape)

# Add dimension at position 1.
# 1D -> 2D: 3 rows and 1 column.
b = x.unsqueeze(1)

print("Unsqueeze dim=1:\n", b)
print("Shape:", b.shape)

# Add another dimension to a 2D tensor.
# 2D -> 3D: (1, 3) -> (1, 1, 3)
c = a.unsqueeze(0)

print("Unsqueeze 2D to 3D:", c.shape)


# ==================================================
# 5. permute()
# ==================================================

print("\n=== 5. permute() ===")

# Create a 3D tensor representing an RGB image.
# H = Height, W = Width, C = Color Channels.
image = torch.zeros(224, 224, 3)

print("Original Shape (H, W, C):", image.shape)

# Original dimensions:
# dim=0 -> Height (224)
# dim=1 -> Width (224)
# dim=2 -> Channels (3)

# Rearrange dimensions from (H, W, C) to (C, H, W).
image_chw = image.permute(2, 0, 1)

print("Permuted Shape (C, H, W):", image_chw.shape)


# ==================================================
# 6. transpose() - 2D Tensor
# ==================================================

print("\n=== 6. transpose() ===")

# For a 2D tensor:
# dim=0 -> Row axis
# dim=1 -> Column axis

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("Original:\n", x)
print("Original Shape:", x.shape)

# Swap rows and columns.
# transpose(0, 1) and transpose(1, 0) are equivalent.
a = x.transpose(0, 1)

print("Transposed:\n", a)
print("New Shape:", a.shape)


# ==================================================
# 7. Summary
# ==================================================

print("\n=== Lesson Summary ===")
print("reshape(): Change tensor shape")
print("view(): Change shape using shared memory")
print("squeeze(): Remove dimensions of size 1")
print("unsqueeze(): Add one dimension of size 1")
print("permute(): Rearrange dimension order")
print("transpose(): Swap two dimensions")
