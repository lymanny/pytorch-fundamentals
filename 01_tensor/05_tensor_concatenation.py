import torch

# ==================================================
# torch.cat() vs torch.stack()
# ==================================================

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 1. torch.cat(): Combine tensors without adding a new dimension.
# 1D + 1D -> 1D (6 elements)
c = torch.cat([a, b], dim=0)

print("cat Result:", c)       # [1, 2, 3, 4, 5, 6]
print("cat Shape:", c.shape)  # (6,)


# 2. torch.stack(): Combine tensors by adding a new dimension.
# 1D + 1D -> 2D (2 rows, 3 columns)
d = torch.stack([a, b], dim=0)

print("stack Result:\n", d)   # [[1, 2, 3], [4, 5, 6]]
print("stack Shape:", d.shape)  # (2, 3)