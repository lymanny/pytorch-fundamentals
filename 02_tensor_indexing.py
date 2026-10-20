
import torch

# ==================================================
# Lesson 01: Tensor Indexing and Slicing
# ==================================================

# Indexing:
# Access specific elements in a tensor.
#
# Slicing:
# Select a range of elements from a tensor.
#
# NOTE: Python indexing starts at 0.


# ==================================================
# 1. Integer Indexing
# ==================================================

# Select a single value using its index.

print("=== 1. Integer Indexing ===")

x = torch.tensor([10, 20, 30, 40, 50])

print("Tensor:", x)
print("Index 0:", x[0])    # First element: 10
print("Index 2:", x[2])    # Third element: 30
print("Index -1:", x[-1])  # Last element: 50


# ==================================================
# 2. List Indexing
# ==================================================

# Select multiple values using a list of indices.

print("\n=== 2. List Indexing ===")

x = torch.tensor([10, 20, 30, 40, 50])

print("Tensor:", x)
print("Indices [0, 2, 4]:", x[[0, 2, 4]])
print("Indices [1, 3]:", x[[1, 3]])


# ==================================================
# 3. Boolean Indexing
# ==================================================

# Select elements that satisfy a condition.
# True = Select the element.
# False = Do not select the element.

print("\n=== 3. Boolean Indexing ===")

x = torch.tensor([10, 20, 30, 40, 50])

print("Tensor:", x)
print("Greater than 20:", x[x > 20])
print("Less than 40:", x[x < 40])
print("Equal to 30:", x[x == 30])


# ==================================================
# 4. Tensor Indexing
# ==================================================

# Select multiple elements using a tensor of indices.

print("\n=== 4. Tensor Indexing ===")

x = torch.tensor([10, 20, 30, 40, 50])

indices = torch.tensor([0, 2, 4])

print("Tensor:", x)
print("Indices:", indices)
print("Selected:", x[indices])


# ==================================================
# 5. Tensor Slicing
# ==================================================

# Syntax: tensor[start:stop:step]
#
# start = Starting index (included) ✅
# stop  = Ending index (excluded) ❌
# step  = Number of positions to move

print("\n=== 5. Tensor Slicing ===")

x = torch.tensor([10, 20, 30, 40, 50])

print("Tensor:", x)
print("x[1:4]:", x[1:4])    # 20, 30, 40
print("x[:3]:", x[:3])      # First 3 elements
print("x[2:]:", x[2:])      # From index 2 to end
print("x[::2]:", x[::2])    # starts with the first element (index 0), then selects every second element.


# ==================================================
# 6. 2D Tensor Indexing and Slicing
# ==================================================

# Select values, rows, and columns from a matrix.

print("\n=== 6. 2D Indexing and Slicing ===")

matrix = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("Matrix:\n", matrix)

print("First Row:", matrix[0])
print("Second Row:", matrix[1])

# Syntax: matrix[row, column]
print("Row 0, Column 1:", matrix[0, 1])  # 20
print("Row 2, Column 2:", matrix[2, 2])  # 90

# ':' means select all elements along that axis.
print("First Column:", matrix[:, 0])
print("Second Column:", matrix[:, 1])

# Select first two rows and first two columns.
print("Top-left 2x2:\n", matrix[:2, :2])


# ==================================================
# 7. Summary
# ==================================================

print("\n=== Lesson Summary ===")
print("Integer Indexing: Select one element")
print("List Indexing: Select multiple elements")
print("Boolean Indexing: Select by condition")
print("Tensor Indexing: Select using index tensors")
print("Slicing: Select a range of elements")
print("2D Indexing: Select rows and columns")
