import torch

# ==================================================
# Matrix Multiplication and Broadcasting
# ==================================================

# Matrix Multiplication:
# Multiplies rows by columns and adds the results.
#
# Broadcasting:
# Allows operations between tensors with
# different but compatible shapes.


# ==================================================
# 1. Element-wise Multiplication (*)
# ==================================================

a = torch.tensor([[1, 2],
                  [3, 4]])

b = torch.tensor([[5, 6],
                  [7, 8]])

# Multiply elements at the same position.
print("Element-wise:\n", a * b)

# Output:
# [[ 5, 12],
#  [21, 32]]


# ==================================================
# 2. Matrix Multiplication (@)
# ==================================================

# Multiply rows by columns.
result = a @ b

print("\nMatrix Multiplication:\n", result)

# Output:
# [[19, 22],
#  [43, 50]]

# Example: (1 * 5) + (2 * 7) = 19


# ==================================================
# 3. Matrix Multiplication Methods
# ==================================================

# These three methods give the same result for 2D tensors.
print("\nUsing @:\n", a @ b)
print("Using matmul():\n", torch.matmul(a, b))
print("Using mm():\n", torch.mm(a, b))

# @ and matmul(): Support matrix multiplication.
# mm(): Only supports 2D matrices.


# ==================================================
# 4. Matrix Multiplication Shapes
# ==================================================

x = torch.ones(2, 3)
y = torch.ones(3, 4)

# Inner dimensions must match.
# (2, 3) @ (3, 4) -> (2, 4)
result = x @ y

print("\nMatrix Shape:", result.shape)


# ==================================================
# 5. Broadcasting with a Number
# ==================================================

x = torch.tensor([10, 20, 30])

# Add 5 to every element.
print("\nBroadcasting Number:", x + 5)

# Output: [15, 25, 35]


# ==================================================
# 6. Broadcasting with a Vector
# ==================================================

matrix = torch.tensor([[10, 20, 30],
                       [40, 50, 60]])

vector = torch.tensor([1, 2, 3])

# Add the vector to every row.
result = matrix + vector

print("\nBroadcasting Vector:\n", result)

# Output:
# [[11, 22, 33],
#  [41, 52, 63]]


# ==================================================
# 7. Broadcasting with a Column
# ==================================================

column = torch.tensor([[100],
                       [200]])

# Add 100 to row 1 and 200 to row 2.
result = matrix + column

print("\nBroadcasting Column:\n", result)

# Output:
# [[110, 120, 130],
#  [240, 250, 260]]