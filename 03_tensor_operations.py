
import torch

# ==================================================
# Lesson 01: Basic Operations and Statistical Functions
# ==================================================

# Basic Operations:
# Perform mathematical calculations on tensors.
#
# Statistical Functions:
# Calculate values such as sum, mean, min, max,
# standard deviation, and variance.


# ==================================================
# 1. Basic Arithmetic Operations
# ==================================================

print("=== 1. Basic Arithmetic Operations ===")

a = torch.tensor([10, 20, 30], dtype=torch.float32)
b = torch.tensor([2, 4, 5], dtype=torch.float32)

print("Tensor A:", a)
print("Tensor B:", b)

# Element-wise operations: Calculate each matching pair.
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)


# ==================================================
# 2. Scalar Operations
# ==================================================

# Apply a single number to every element.

print("\n=== 2. Scalar Operations ===")

x = torch.tensor([10, 20, 30])

print("Original:", x)
print("Add 5:", x + 5)
print("Multiply by 2:", x * 2)
print("Subtract 3:", x - 3)


# ==================================================
# 3. Power and Square Root
# ==================================================

print("\n=== 3. Power and Square Root ===")

x = torch.tensor([4.0, 9.0, 16.0])

print("Original:", x)
print("Square:", x ** 2)
print("Square Root:", torch.sqrt(x))


# ==================================================
# 4. Statistical Functions
# ==================================================

print("\n=== 4. Statistical Functions ===")

x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

print("Tensor:", x)

print("Sum:", torch.sum(x))
print("Mean:", torch.mean(x))
print("Minimum:", torch.min(x))
print("Maximum:", torch.max(x))

# Variance: Measures how spread out values are from the mean.
# Mean = (10 + 20 + 30 + 40 + 50) / 5 = 30
# Squared differences = 400 + 100 + 0 + 100 + 400 = 1000
# Variance = 1000 / (5 - 1) = 250 (sample variance)
print("Variance:", torch.var(x))

# Standard Deviation: Measures data spread in the original units.
# It is calculated by taking the square root of variance.
# Standard Deviation = sqrt(250) = 15.8114
print("Standard Deviation:", torch.std(x))


# ==================================================
# 5. Find Minimum and Maximum Indices
# ==================================================

print("\n=== 5. Min and Max Indices ===")

x = torch.tensor([10, 50, 20, 80, 30])

# argmin(): Index of the smallest value
# argmax(): Index of the largest value

print("Tensor:", x)
print("Min Index:", torch.argmin(x))
print("Max Index:", torch.argmax(x))


# ==================================================
# 6. Statistical Functions on a 2D Tensor
# ==================================================

print("\n=== 6. Statistics on a 2D Tensor ===")

matrix = torch.tensor([
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0]
])

print("Matrix:\n", matrix)

# Calculate the sum of all elements.
print("Total Sum:", matrix.sum())

# Calculate the mean of all elements.
print("Total Mean:", matrix.mean())

# dim=0: Calculate down the rows (one result per column).
print("Column Sum:", matrix.sum(dim=0))

# dim=1: Calculate across columns (one result per row).
print("Row Sum:", matrix.sum(dim=1))


# ==================================================
# 7. Summary
# ==================================================

print("\n=== Lesson Summary ===")
print("Arithmetic: +, -, *, /")
print("Scalar: Calculate using a single number")
print("Power: **")
print("Square Root: torch.sqrt()")
print("Statistics: sum, mean, min, max, std, var")
print("Indices: argmin, argmax")
print("Dimensions: dim=0 (columns), dim=1 (rows)")
