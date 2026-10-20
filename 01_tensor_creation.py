
import torch

# ==================================================
# Lesson 01: Tensor Creation, Shape, Dtype, Device
# ==================================================

# Tensor:
# A multidimensional array used to store and process
# numerical data in PyTorch.
#
# Similar to NumPy arrays, but also supports
# automatic differentiation and GPU acceleration.


# ==================================================
# 1. Tensor Creation
# ==================================================

print("=== 1. Tensor Creation ===")

# torch.tensor(): Create a tensor using values we provide
a = torch.tensor([2, 4, 6])

# torch.zeros(): Create tensors filled with zeros
b = torch.zeros(3)        # 1D: 3 zeros
c = torch.zeros(3, 4)     # 2D: 3 rows, 4 columns
d = torch.zeros(2, 3, 4)  # 3D: 2 matrices of 3 rows, 4 columns

# torch.ones(): Create tensors filled with ones
e = torch.ones(3)         # 1D: 3 ones
f = torch.ones(2, 3)      # 2D: 2 rows, 3 columns

# torch.rand(): Create random values between 0 and 1
g = torch.rand(3)         # 1D: 3 random values
h = torch.rand(2, 3)      # 2D: 2 rows, 3 columns

print("Given Values:", a)
print("Zeros 1D:", b)
print("Zeros 2D:\n", c)
print("Zeros 3D:\n", d)
print("Ones 1D:", e)
print("Ones 2D:\n", f)
print("Random 1D:", g)
print("Random 2D:\n", h)


# ==================================================
# 2. Tensor Shape and Dimensions
# ==================================================

print("\n=== 2. Tensor Shape ===")

# 0D: Scalar (single value)
scalar = torch.tensor(5)

# 1D: Vector (list of values)
vector = torch.tensor([1, 2, 3])

# 2D: Matrix (rows and columns)
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

# 3D: Two matrices of size 2x2
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

# Create a 4D tensor of zeros representing 10 RGB images, each 224×224 pixels (B, C, H, W).
images = torch.zeros(10, 3, 224, 224)

# shape = size along each dimension
# ndim = number of dimensions (axes)
# numel() = total number of elements

print("0D Scalar:", scalar.shape, "ndim:", scalar.ndim)
print("1D Vector:", vector.shape, "ndim:", vector.ndim)
print("2D Matrix:", matrix.shape, "ndim:", matrix.ndim)
print("3D Tensor:", tensor_3d.shape, "ndim:", tensor_3d.ndim)
print("4D Images:", images.shape, "ndim:", images.ndim)

print("Matrix Elements:", matrix.numel())


# ==================================================
# 3. Tensor Dtype (Data Type)
# ==================================================

print("\n=== 3. Tensor Dtype ===")

# int64: Whole numbers
int_tensor = torch.tensor([1, 2, 3])

# float32: Decimal numbers
float_tensor = torch.tensor([1.5, 2.5, 3.5])

# float64: Higher-precision decimal numbers
double_tensor = torch.tensor(
    [1.5, 2.5, 3.5],
    dtype=torch.float64
)

# bool: True or False
bool_tensor = torch.tensor([True, False, True])

print("int64:", int_tensor.dtype)
print("float32:", float_tensor.dtype)
print("float64:", double_tensor.dtype)
print("bool:", bool_tensor.dtype)


# ==================================================
# 4. Tensor Device
# ==================================================

print("\n=== 4. Tensor Device ===")

# Check whether an NVIDIA CUDA GPU is available
print("CUDA Available:", torch.cuda.is_available())

# Select GPU if available, otherwise CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create a tensor on CPU
x = torch.tensor([10, 20, 30])

print("Original Device:", x.device)

# Move the tensor to the selected device
x = x.to(device)

print("Selected Device:", device)
print("Tensor Device:", x.device)
print("Tensor:", x)


# ==================================================
# 5. Summary
# ==================================================

print("\n=== Lesson Summary ===")
print("Creation: Create numerical tensors")
print("Shape: Size of each dimension")
print("Dtype: Type of stored values")
print("Device: CPU or GPU")
