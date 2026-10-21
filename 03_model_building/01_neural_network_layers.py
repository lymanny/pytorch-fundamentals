
import torch
import torch.nn as nn

# ==================================================
# Model Building: nn.Linear, nn.Conv2d, nn.Sequential
# ==================================================

# nn.Linear(): Creates a fully connected neural network layer.
# Purpose: Transform input features into output features.
#
# nn.Conv2d(): Creates a 2D convolutional layer.
# Purpose: Extract features such as edges and textures from images.
#
# nn.Sequential(): Combines multiple layers in order.
# Purpose:  runs layers in order.


# ==================================================
# 1. nn.Linear()
# ==================================================

# Each student has 3 features: Study Hours, Sleep Hours, Attendance.
students = torch.tensor([
    [5.0, 8.0, 9.0],  # Student 1
    [7.0, 6.0, 8.0],  # Student 2
    [4.0, 7.0, 6.0],  # Student 3
    [8.0, 8.0, 9.0],  # Student 4
    [6.0, 7.0, 8.0]   # Student 5
])

# Transform 3 input features into 2 output features.
linear = nn.Linear(3, 2)

# Pass student data through the Linear layer.
result = linear(students)

print("=== nn.Linear ===")
print("Input Data:\n", students)
print("Input Shape:", students.shape)    # (5, 3): 5 students, 3 input features
print("Output Data:\n", result)
print("Output Shape:", result.shape)    # (5, 2): 5 students, 2 output features


# ==================================================
# 2. nn.Conv2d()
# ==================================================

# Create a convolutional layer with 3 input channels, 2 filters, and a 3x3 kernel.
conv = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)

# Create 1 simulated RGB image with size 5x5 (B, C, H, W).
image = torch.rand(1, 3, 5, 5)

# Pass the image through the convolutional layer.
result = conv(image)

# Output height and width: 5 - 3 + 1 = 3 (stride=1, padding=0).
print("\n=== nn.Conv2d ===")
print("Input Shape:", image.shape)     # (1, 3, 5, 5): 1 image, 3 RGB channels, 5x5 pixels
print("Output Shape:", result.shape)   # (1, 2, 3, 3): 1 image, 2 feature maps, each 3x3


# ==================================================
# 3. nn.Sequential()
# ==================================================

# Combine multiple neural network layers in order.
model = nn.Sequential(
    nn.Linear(3, 4),  # Transform 3 input features into 4 features.
    nn.ReLU(),         # Change negative values to 0.
    nn.Linear(4, 2)   # Transform 4 features into 2 output features.
)

# Pass 5 students through each layer to understand the shapes.
x = students

print("\n=== nn.Sequential ===")
print("Input Shape:", x.shape)  # (5, 3): 5 students, 3 features

x = model[0](x)
print("After Linear 1:", x.shape)  # (5, 4): 5 students, 4 features

x = model[1](x)
print("After ReLU:", x.shape)      # (5, 4): Shape stays the same

x = model[2](x)
print("After Linear 2:", x.shape)  # (5, 2): 5 students, 2 features

# Sequential can also run all layers automatically.
output = model(students)

print("Final Output Shape:", output.shape)  # (5, 2)
print("Model:\n", model)
