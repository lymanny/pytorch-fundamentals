
import torch
import torch.nn as nn

# ==================================================
# Model Building: Subclassing nn.Module
# ==================================================

# nn.Module: Base class for creating neural networks.
# Purpose: Build custom neural network models.
#
# __init__(): Defines and initializes model layers.
# Purpose: Set up the model structure.
#
# super().__init__(): Initializes the parent nn.Module class.
# Purpose: Enable PyTorch model features.
#
# forward(): Defines how input data passes through layers.
# Purpose: Calculate the model output.


# ==================================================
# 1. Create a Custom Neural Network
# ==================================================

class StudentModel(nn.Module):

    # Define and initialize model layers.
    def __init__(self):
        super().__init__()

        print("1. Running __init__(): Creating model layers")

        self.fc1 = nn.Linear(3, 4)  # Transform 3 features into 4 features.
        self.relu = nn.ReLU()       # Change negative values to 0.
        self.fc2 = nn.Linear(4, 2)  # Transform 4 features into 2 features.

    # Define how input data passes through the layers.
    def forward(self, x):
        print("2. Running forward(): Processing input data")

        x = self.fc1(x)
        print("After Linear 1:", x.shape)  # (5, 4)

        x = self.relu(x)
        print("After ReLU:", x.shape)      # (5, 4)

        x = self.fc2(x)
        print("After Linear 2:", x.shape)  # (5, 2)

        return x


# ==================================================
# 2. Create Student Data
# ==================================================

# Each student has 3 features: Study Hours, Sleep Hours, Attendance.
students = torch.tensor([
    [5.0, 8.0, 9.0],  # Student 1
    [7.0, 6.0, 8.0],  # Student 2
    [4.0, 7.0, 6.0],  # Student 3
    [8.0, 8.0, 9.0],  # Student 4
    [6.0, 7.0, 8.0]   # Student 5
])


# ==================================================
# 3. Create and Run the Model
# ==================================================

# Create the model and automatically run __init__().
model = StudentModel()

# Pass student data to the model and automatically run forward().
output = model(students)

print("\n=== Subclassing nn.Module ===")
print("Input Shape:", students.shape)  # (5, 3): 5 students, 3 features
print("Output Shape:", output.shape)   # (5, 2): 5 students, 2 features
print("Output:\n", output)
print("Model:\n", model)
