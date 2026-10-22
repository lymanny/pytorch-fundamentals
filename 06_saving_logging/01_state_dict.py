
import torch
import torch.nn as nn

# ==================================================
# Saving and Logging: state_dict
# ==================================================

# state_dict(): Returns model parameters and persistent buffers as a dictionary.
# Purpose: Access model parameters for saving and loading.
#
# torch.save(): Saves model parameters to a file.
# torch.load(): Loads saved parameters from a file.
# load_state_dict(): Restores saved parameters into a model.
# .pth: Common file extension for PyTorch models and checkpoints.


# ==================================================
# 1. Create Model and Data
# ==================================================

# Create a Linear model with 1 input and 1 output.
model = nn.Linear(1, 1)

# Set starting weight and bias for predictable results.
with torch.no_grad():
    model.weight.fill_(10.0)
    model.bias.fill_(10.0)

# A student studied 5 hours.
hours = torch.tensor([[5.0]])


# ==================================================
# 2. View Model Parameters
# ==================================================

# Get the model's state dictionary.
parameters = model.state_dict()

# Display the model's weights and bias.
print("=== Model Parameters ===")
for name, value in parameters.items():
    print(name, ":", value)


# ==================================================
# 3. Save Model Parameters
# ==================================================

# Save model parameters to a .pth file.
torch.save(model.state_dict(), "student_model.pth")

print("\nModel parameters saved successfully!")


# ==================================================
# 4. Load Model Parameters
# ==================================================

# Create a new model with the same architecture.
new_model = nn.Linear(1, 1)

# Load the saved parameters from the file.
saved_parameters = torch.load(
    "student_model.pth",
    map_location="cpu",  # Load saved tensors onto the CPU.
    weights_only=True    # Restrict loading to safer supported data types.
)

# Restore the saved weights and bias into the new model.
new_model.load_state_dict(saved_parameters)

# Set the new model to evaluation mode.
new_model.eval()


# ==================================================
# 5. Compare Model Predictions
# ==================================================

# Disable gradient tracking during prediction.
with torch.no_grad():

    # Make a prediction using the original model.
    original_prediction = model(hours)

    # Make a prediction using the loaded model.
    loaded_prediction = new_model(hours)

# Display both predictions.
print("\n=== Model Predictions ===")
print("Original Model:", round(original_prediction.item(), 2))
print("Loaded Model:", round(loaded_prediction.item(), 2))
