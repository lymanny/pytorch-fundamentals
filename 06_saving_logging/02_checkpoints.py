
import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================
# Saving and Logging: Checkpoints and Saving Models
# ==================================================

# Saving Model: Stores model parameters for future use.
# Checkpoint: Stores model parameters and training progress.
# Purpose: Save and resume training without starting over.
#
# torch.save(): Saves model parameters or checkpoints to a file.
# torch.load(): Loads saved model parameters or checkpoints.
# state_dict(): Gets model or optimizer state.
# load_state_dict(): Restores saved model or optimizer state.


# ==================================================
# 1. Create Model and Data
# ==================================================

# Create a model with 1 input and 1 output.
model = nn.Linear(1, 1)

# Set starting weight and bias for predictable results.
with torch.no_grad():
    model.weight.fill_(10.0)
    model.bias.fill_(10.0)

# A student studied 5 hours and received a score of 70.
hours = torch.tensor([[5.0]])
actual_score = torch.tensor([[70.0]])

# Create the loss function and optimizer.
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==================================================
# 2. Train for 5 Epochs
# ==================================================

for epoch in range(5):

    # Set the model to training mode.
    model.train()

    # Make a prediction and calculate loss.
    prediction = model(hours)
    loss = loss_fn(prediction, actual_score)

    # Clear gradients, calculate gradients, and update parameters.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Display the training loss.
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")


# ==================================================
# 3. Save Model and Checkpoint
# ==================================================

# Save only the model parameters.
torch.save(model.state_dict(), "student_model.pth")

# Create a checkpoint to save training progress.
checkpoint = {
    "epoch": 5,                                  # Last completed epoch.
    "model_state_dict": model.state_dict(),      # Model parameters.
    "optimizer_state_dict": optimizer.state_dict(), # Optimizer state.
    "loss": loss.item()                          # Last training loss.
}

# Save the checkpoint to a .pth file.
torch.save(checkpoint, "student_checkpoint.pth")

print("\nModel and checkpoint saved successfully!")


# ==================================================
# 4. Load Checkpoint
# ==================================================

# Create a new model with the same architecture.
new_model = nn.Linear(1, 1)

# Create a new optimizer for the new model.
new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)

# Load the saved checkpoint.
saved_checkpoint = torch.load(
    "student_checkpoint.pth",
    map_location="cpu",  # Load tensors onto the CPU.
    weights_only=True    # Restrict loading to safer supported data types.
)

# Restore saved model parameters.
new_model.load_state_dict(saved_checkpoint["model_state_dict"])

# Restore saved optimizer state.
new_optimizer.load_state_dict(saved_checkpoint["optimizer_state_dict"])

# Get the last completed epoch.
start_epoch = saved_checkpoint["epoch"]


# ==================================================
# 5. Resume Training
# ==================================================

# Continue training for another 5 epochs.
for epoch in range(start_epoch, start_epoch + 5):

    # Set the loaded model to training mode.
    new_model.train()

    # Make a prediction and calculate loss.
    prediction = new_model(hours)
    loss = loss_fn(prediction, actual_score)

    # Clear gradients, calculate gradients, and update parameters.
    new_optimizer.zero_grad()
    loss.backward()
    new_optimizer.step()

    # Display training loss.
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")


# ==================================================
# 6. Final Evaluation
# ==================================================

# Set the loaded model to evaluation mode.
new_model.eval()

# Disable gradient tracking during prediction.
with torch.no_grad():
    prediction = new_model(hours)

print("\n=== Final Evaluation ===")
print("Predicted Score:", round(prediction.item(), 2))
print("Actual Score:", actual_score.item())
