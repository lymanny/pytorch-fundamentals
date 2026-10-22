
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# ==================================================
# Saving and Logging: TensorBoard
# ==================================================

# TensorBoard: Visualizes model training metrics using graphs.
# Purpose: Monitor training progress and model performance.
#
# SummaryWriter(): Creates a writer to record training metrics.
# writer.add_scalar(): Records a numeric value for visualization.
# writer.close(): Finishes writing and closes the log writer.
# log_dir: Specifies the folder for saving TensorBoard logs.
# Path(__file__): Gets the path of the current Python file.


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
# 2. Create TensorBoard Writer
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent

# Set the TensorBoard log directory inside the project root.
log_dir = project_root / "runs" / "student_score"

# Create a writer to record training metrics.
writer = SummaryWriter(log_dir=str(log_dir))

# Display the exact log directory.
print("TensorBoard Log Directory:", log_dir)


# ==================================================
# 3. Train and Record Loss
# ==================================================

# Repeat training for 15 epochs.
for epoch in range(15):

    # Set the model to training mode.
    model.train()

    # Make a prediction and calculate loss.
    prediction = model(hours)
    loss = loss_fn(prediction, actual_score)

    # Clear previous gradients.
    optimizer.zero_grad()

    # Calculate new gradients using Autograd.
    loss.backward()

    # Update model weights and bias.
    optimizer.step()

    # Set the model to evaluation mode.
    model.eval()

    # Calculate the loss after updating the model.
    with torch.no_grad():
        prediction = model(hours)
        eval_loss = loss_fn(prediction, actual_score)

    # Record training loss in TensorBoard.
    writer.add_scalar("Loss/Training", loss.item(), epoch + 1)

    # Record evaluation loss in TensorBoard.
    writer.add_scalar("Loss/Evaluation", eval_loss.item(), epoch + 1)

    # Display training results.
    print(f"Epoch {epoch + 1}: Loss = {eval_loss.item():.4f}")


# ==================================================
# 4. Close TensorBoard Writer
# ==================================================

# Save remaining logs and close the writer.
writer.close()

print("\nTensorBoard logs saved successfully!")


# ==================================================
# 5. Final Evaluation
# ==================================================

# Set the model to evaluation mode.
model.eval()

# Make a prediction without calculating gradients.
with torch.no_grad():
    final_prediction = model(hours)

print("\n=== Final Evaluation ===")
print("Predicted Score:", round(final_prediction.item(), 2))
print("Actual Score:", actual_score.item())


# ==================================================
# 6. View Results in TensorBoard
# ==================================================

# Step 1: Open the PyCharm terminal in the project root.
# Step 2: Run the command: tensorboard --logdir=runs
# Step 3: Open http://localhost:6006/ in your browser.
# Step 4: View the Loss/Training and Loss/Evaluation graphs.
# Step 5: Press Ctrl + C in the terminal to stop TensorBoard.
