
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from dataset import get_data_loaders
from model import CIFAR10CNN

# ==================================================
# CIFAR-10 CNN Image Classification: Training
# ==================================================

# Training: Teaches a CNN to recognize objects in RGB images.
# Purpose: Improve classification accuracy by updating model parameters.
#
# CrossEntropyLoss(): Calculates multiclass classification loss.
# optim.Adam(): Updates model weights and biases.
# model.train(): Sets the CNN to training mode.
# loss.backward(): Calculates gradients using Autograd.
# optimizer.step(): Updates model parameters.
# torch.save(): Saves trained model parameters.
# SummaryWriter(): Records training metrics for TensorBoard.


# ==================================================
# 1. Set Directories and Device
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Set directories for saving model parameters and TensorBoard logs.
checkpoint_dir = project_root / "checkpoints" / "cifar10"
log_dir = project_root / "runs" / "cifar10"

# Create the checkpoint directory if it does not exist.
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Use the CPU for this beginner project.
device = torch.device("cpu")

print("Device:", device)


# ==================================================
# 2. Load CIFAR-10 Dataset
# ==================================================

# Load the training and test DataLoaders with 64 images per batch.
train_loader, test_loader = get_data_loaders(batch_size=64)

# Display the number of training images.
print("Training Images:", len(train_loader.dataset))


# ==================================================
# 3. Create CNN Model
# ==================================================

# Create the CIFAR-10 CNN model.
model = CIFAR10CNN().to(device)

# Create the loss function for 10-class classification.
loss_fn = nn.CrossEntropyLoss()

# Create the Adam optimizer with a learning rate of 0.001.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the TensorBoard writer.
writer = SummaryWriter(log_dir=str(log_dir))


# ==================================================
# 4. Train CNN Model
# ==================================================

# Set the number of training epochs.
num_epochs = 5

# Repeat training for 5 epochs.
for epoch in range(num_epochs):

    # Set the model to training mode.
    model.train()

    # Reset loss, correct predictions, and image count.
    total_loss = 0.0
    correct = 0
    total = 0

    # Process training images one batch at a time.
    for images, labels in train_loader:

        # Move images and labels to the selected device.
        images = images.to(device)
        labels = labels.to(device)

        # Make predictions using the CNN.
        outputs = model(images)

        # Calculate the classification loss.
        loss = loss_fn(outputs, labels)

        # Clear previous gradients.
        optimizer.zero_grad()

        # Calculate new gradients using Autograd.
        loss.backward()

        # Update the model weights and biases.
        optimizer.step()

        # Add the batch loss multiplied by its image count.
        total_loss += loss.item() * images.size(0)

        # Get the predicted class for each image.
        predictions = outputs.argmax(dim=1)

        # Count correct predictions.
        correct += (predictions == labels).sum().item()

        # Count processed images.
        total += labels.size(0)

    # Calculate average training loss per image.
    avg_loss = total_loss / total

    # Calculate training accuracy as a percentage.
    accuracy = 100 * correct / total

    # Record training loss in TensorBoard.
    writer.add_scalar("Loss/Training", avg_loss, epoch + 1)

    # Record training accuracy in TensorBoard.
    writer.add_scalar("Accuracy/Training", accuracy, epoch + 1)

    # Display training results for each epoch.
    print(
        f"Epoch {epoch + 1}/{num_epochs}: "
        f"Loss = {avg_loss:.4f}, "
        f"Accuracy = {accuracy:.2f}%"
    )


# ==================================================
# 5. Save Trained CNN Model
# ==================================================

# Set the path for saving the trained CNN model.
model_path = checkpoint_dir / "cifar10_cnn.pth"

# Save the trained model parameters.
torch.save(model.state_dict(), model_path)

# Save remaining TensorBoard logs and close the writer.
writer.close()

print("\nTraining completed successfully!")
print("Model Saved:", model_path)
print("TensorBoard Logs:", log_dir)


# ==================================================
# 6. View Results in TensorBoard
# ==================================================

# Step 1: Open the PyCharm terminal in the project root.
# Step 2: Run the command: tensorboard --logdir=runs (both), runs/mnist (MNIST), or runs/cifar10 (CIFAR-10).
# Step 3: Open http://localhost:6006/ in your browser.
# Step 4: Select the CIFAR-10 run and view training graphs.
# Step 5: Press Ctrl + C in the terminal to stop TensorBoard.
