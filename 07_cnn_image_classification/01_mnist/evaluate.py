
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_data_loaders
from model import MNISTCNN

# ==================================================
# MNIST CNN Image Classification: Evaluation
# ==================================================

# Evaluation: Measures model performance on unseen test images.
# Purpose: Check how accurately the trained CNN classifies digits.
#
# model.eval(): Sets the CNN to evaluation mode.
# torch.no_grad(): Disables gradient tracking during evaluation.
# CrossEntropyLoss(): Calculates multiclass classification loss.
# argmax(dim=1): Gets the predicted digit for each image.
# load_state_dict(): Restores saved parameters into the model.


# ==================================================
# 1. Set Model Path and Device
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Set the path to the trained CNN model.
model_path = project_root / "checkpoints" / "mnist" / "mnist_cnn.pth"

# Use the CPU for model evaluation.
device = torch.device("cpu")


# ==================================================
# 2. Load Test Dataset
# ==================================================

# Load the MNIST DataLoaders.
_, test_loader = get_data_loaders(batch_size=64)

# Display the number of test images.
print("Test Images:", len(test_loader.dataset))


# ==================================================
# 3. Load Trained CNN Model
# ==================================================

# Create the same CNN architecture used during training.
model = MNISTCNN().to(device)

# Load the saved model parameters from the file.
saved_parameters = torch.load(
    model_path,
    map_location=device,  # Load saved tensors onto the CPU.
    weights_only=True     # Restrict loading to safer supported data types.
)

# Restore trained weights and biases into the CNN.
model.load_state_dict(saved_parameters)

# Set the model to evaluation mode.
model.eval()

# Create the classification loss function.
loss_fn = nn.CrossEntropyLoss()

print("Trained CNN loaded successfully!")


# ==================================================
# 4. Evaluate CNN Model
# ==================================================

# Initialize counters for evaluation.
total_loss = 0.0
correct = 0
total = 0

# Disable gradient tracking during evaluation.
with torch.no_grad():

    # Process test images one batch at a time.
    for images, labels in test_loader:

        # Move images and labels to the selected device.
        images = images.to(device)
        labels = labels.to(device)

        # Make predictions using the trained CNN.
        outputs = model(images)

        # Calculate classification loss.
        loss = loss_fn(outputs, labels)

        # Add the batch loss weighted by its number of images.
        total_loss += loss.item() * images.size(0)

        # Get the predicted digit for each image.
        predictions = outputs.argmax(dim=1)

        # Count correct predictions.
        correct += (predictions == labels).sum().item()

        # Count processed images.
        total += labels.size(0)

# Calculate average test loss.
avg_loss = total_loss / total

# Calculate test accuracy as a percentage.
accuracy = 100 * correct / total

# Calculate incorrect predictions.
incorrect = total - correct

# Display evaluation results.
print("\n=== CNN Evaluation Results ===")
print("Test Images:", total)
print("Correct Predictions:", correct)
print("Incorrect Predictions:", incorrect)
print("Test Accuracy:", round(accuracy, 2), "%")
print("Average Test Loss:", round(avg_loss, 4))


# ==================================================
# 5. Visualize Sample Predictions
# ==================================================

# Get one batch of test images and labels.
images, labels = next(iter(test_loader))

# Set the model to evaluation mode.
model.eval()

# Predict the digits without tracking gradients.
with torch.no_grad():
    outputs = model(images.to(device))
    predictions = outputs.argmax(dim=1).cpu()

# Create a figure to display 10 test images.
plt.figure(figsize=(10, 4))

# Display the first 10 test images and predictions.
for i in range(10):

    # Create a subplot with 2 rows and 5 columns.
    plt.subplot(2, 5, i + 1)

    # Remove the channel dimension and show the image.
    plt.imshow(images[i].squeeze(), cmap="gray")

    # Get the actual and predicted digit.
    actual = labels[i].item()
    predicted = predictions[i].item()

    # Display the prediction and actual label.
    plt.title(f"Pred: {predicted} | Actual: {actual}")

    # Hide image axes.
    plt.axis("off")

# Adjust spacing and display predictions.
plt.tight_layout()
plt.show()
