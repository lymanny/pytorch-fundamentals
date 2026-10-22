
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_data_loaders, classes
from model import CIFAR10CNN

# ==================================================
# CIFAR-10 CNN Image Classification: Evaluation
# ==================================================

# Evaluation: Measures CNN performance on unseen test images.
# Purpose: Check how accurately the CNN classifies objects.
#
# model.eval(): Sets the CNN to evaluation mode.
# torch.no_grad(): Disables gradient tracking during evaluation.
# CrossEntropyLoss(): Calculates multiclass classification loss.
# argmax(dim=1): Gets the predicted class for each image.
# load_state_dict(): Restores trained model parameters.


# ==================================================
# 1. Set Model Path and Device
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Set the path to the saved CIFAR-10 model.
model_path = project_root / "checkpoints" / "cifar10" / "cifar10_cnn.pth"

# Use the CPU for evaluation.
device = torch.device("cpu")


# ==================================================
# 2. Load Test Dataset
# ==================================================

# Load the CIFAR-10 test DataLoader.
_, test_loader = get_data_loaders(batch_size=64)

# Display the number of test images.
print("Test Images:", len(test_loader.dataset))


# ==================================================
# 3. Load Trained CNN Model
# ==================================================

# Create the same CNN architecture used during training.
model = CIFAR10CNN().to(device)

# Load the saved model parameters.
saved_parameters = torch.load(
    model_path,
    map_location=device,  # Load saved tensors onto the CPU.
    weights_only=True     # Restrict loading to safer supported data types.
)

# Restore the trained weights and biases into the CNN.
model.load_state_dict(saved_parameters)

# Set the model to evaluation mode.
model.eval()

# Create the classification loss function.
loss_fn = nn.CrossEntropyLoss()

print("Trained CNN loaded successfully!")


# ==================================================
# 4. Evaluate CNN Model
# ==================================================

# Initialize counters for loss, correct predictions, and total images.
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

        # Calculate the classification loss.
        loss = loss_fn(outputs, labels)

        # Add the batch loss multiplied by its image count.
        total_loss += loss.item() * images.size(0)

        # Get the predicted class for each image.
        predictions = outputs.argmax(dim=1)

        # Count correct predictions.
        correct += (predictions == labels).sum().item()

        # Count processed images.
        total += labels.size(0)

# Calculate average test loss.
avg_loss = total_loss / total

# Calculate test accuracy as a percentage.
accuracy = 100 * correct / total

# Calculate the number of incorrect predictions.
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

# Make predictions without calculating gradients.
with torch.no_grad():
    outputs = model(images.to(device))
    predictions = outputs.argmax(dim=1).cpu()

# Create a figure to display 10 test images.
plt.figure(figsize=(12, 5))

# Display the first 10 images and predictions.
for i in range(10):

    # Create a subplot with 2 rows and 5 columns.
    plt.subplot(2, 5, i + 1)

    # Restore normalized RGB pixels from [-1, 1] to [0, 1].
    image = images[i] * 0.5 + 0.5

    # Change image dimensions from (C, H, W) to (H, W, C).
    image = image.permute(1, 2, 0)

    # Display the RGB image.
    plt.imshow(image)

    # Get the actual and predicted class names.
    actual = classes[labels[i].item()]
    predicted = classes[predictions[i].item()]

    # Display actual and predicted classes.
    plt.title(f"Pred: {predicted}\nActual: {actual}", fontsize=9)

    # Hide image axes.
    plt.axis("off")

# Adjust spacing and display the images.
plt.tight_layout()
plt.show()
