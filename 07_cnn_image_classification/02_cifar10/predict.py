
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_data_loaders, classes
from model import CIFAR10CNN

# ==================================================
# CIFAR-10 CNN Image Classification: Prediction
# ==================================================

# Prediction: Uses a trained CNN to classify RGB images.
# Purpose: Predict object categories without retraining.
#
# model.eval(): Sets the CNN to evaluation mode.
# torch.no_grad(): Disables gradient tracking during prediction.
# torch.softmax(): Converts raw output scores into probabilities.
# argmax(dim=1): Gets the predicted class index.
# load_state_dict(): Restores trained CNN parameters.


# ==================================================
# 1. Set Model Path and Device
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Set the path to the saved CIFAR-10 model.
model_path = project_root / "checkpoints" / "cifar10" / "cifar10_cnn.pth"

# Use the CPU for prediction.
device = torch.device("cpu")


# ==================================================
# 2. Load Test Dataset
# ==================================================

# Load the CIFAR-10 test dataset.
_, test_loader = get_data_loaders(batch_size=64)

# Get one batch of test images and labels.
images, labels = next(iter(test_loader))

# Select the first 10 images and labels.
images = images[:10].to(device)
labels = labels[:10]


# ==================================================
# 3. Load Trained CNN Model
# ==================================================

# Create the same CNN architecture used during training.
model = CIFAR10CNN().to(device)

# Load the trained model parameters from the file.
saved_parameters = torch.load(
    model_path,
    map_location=device,  # Load saved tensors onto the CPU.
    weights_only=True     # Restrict loading to safer supported data types.
)

# Restore the trained weights and biases.
model.load_state_dict(saved_parameters)

# Set the model to evaluation mode.
model.eval()


# ==================================================
# 4. Predict Object Categories
# ==================================================

# Disable gradient tracking during prediction.
with torch.no_grad():

    # Get raw output scores for the selected images.
    outputs = model(images)

    # Convert raw scores into class probabilities.
    probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted class index for each image.
    predictions = probabilities.argmax(dim=1)

    # Get the highest probability for each image.
    confidences = probabilities.max(dim=1).values


# ==================================================
# 5. Display Prediction Results
# ==================================================

print("=== CIFAR-10 Predictions ===")

# Display actual class, predicted class, and confidence.
for i in range(len(images)):

    # Convert the confidence into a percentage.
    confidence = confidences[i].item() * 100

    # Get the actual and predicted class names.
    actual = classes[labels[i].item()]
    predicted = classes[predictions[i].item()]

    # Display the prediction information.
    print(
        f"Image {i + 1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}, "
        f"Confidence = {confidence:.2f}%"
    )


# ==================================================
# 6. Visualize Predictions
# ==================================================

# Create a figure for 10 CIFAR-10 images.
plt.figure(figsize=(12, 5))

# Display each image with its prediction.
for i in range(len(images)):

    # Create a subplot with 2 rows and 5 columns.
    plt.subplot(2, 5, i + 1)

    # Restore normalized RGB values from [-1, 1] to [0, 1].
    image = images[i].cpu() * 0.5 + 0.5

    # Change image dimensions from (C, H, W) to (H, W, C).
    image = image.permute(1, 2, 0)

    # Get predicted class, actual class, and confidence.
    predicted = classes[predictions[i].item()]
    actual = classes[labels[i].item()]
    confidence = confidences[i].item() * 100

    # Display the RGB image.
    plt.imshow(image)

    # Display prediction information above the image.
    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence:.1f}%",
        fontsize=9
    )

    # Hide image axes.
    plt.axis("off")

# Adjust spacing and display the images.
plt.tight_layout()
plt.show()
