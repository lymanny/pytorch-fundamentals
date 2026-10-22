
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_data_loaders
from model import MNISTCNN

# ==================================================
# MNIST CNN Image Classification: Prediction
# ==================================================

# Prediction: Uses a trained CNN to classify handwritten digits.
# Purpose: Predict digit classes from images without retraining.
#
# model.eval(): Sets the CNN to evaluation mode.
# torch.no_grad(): Disables gradient tracking during prediction.
# torch.softmax(): Converts output scores into class probabilities.
# argmax(dim=1): Gets the class with the highest probability.
# load_state_dict(): Restores trained model parameters.


# ==================================================
# 1. Set Model Path and Device
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Set the path to the saved CNN model.
model_path = project_root / "checkpoints" / "mnist" / "mnist_cnn.pth"

# Use the CPU for prediction.
device = torch.device("cpu")


# ==================================================
# 2. Load Test Dataset
# ==================================================

# Load the MNIST test dataset.
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
model = MNISTCNN().to(device)

# Load the saved model parameters.
saved_parameters = torch.load(
    model_path,
    map_location=device,  # Load saved tensors onto the CPU.
    weights_only=True     # Restrict loading to safer supported data types.
)

# Restore the trained weights into the model.
model.load_state_dict(saved_parameters)

# Set the model to evaluation mode.
model.eval()


# ==================================================
# 4. Predict Handwritten Digits
# ==================================================

# Disable gradient tracking during prediction.
with torch.no_grad():

    # Get raw output scores for the selected images.
    outputs = model(images)

    # Convert raw scores into class probabilities.
    probabilities = torch.softmax(outputs, dim=1)

    # Get the predicted digit for each image.
    predictions = probabilities.argmax(dim=1)

    # Get the highest predicted probability for each image.
    confidences = probabilities.max(dim=1).values


# ==================================================
# 5. Display Prediction Results
# ==================================================

print("=== MNIST Predictions ===")

# Display each image's actual digit, prediction, and confidence.
for i in range(len(images)):

    # Convert the confidence score into a percentage.
    confidence = confidences[i].item() * 100

    # Display prediction information.
    print(
        f"Image {i + 1}: "
        f"Actual = {labels[i].item()}, "
        f"Predicted = {predictions[i].item()}, "
        f"Confidence = {confidence:.2f}%"
    )


# ==================================================
# 6. Visualize Predictions
# ==================================================

# Create a figure to display 10 handwritten digits.
plt.figure(figsize=(12, 5))

# Display each image with its prediction.
for i in range(len(images)):

    # Create a subplot with 2 rows and 5 columns.
    plt.subplot(2, 5, i + 1)

    # Display the grayscale handwritten digit.
    plt.imshow(images[i].cpu().squeeze(), cmap="gray")

    # Get the predicted digit and confidence.
    predicted = predictions[i].item()
    actual = labels[i].item()
    confidence = confidences[i].item() * 100

    # Display the actual digit, prediction, and confidence.
    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence:.1f}%"
    )

    # Hide the image axes.
    plt.axis("off")

# Adjust spacing and display predictions.
plt.tight_layout()
plt.show()
