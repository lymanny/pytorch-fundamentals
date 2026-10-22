
import torch
import torch.nn as nn

# ==================================================
# CIFAR-10 CNN Image Classification: Model
# ==================================================

# CNN: Learns image features to classify objects.
# Purpose: Recognize 10 categories of RGB images.
#
# nn.Module: Base class for building neural networks.
# nn.Conv2d(): Extracts image features using convolution filters.
# nn.ReLU(): Adds nonlinearity to the neural network.
# nn.MaxPool2d(): Reduces feature map dimensions.
# nn.Flatten(): Converts feature maps into feature vectors.
# nn.Linear(): Connects features to output neurons.


# ==================================================
# 1. Build CNN Model
# ==================================================

# Define a CNN model using nn.Module.
class CIFAR10CNN(nn.Module):

    # Initialize the CNN layers.
    def __init__(self):
        super().__init__()

        # Extract features using convolution and pooling layers.
        self.features = nn.Sequential(

            # First convolution: 3 RGB channels to 32 feature maps.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),

            # Apply ReLU activation.
            nn.ReLU(),

            # Reduce feature maps from 32x32 to 16x16.
            nn.MaxPool2d(kernel_size=2),

            # Second convolution: 32 channels to 64 feature maps.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),

            # Apply ReLU activation.
            nn.ReLU(),

            # Reduce feature maps from 16x16 to 8x8.
            nn.MaxPool2d(kernel_size=2),

            # Third convolution: 64 channels to 128 feature maps.
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),

            # Apply ReLU activation.
            nn.ReLU(),

            # Reduce feature maps from 8x8 to 4x4.
            nn.MaxPool2d(kernel_size=2)
        )

        # Convert feature maps into feature vectors.
        self.flatten = nn.Flatten()

        # Classify the extracted image features.
        self.classifier = nn.Sequential(

            # Convert 2048 features into 128 neurons.
            nn.Linear(128 * 4 * 4, 128),

            # Apply ReLU activation.
            nn.ReLU(),

            # Produce 10 output scores for CIFAR-10 classes.
            nn.Linear(128, 10)
        )

    # Define how input images pass through the CNN.
    def forward(self, x):

        # Extract image features.
        x = self.features(x)

        # Flatten the feature maps.
        x = self.flatten(x)

        # Produce classification scores.
        x = self.classifier(x)

        # Return raw output scores.
        return x


# ==================================================
# 2. Test CNN Model
# ==================================================

if __name__ == "__main__":

    # Create the CNN model.
    model = CIFAR10CNN()

    # Create a sample batch of 64 RGB images.
    images = torch.randn(64, 3, 32, 32)

    # Make predictions without calculating gradients.
    with torch.no_grad():
        outputs = model(images)

    # Display the CNN architecture.
    print("=== CNN Architecture ===")
    print(model)

    # Display input and output shapes.
    print("\n=== Model Test ===")
    print("Input Shape:", images.shape)
    print("Output Shape:", outputs.shape)

    # Get the predicted class for the first image.
    predicted_class = outputs[0].argmax().item()

    print("Predicted Class Index:", predicted_class)
