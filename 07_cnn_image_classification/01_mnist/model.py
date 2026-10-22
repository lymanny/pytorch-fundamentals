
import torch
import torch.nn as nn

# ==================================================
# MNIST CNN Image Classification: Model
# ==================================================

# CNN: Neural network that learns features from images.
# Purpose: Classify handwritten digits from 0 to 9.
#
# nn.Module: Base class for building neural networks.
# nn.Conv2d(): Extracts image features using convolution filters.
# nn.ReLU(): Applies a nonlinear activation function.
# nn.MaxPool2d(): Reduces feature map dimensions.
# nn.Flatten(): Converts feature maps into a 1D feature vector per image.
# nn.Linear(): Connects features to output neurons.


# ==================================================
# 1. Build CNN Model
# ==================================================

# Define a CNN model using nn.Module.
class MNISTCNN(nn.Module):

    # Initialize the CNN layers.
    def __init__(self):
        super().__init__()

        # Extract image features using convolution and pooling.
        self.features = nn.Sequential(

            # First convolution: 1 input channel to 16 feature maps.
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),

            # Apply ReLU activation.
            nn.ReLU(),

            # Reduce image size from 28x28 to 14x14.
            nn.MaxPool2d(kernel_size=2),

            # Second convolution: 16 channels to 32 feature maps.
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),

            # Apply ReLU activation.
            nn.ReLU(),

            # Reduce feature maps from 14x14 to 7x7.
            nn.MaxPool2d(kernel_size=2)
        )

        # Convert feature maps into a flat vector.
        self.flatten = nn.Flatten()

        # Classify extracted image features.
        self.classifier = nn.Sequential(

            # Convert 1568 features into 128 neurons.
            nn.Linear(32 * 7 * 7, 128),

            # Apply ReLU activation.
            nn.ReLU(),

            # Produce 10 output scores for digits 0 to 9.
            nn.Linear(128, 10)
        )

    # Define how images pass through the CNN.
    def forward(self, x):

        # Extract image features.
        x = self.features(x)

        # Flatten feature maps.
        x = self.flatten(x)

        # Produce classification scores.
        x = self.classifier(x)

        # Return output scores.
        return x


# ==================================================
# 2. Test CNN Model
# ==================================================

if __name__ == "__main__":

    # Create the CNN model.
    model = MNISTCNN()

    # Create a sample batch of 64 grayscale images.
    images = torch.randn(64, 1, 28, 28)

    # Make predictions without tracking gradients.
    with torch.no_grad():
        outputs = model(images)

    # Display the CNN architecture.
    print("=== CNN Architecture ===")
    print(model)

    # Display input and output shapes.
    print("\n=== Model Test ===")
    print("Input Shape:", images.shape)
    print("Output Shape:", outputs.shape)

    # Get the predicted digit for the first image.
    predicted_digit = outputs[0].argmax().item()

    print("Predicted Digit:", predicted_digit)
