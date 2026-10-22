
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ==================================================
# Data Pipeline: torchvision.transforms
# ==================================================

# transforms: Prepares images for neural networks.
# Purpose: Resize, convert, normalize, and augment images.
#
# Resize(): Changes image size.
# RandomHorizontalFlip(): Randomly flips images horizontally.
# ToTensor(): Converts images into tensors.
# Normalize(): Normalizes pixel values.
# Compose(): Combines transformations in order.
#
# min(): Finds the smallest pixel value.
# max(): Finds the largest pixel value.
# Purpose: Check the pixel value range after normalization.


# ==================================================
# 1. Load Image
# ==================================================

# Load the Zenitsu image as RGB.
image = Image.open("images/zenitsu.png").convert("RGB")

print("Original Size:", image.size)


# ==================================================
# 2. Transform Image
# ==================================================

# Apply image transformations in order.
transform = transforms.Compose([
    transforms.Resize(256),                  # Resize shorter side to 256 while keeping proportions.
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally.
    transforms.ToTensor(),                   # Convert image to tensor with values from 0 to 1.
    transforms.Normalize(                    # Normalize pixel values to approximately -1 to 1.
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Apply transformations to the image.
result = transform(image)

print("\n=== Transformed Image ===")
print("Tensor Shape:", result.shape)         # (C, H, W)
print("Minimum Value:", result.min().item()) # Check the smallest pixel value.
print("Maximum Value:", result.max().item()) # Check the largest pixel value.


# ==================================================
# 3. Display Images
# ==================================================

# Reverse normalization to display the original colors.
display_image = result * 0.5 + 0.5

# Convert tensor shape from (C, H, W) to (H, W, C).
display_image = display_image.permute(1, 2, 0)

# Display the original image.
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Display the transformed image.
plt.subplot(1, 2, 2)
plt.imshow(display_image.clamp(0, 1))
plt.title("Transformed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
