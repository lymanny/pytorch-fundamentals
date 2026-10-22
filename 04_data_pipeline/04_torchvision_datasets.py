
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================================================
# Data Pipeline: torchvision.datasets
# ==================================================

# torchvision.datasets: Provides ready-to-use image datasets.
# Purpose: Load images and labels easily.
#
# MNIST(): Loads handwritten digit images from 0 to 9.
# download=True: Downloads the dataset if it does not exist.
# ToTensor(): Converts images into tensors.
#
# iter(loader): Creates an iterator to access batches.
# next(): Gets the next batch from the iterator.


# ==================================================
# 1. Load Dataset
# ==================================================

# Load the MNIST training dataset.
dataset = datasets.MNIST(
    root="data",                     # Folder to store the dataset.
    train=True,                      # Use training images.
    download=True,                   # Download the dataset if needed.
    transform=transforms.ToTensor()  # Convert images into tensors.
)

# Display the total number of images.
print("=== MNIST Dataset ===")
print("Total Images:", len(dataset))  # 60000

# Get the first image and its correct label.
image, label = dataset[0]

print("Image Shape:", image.shape)  # (1, 28, 28): 1 channel, 28 height, 28 width
print("Label:", label)              # Digit from 0 to 9


# ==================================================
# 2. Load Images in Batches
# ==================================================

# Create batches of 32 images and shuffle their order.
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("\n=== DataLoader ===")
print("Total Batches:", len(loader))  # 1875

# iter(loader) creates an iterator and next() gets the first batch.
images, labels = next(iter(loader))

# Display the first batch information.
print("Batch Shape:", images.shape)  # (32, 1, 28, 28): 32 grayscale images
print("Labels Shape:", labels.shape) # (32,): 32 labels
print("Labels:", labels)
