
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================================================
# MNIST CNN Image Classification: Dataset
# ==================================================

# MNIST: Dataset of handwritten digits from 0 to 9.
# Purpose: Load and prepare images for CNN classification.
#
# transforms.ToTensor(): Converts images into PyTorch tensors.
# datasets.MNIST(): Downloads or loads MNIST images and labels.
# DataLoader(): Organizes images into batches.
# batch_size: Number of images in each batch.
# shuffle=True: Randomizes the order of images each epoch.


# ==================================================
# 1. Set Dataset Directory
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Create a path for storing downloaded datasets.
data_dir = project_root / "data"


# ==================================================
# 2. Load MNIST Dataset
# ==================================================

def get_data_loaders(batch_size=64):

    # Convert images to tensors with pixel values from 0 to 1.
    transform = transforms.ToTensor()

    # Load 60,000 images for training.
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    # Load 10,000 images for testing.
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    # Create batches of training images in random order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create batches of test images without shuffling.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Return both DataLoaders for use in other Python files.
    return train_loader, test_loader


# ==================================================
# 3. Test Dataset Loading
# ==================================================

if __name__ == "__main__":

    # Load training and test DataLoaders.
    train_loader, test_loader = get_data_loaders()

    # Get the total number of training and test images.
    print("Training Images:", len(train_loader.dataset))
    print("Test Images:", len(test_loader.dataset))

    # Get one batch of training images and labels.
    images, labels = next(iter(train_loader))

    # Display batch shapes.
    print("\n=== Batch Information ===")
    print("Images Shape:", images.shape)
    print("Labels Shape:", labels.shape)

    # Display the first 10 labels.
    print("Sample Labels:", labels[:10].tolist())


    # ==================================================
    # 4. Visualize Sample Images
    # ==================================================

    # Create a figure for 10 images.
    plt.figure(figsize=(10, 4))

    # Display the first 10 handwritten digits.
    for i in range(10):

        # Create a subplot with 2 rows and 5 columns.
        plt.subplot(2, 5, i + 1)

        # Remove the channel dimension and display in grayscale.
        plt.imshow(images[i].squeeze(), cmap="gray")

        # Display the actual digit label.
        plt.title(f"Label: {labels[i].item()}")

        # Hide image axes.
        plt.axis("off")

    # Adjust spacing and display the images.
    plt.tight_layout()
    plt.show()
