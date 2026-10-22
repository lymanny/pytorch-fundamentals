
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================================================
# CIFAR-10 CNN Image Classification: Dataset
# ==================================================

# CIFAR-10: Dataset containing 10 categories of color images.
# Purpose: Load and prepare RGB images for CNN classification.
#
# transforms.ToTensor(): Converts images into PyTorch tensors.
# transforms.Normalize(): Normalizes image channels.
# datasets.CIFAR10(): Downloads or loads CIFAR-10 images and labels.
# DataLoader(): Organizes images into batches.
# batch_size: Number of images per batch.
# shuffle=True: Randomizes the training image order.


# ==================================================
# 1. Set Dataset Directory
# ==================================================

# Get the project root folder.
project_root = Path(__file__).resolve().parent.parent.parent

# Set the directory for downloaded datasets.
data_dir = project_root / "data"

# Define the 10 CIFAR-10 class names.
classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


# ==================================================
# 2. Load CIFAR-10 Dataset
# ==================================================

def get_data_loaders(batch_size=64):

    # Convert RGB images to tensors and normalize their channels.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load 50,000 images for training.
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    # Load 10,000 images for testing.
    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    # Create batches of shuffled training images.
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

    # Return training and test DataLoaders.
    return train_loader, test_loader


# ==================================================
# 3. Test Dataset Loading
# ==================================================

if __name__ == "__main__":

    # Load training and test DataLoaders.
    train_loader, test_loader = get_data_loaders()

    # Display dataset sizes.
    print("Training Images:", len(train_loader.dataset))
    print("Test Images:", len(test_loader.dataset))

    # Get one batch of training images and labels.
    images, labels = next(iter(train_loader))

    # Display batch shapes.
    print("\n=== Batch Information ===")
    print("Images Shape:", images.shape)
    print("Labels Shape:", labels.shape)

    # Display the first 10 class labels.
    print("Sample Labels:", [classes[i] for i in labels[:10].tolist()])


    # ==================================================
    # 4. Visualize Sample Images
    # ==================================================

    # Create a figure to display 10 sample images.
    plt.figure(figsize=(10, 4))

    # Display the first 10 CIFAR-10 images.
    for i in range(10):

        # Create a subplot with 2 rows and 5 columns.
        plt.subplot(2, 5, i + 1)

        # Restore pixel values from [-1, 1] to [0, 1].
        image = images[i] * 0.5 + 0.5

        # Convert image shape from (C, H, W) to (H, W, C).
        image = image.permute(1, 2, 0)

        # Display the RGB image.
        plt.imshow(image)

        # Display the actual class name.
        plt.title(classes[labels[i].item()])

        # Hide image axes.
        plt.axis("off")

    # Adjust spacing and display the images.
    plt.tight_layout()
    plt.show()
