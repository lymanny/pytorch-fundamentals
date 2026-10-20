
import torch

# ==================================================
# Axis Order of Image Tensors (B, C, H, W)
# ==================================================

# Axis Order:
# The arrangement of dimensions in an image tensor.
#
# B = Batch size (number of images)
# C = Channels (RGB = 3, Grayscale = 1)
# H = Height (pixels)
# W = Width (pixels)
#
# CHW  = Single image
# BCHW = Batch of images


# ==================================================
# 1. Single Image (CHW)
# ==================================================

# One RGB image: 3 channels, 224x224 pixels.
image = torch.zeros(3, 224, 224)

print("Single Image:", image.shape)
# (3, 224, 224)


# ==================================================
# 2. Batch of Images (BCHW)
# ==================================================

# 10 RGB images, each 224x224 pixels.
images = torch.zeros(10, 3, 224, 224)

print("\nBatch Images:", images.shape)
# (10, 3, 224, 224)


# ==================================================
# 3. Convert HWC -> CHW
# ==================================================

# permute(): Rearranges dimension order.
image = torch.zeros(224, 224, 3)

image = image.permute(2, 0, 1)

print("\nCHW Shape:", image.shape)
# (3, 224, 224)


# ==================================================
# 4. Convert CHW -> BCHW
# ==================================================

# unsqueeze(): Adds a new dimension.
image = image.unsqueeze(0)

print("\nBCHW Shape:", image.shape)
# (1, 3, 224, 224)


# ==================================================
# 5. Access Image Dimensions
# ==================================================

images = torch.zeros(10, 3, 224, 224)

print("\nBatch Size:", images.shape[0])  # 10
print("Channels:", images.shape[1])     # 3
print("Height:", images.shape[2])       # 224
print("Width:", images.shape[3])        # 224
