
import torch
from torch.utils.data import Dataset, DataLoader

# ==================================================
# Data Pipeline: DataLoader
# ==================================================

# DataLoader: Loads dataset samples in batches.
# Purpose: Prepare batches for training and evaluation.
#
# batch_size: Maximum number of samples in each batch.
# shuffle=True: Randomly changes the order of samples.
# shuffle=False: Keeps the original order of samples.
#
# DataLoader(dataset, batch_size=2): Creates batches of up to 2 students.
# len(loader): Returns the total number of batches.
# enumerate(loader): Provides batch indexes and batch data.


# ==================================================
# 1. Create a Student Dataset
# ==================================================

class StudentDataset(Dataset):

    # Store student study hours and exam scores.
    def __init__(self):
        self.hours = torch.tensor([2.0, 4.0, 6.0])
        self.scores = torch.tensor([60.0, 75.0, 90.0])

    # Return the total number of students.
    def __len__(self):
        return len(self.hours)

    # Return one student's study hours and exam score.
    def __getitem__(self, index):
        return self.hours[index], self.scores[index]


# ==================================================
# 2. Create a DataLoader
# ==================================================

# Create the student dataset.
dataset = StudentDataset()

# Create batches 2 students without shuffling.
loader = DataLoader(dataset, batch_size=2, shuffle=False)

print("=== DataLoader ===")
print("Total Students:", len(dataset))  # 3
print("Total Batches:", len(loader))     # 2


# ==================================================
# 3. Load Data in Batches
# ==================================================

# Get each batch index and its student data.
for batch_index, (hours, scores) in enumerate(loader):
    print(f"\nBatch {batch_index + 1}")
    print("Study Hours:", hours)
    print("Exam Scores:", scores)


# ==================================================
# 4. Load Data with Shuffling
# ==================================================

# Randomly change the student order before creating batches.
shuffle_loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("\n=== DataLoader with Shuffle ===")

# Get each batch in a random order.
for batch_index, (hours, scores) in enumerate(shuffle_loader):
    print(f"\nBatch {batch_index + 1}")
    print("Study Hours:", hours)
    print("Exam Scores:", scores)
