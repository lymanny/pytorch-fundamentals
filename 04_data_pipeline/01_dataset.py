
import torch
from torch.utils.data import Dataset

# ==================================================
# Data Pipeline: Dataset
# ==================================================

# Dataset: A PyTorch base class for creating custom datasets.
# Purpose: Store and access individual data samples and labels.
#
# __init__(): Runs when the dataset object is created.
# Purpose: Store the dataset information.
#
# __len__(): Runs when len(dataset) is called.
# Purpose: Return the total number of samples.
#
# __getitem__(): Runs when dataset[index] is called.
# Purpose: Return one sample and its label.


# ==================================================
# 1. Create a Custom Dataset
# ==================================================

# Create a custom dataset by inheriting from Dataset.
class StudentDataset(Dataset):

    # Store student study hours and exam scores.
    def __init__(self):
        self.hours = torch.tensor([2.0, 4.0, 6.0])
        self.scores = torch.tensor([60.0, 75.0, 90.0])

    # Return the number of students.
    def __len__(self):
        return len(self.hours)

    # Return one student's study hours and exam score.
    def __getitem__(self, index):
        return self.hours[index], self.scores[index]


# ==================================================
# 2. Create and Access the Dataset
# ==================================================

# Create the dataset and automatically run __init__().
dataset = StudentDataset()

# Call __len__() to count the students.
print("Total Students:", len(dataset))  # 3

# Call __getitem__(0) to access the first student.
hours, score = dataset[0]

# Display the first student's information.
print("Study Hours:", hours.item())  # 2.0
print("Exam Score:", score.item())   # 60.0


# ==================================================
# 3. Access All Students
# ==================================================

# Loop through all students using their indexes.
for index in range(len(dataset)):
    hours, score = dataset[index]
    print(f"Student {index + 1}: {hours.item()} hours, Score: {score.item()}")
