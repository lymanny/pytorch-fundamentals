
import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================
# Training Loop: Learning Rate Scheduler
# ==================================================

# Learning Rate Scheduler: Adjusts the learning rate during training.
# Purpose: Help the model train effectively.
#
# StepLR(): Reduces the learning rate after a fixed number of epochs.
# step_size=5: Reduce the learning rate every 5 epochs.
# gamma=0.5: Multiply the learning rate by 0.5.
# scheduler.step(): Updates the learning rate after each epoch.
# optimizer.param_groups[0]["lr"]: Gets the current learning rate.


# ==================================================
# 1. Create Model and Data
# ==================================================

# Create a Linear model with 1 input and 1 output.
model = nn.Linear(1, 1)

# Set starting weight and bias.
with torch.no_grad():
    model.weight.fill_(10.0)
    model.bias.fill_(10.0)

# A student studied 5 hours and received a score of 70.
hours = torch.tensor([[5.0]])
actual_score = torch.tensor([[70.0]])


# ==================================================
# 2. Create Loss, Optimizer, and Scheduler
# ==================================================

# Create the loss function.
loss_fn = nn.MSELoss()

# Create an optimizer with an initial learning rate of 0.01.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create a scheduler to reduce the learning rate by half every 5 epochs.
scheduler = optim.lr_scheduler.StepLR(
    optimizer,     # Optimizer whose learning rate will be updated.
    step_size=5,   # Reduce the learning rate every 5 epochs.
    gamma=0.5      # Multiply the learning rate by 0.5 (reduce by 50%).
)


# ==================================================
# 3. Train for 15 Epochs
# ==================================================

for epoch in range(15):

    # Set the model to training mode.
    model.train()

    # Make a prediction and calculate loss.
    prediction = model(hours)
    loss = loss_fn(prediction, actual_score)

    # Clear gradients, calculate gradients, and update weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get the learning rate used in this epoch.
    current_lr = optimizer.param_groups[0]["lr"]

    # Display the epoch, prediction, loss, and learning rate.
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}, LR = {current_lr:.4f}")

    # Update the learning rate for the next epoch.
    scheduler.step()


# ==================================================
# 4. Final Evaluation
# ==================================================

# Set the model to evaluation mode.
model.eval()

# Make a prediction without calculating gradients.
with torch.no_grad():
    final_prediction = model(hours)

print("\n=== Final Evaluation ===")
print("Predicted Score:", round(final_prediction.item(), 2))
print("Actual Score:", actual_score.item())
print("Final Learning Rate:", scheduler.get_last_lr()[0])
