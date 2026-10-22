
import torch
import torch.nn as nn
import torch.optim as optim

# ==================================================
# Training Loop: Training and Evaluation Functions
# ==================================================

# Training Function: Trains the model by updating its parameters.
# Purpose: Help the model learn from data.
#
# Evaluation Function: Checks predictions without updating parameters.
# Purpose: Measure model performance.
#
# Epoch: One complete pass through the training dataset.
# Purpose: Repeat training to improve predictions.
#
# model.train(): Sets the model to training mode.
# model.eval(): Sets the model to evaluation mode.
# torch.no_grad(): Disables gradient tracking during evaluation.


# ==================================================
# 1. Create Model and Data
# ==================================================

# Create a model with 1 input and 1 output.
model = nn.Linear(1, 1)

# Set starting weight and bias for predictable results.
with torch.no_grad():
    model.weight.fill_(10.0)
    model.bias.fill_(10.0)

# A student studied 5 hours and received a score of 70.
hours = torch.tensor([[5.0]])
actual_score = torch.tensor([[70.0]])

# Create the loss function and optimizer.
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==================================================
# 2. Training Function
# ==================================================

# Define a function to train the model for one epoch.
def train():

    # 1. Set the model to training mode.
    model.train()

    # 2. Make a prediction using study hours.
    prediction = model(hours)

    # 3. Calculate the prediction error.
    loss = loss_fn(prediction, actual_score)

    # 4. Clear previous gradients.
    optimizer.zero_grad()

    # 5. Calculate new gradients using Autograd.
    loss.backward()

    # 6. Update model weights and bias.
    optimizer.step()

    # 7. Return the loss value before the update.
    return loss.item()


# ==================================================
# 3. Evaluation Function
# ==================================================

# Define a function to evaluate the model.
def evaluate():

    # Set the model to evaluation mode.
    model.eval()

    # Disable gradients because parameters are not updated.
    with torch.no_grad():

        # Make a prediction.
        prediction = model(hours)

        # Calculate prediction error.
        loss = loss_fn(prediction, actual_score)

    # Return the predicted score and loss.
    return prediction.item(), loss.item()


# ==================================================
# 4. Train for Multiple Epochs
# ==================================================

# Repeat training for 15 epochs.
for epoch in range(15):

    # Train the model and update its parameters.
    train_loss = train()

    # Evaluate the updated model.
    prediction, eval_loss = evaluate()

    # Display the prediction and loss after each epoch.
    print(f"Epoch {epoch + 1}: Prediction = {prediction:.4f}, Loss = {eval_loss:.4f}") # .4f means display a number with 4 digits after the decimal point.


# ==================================================
# 5. Final Evaluation
# ==================================================

# Get the final prediction and loss.
prediction, loss = evaluate()

print("\n=== Final Evaluation ===")
print("Predicted Score:", round(prediction, 2)) # Round to 2 digits after the decimal point.
print("Actual Score:", actual_score.item())
print("Final Loss:", round(loss, 4)) # Round to 4 digits after the decimal point.
