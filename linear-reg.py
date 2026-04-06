# %% Cell 1
# pytorch core functionallity
import torch
# components for building neural networks
from torch.hub import load_state_dict_from_url
import torch.nn as nn
# tools for training those networks
import torch.optim as optim

# %% Cell 2
# Distance in miles
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Delivery time in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# %% Cell 3
# Defining the model
# Sequential is a container that passes data through layers in order, but makes 
# it easy for you to swap components without all of that messy rewiring.
model = nn.Sequential(nn.Linear(1, 1))

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # 0. Reset the optimizer
    optimizer.zero_grad()
    # 1. Make predictions
    outputs = model(distances)
    # 2. Calculate the loss - how bad was this guess?
    loss = loss_function(outputs, times)
    # 3. Calculate adjustments
    loss.backward()
    # 4. Update the model
    optimizer.step()
    
with torch.no_grad():
    test_distance = torch.tensor([[25.0]], dtype=torch.float32)
    predicted_time = model(test_distance)
    print(f"Predicted time for 25 miles: {predicted_time.item():.1f} minutes")
