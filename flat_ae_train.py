import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from flat_ae import Autoencoder

inner_dim = 250

# Load train data
train = pd.read_csv('data/sensor_train.csv')
train = train.sort_index(axis=1)
train_groups = train['group_index'].unique()

# Load test data
test = pd.read_csv('data/sensor_test.csv')
test = test.sort_index(axis=1)
test_groups = test['group_index'].unique()

# Get sensor scales
sensor_cols = [col for col in train.columns if 'sensor_' in col]
max_sensor_values = train[sensor_cols].max().values

# Init model, criterion, optimizers
model = Autoencoder(500*52, inner_dim, max_sensor_values)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5, gamma=0.75)

num_epochs = 20

# Data savers
epochs = []
train_avg_loss = []
test_avg_loss = []

# Train
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}",end='\n')
    model.train()
    train_loss = 0
    for group in train_groups:
        df = train[train['group_index'] == group]

        input = torch.tensor(df[sensor_cols].to_numpy() / max_sensor_values, dtype=torch.float32)
        input = torch.nan_to_num(input, nan=0.0)
        input = input.reshape(1, -1)

        decoded = model(input)

        train_loss += criterion(decoded, input)


    model.eval()
    test_loss = 0
    for group in test_groups:
        df = test[test['group_index'] == group]

        input = torch.tensor(df[sensor_cols].to_numpy() / max_sensor_values, dtype=torch.float32)
        input = torch.nan_to_num(input, nan=0.0)
        input = input.reshape(1, -1)

        decoded = model(input)

        test_loss += criterion(decoded, input)

    epochs.append(epoch)
    train_avg_loss.append(train_loss.item() / len(train_groups))
    test_avg_loss.append(test_loss.item() / len(test_groups))

    print(f"train loss: {train_loss.item()/len(train_groups)}, test loss: {test_loss.item()/len(test_groups)}")
    train_loss.backward()
    optimizer.step()
    scheduler.step()

torch.save(model, f'models/flat_ae_{inner_dim}.pt')

# Plot results
plt.figure()
plt.plot(epochs, train_avg_loss, '-', label='Train avg loss')
plt.plot(epochs, test_avg_loss, '-', label='Test avg loss')
plt.show()