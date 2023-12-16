import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.tri import Triangulation

# Load MATLAB file
mat_data = scipy.io.loadmat('cylinder_nektar_t0_vorticity.mat')

# Define the LEMCell
class LEMCell(nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMCell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.dt = dt
        self.inp2hid = nn.Linear(ninp, 4 * nhid)
        self.hid2hid = nn.Linear(nhid, 3 * nhid)
        self.transform_z = nn.Linear(nhid, nhid)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.nhid)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, y, z):
        transformed_inp = self.inp2hid(x)
        transformed_hid = self.hid2hid(y)
        i_dt1, i_dt2, i_z, i_y = transformed_inp.chunk(4, 1)
        h_dt1, h_dt2, h_y = transformed_hid.chunk(3, 1)

        ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1)
        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2)

        z = (1. - ms_dt) * z + ms_dt * torch.tanh(i_y + h_y)
        y = (1. - ms_dt_bar) * y + ms_dt_bar * torch.tanh(self.transform_z(z) + i_z)

        return y, z

# Define the LEM model
class LEM(nn.Module):
    def __init__(self, ninp, nhid, nout, dt=1.):
        super(LEM, self).__init__()
        self.nhid = nhid
        self.cell = LEMCell(ninp, nhid, dt)
        self.classifier = nn.Linear(nhid, nout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)

    def forward(self, input):
        y = input.data.new(input.size(1), self.nhid).zero_()
        z = input.data.new(input.size(1), self.nhid).zero_()
        for x in input:
            y, z = self.cell(x, y, z)
        out = self.classifier(y)
        return out

# Set random seed for reproducibility
torch.manual_seed(42)

w = mat_data['w'].reshape(-1,)
w = w[::200]
# Toy problem data
input_size = 1
hidden_size = 32
output_size = 1
sequence_length = 249
batch_size = 1
num_epochs = 20000

# Generate sine wave data
w_train = w[:40000]
w_test = w[40000:]

# Split data into input and target sequences
input_data = w_train[:-1]
target_data = w_train[1:]

# Convert data to tensors
input_tensor = torch.tensor(input_data).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(target_data).view(batch_size, sequence_length, output_size).float()

# Create LEM instance
lem = LEM(input_size, hidden_size, output_size, dt=0.1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lem.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output = lem(input_tensor)
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')


quit()
# Generate predictions on new range of values
x_pred = np.linspace(0, 4 * np.pi, len(target_data))
input_pred = np.sin(x_pred)

# Convert data to tensor for prediction
input_tensor_pred = torch.tensor(input_pred).view(batch_size, -1, input_size).float()

with torch.no_grad():
    prediction = lem(input_tensor_pred)

# Flatten prediction tensor
prediction = prediction.view(-1).numpy()

# Plot the results
plt.plot(x_train[:-1], input_data, label='Input Sequence (Train)')
plt.plot(x_train[1:], target_data, label='Target Sequence (Train)')
plt.plot(x_pred, prediction, label='Predicted Sequence (Prediction)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
