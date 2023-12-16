import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# Define the coRNNCell
class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

# Define the coRNN model
class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp, n_hid, dt, gamma, epsilon)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):
        hy = torch.zeros(x.size(1), self.n_hid)
        hz = torch.zeros(x.size(1), self.n_hid)

        for t in range(x.size(0)):
            hy, hz = self.cell(x[t], hy, hz)
        output = self.readout(hy)

        return output

# Toy problem data
input_size = 256
hidden_size = 32
output_size = 256
sequence_length = 75
batch_size = 1
num_epochs = 20000

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the .mat file
mat_data = scipy.io.loadmat('Data/burgers_shock.mat')

# Access the variables stored in the .mat file
# The variable names in the .mat file become keys in the loaded dictionary
x = mat_data['x']
t = mat_data['t']
u = mat_data['usol']

# Use the loaded variables as needed
print(x.shape)
print(t.shape)
print(u.shape)

X, T = np.meshgrid(x, t)
# Define custom color levels
c_levels = np.linspace(np.min(u), np.max(u), 100)

# Plot the contour
plt.figure(figsize=(15, 5))
plt.contourf(T, X, u.T, levels=c_levels, cmap='coolwarm')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Burgers')
plt.colorbar()  # Add a colorbar for the contour levels
plt.show()

input_data = u[:,0:75]
target_data = u[:,1:76]

test_data = u[:,75:99]
test_target = u[:,76:100]

print("test data shape", test_data.shape)
print("test target shape", test_target.shape)

print("input data shape",input_data.shape)
print("Target data shape",target_data.shape)

# Convert data to tensors
input_tensor = torch.tensor(input_data.T).view(batch_size, sequence_length, input_size).float()
target_tensor = torch.tensor(target_data.T).view(batch_size, sequence_length, output_size).float()

print("input tensor shape",input_tensor.shape)
print("Target tensor shape",target_tensor.shape)

# Convert test data to tensors
test_tensor = torch.tensor(test_data.T).view(batch_size, 24, input_size).float()
test_target_tensor = torch.tensor(test_target.T).view(batch_size, 24, output_size).float()

# # Generate sine wave data
# x_train = np.linspace(0, 2*np.pi, sequence_length+1)
# y_train = np.sin(x_train)
#
# # Split data into input and target sequences
# input_data = y_train[:-1]
# target_data = y_train[1:]
#
# # Convert data to tensors
# input_tensor = torch.tensor(input_data).view(batch_size, sequence_length, input_size).float()
# target_tensor = torch.tensor(target_data).view(batch_size, sequence_length, output_size).float()

# Create coRNN instance
cornn = coRNN(input_size, hidden_size, output_size, dt=0.1, gamma=1.0, epsilon=0.01)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cornn.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output = cornn(input_tensor)
    loss = criterion(output, target_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.8f}')

# Generate predictions on new range of values
#x_pred = np.linspace(0, 4*np.pi, len(input_data))
#input_pred = np.sin(x_pred)

# Convert data to tensor for prediction
#input_tensor_pred = torch.tensor(input_pred).view(batch_size, -1, input_size).float()

with torch.no_grad():
    prediction = cornn(test_tensor)

print(prediction.shape)

final_time_output = prediction[-1, :]
print(final_time_output.shape)

final_out = final_time_output.detach().numpy().reshape(-1,1)
final_true = u[:,-1].reshape(-1,1)

plt.plot(x, final_out)
plt.plot(x, final_true)
plt.show()
quit()

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