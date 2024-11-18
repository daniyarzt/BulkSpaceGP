import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from utilities import get_hessian_eigenvalues
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image to a 784-dimensional vector
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_size = 128     # Number of units in the hidden layer
output_size = 10      # Number of classes for MNIST digits (0–9)
batch_size = 64
learning_rate = 0.01
num_epochs_gd = 10
num_epochs_bulk = 0

# Check for CUDA availability
device = torch.device("cuda")

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
print(len(full_train_dataset))

# Create a subset of 10% of the training dataset
train_size = 5000  # 10% of the data
indices = np.random.choice(len(full_train_dataset), train_size, replace=False)
train_dataset = Subset(full_train_dataset, indices)
print(indices.shape)

# Use the entire test dataset for evaluation
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MLP(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def projected_step_bulk(model, loss, criterion, dataset, batch_size):
    evals, evecs = get_hessian_eigenvalues(model, criterion, dataset, physical_batch_size=batch_size)
    evecs = evecs.to(device).transpose(1, 0)
    
    with torch.no_grad():
        grad = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        vec_grad = parameters_to_vector(grad).to(device)
        step = vec_grad.detach()
        
        for vec in evecs:
            step -= torch.dot(vec_grad, vec) * vec
        
        vec_params = parameters_to_vector(model.parameters()).to(device)
        vec_params -= step * learning_rate
        vector_to_parameters(vec_params, model.parameters())
        model.zero_grad()

# Training loop
losses = []
for epoch in range(num_epochs_gd):
    sum_loss = 0.0
    batches = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        sum_loss += loss.item()
        batches += 1

    print(f'Epoch [{epoch + 1}/{num_epochs_gd}], Loss: {(sum_loss / batches):.4f}')

# Testing loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

print('Started bulk-GD')
for epoch in range(num_epochs_bulk):
    sum_loss = 0
    batches = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        losses.append(loss.item())
        sum_loss += loss.item()
        batches += 1
        
        dataset = TensorDataset(images, labels)
        projected_step_bulk(model, loss, criterion, dataset, batch_size=batch_size)

    print(f'Epoch [{epoch + 1}/{num_epochs_bulk}], Loss: {(sum_loss / batches):.4f}')

# Testing loop after bulk-GD
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set after bulk-GD: {100 * correct / total:.2f}%')
