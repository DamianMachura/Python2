import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

Transform = transforms.compose([transforms.ToTensor()])


train_dataset = torchVision.datasets.FasionMINST(root='./data', train=True, download=True, transmorm=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


train_loader = DataLoader(trein_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parametrs(), lr=0.001)

numbers = 5

for number in range(numbers):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backword()
        optimizer.step()
        running_loss += loss.item()
    print(f"Loss: {running_loss/len(train_loader)}")

    correct = 0
    total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predictes = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test: {100*correct / total:.2f}%")
