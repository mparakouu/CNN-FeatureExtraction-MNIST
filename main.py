import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 64
learning_rate = 0.001
num_epochs = 10
 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=0)
        self.fullyConnected1 = nn.Linear(in_features=12*4*4, out_features=10)  

    def throughNN(self, x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = x.view(-1, 12*4*4)  
        output = self.fullyConnected1(x)
        return output

# load & transform dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load dataset
dataset = datasets.MNIST(root='../Thedata', train=True, download=True, transform=transform)

# train : 80% , validation : 100 - 80 = 20%
training = int(0.8 * len(dataset)) 
validation = len(dataset) - training
training_subset, validation_subset = random_split(dataset, [training, validation])

print(f"Training: {len(training_subset)}")
print(f"Validation: {len(validation_subset)}")

# Dataloaders
trainData_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
valData_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)


model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(args, model, epoch, device, trainData_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(trainData_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) # predictions based on input data
        loss = F.nll_loss(output, target) # loss -> predictions vs real
        loss.backward() # gradients
        optimizer.step() # resfresh weights, from gradients 
        if batch_idx == 0 or (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainData_loader.dataset),
                100. * batch_idx / len(trainData_loader), loss.item()))
            if args.dry_run:
                break


def sample_of_classes(dataset):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(10, 5))
    for k in range(len(classes)):
        plt.subplot(2, 5, k + 1)
        sample_image = dataset.data[dataset.targets == k][0]
        plt.imshow(sample_image, cmap='gray')
        plt.title(f'Class {classes[k]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

sample_of_classes(dataset)
