import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import SubsetRandomSampler
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

    def forward(self, x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.pooling(x)
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

def sample_of_classes(dataset):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.figure(figsize=(10, 5))
    for k in range(len(classes)):
        plt.subplot(2, 5, k + 1)
        sample_image = dataset.data[dataset.targets == k][0]
        plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
        plt.title(f'Class {classes[k]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
sample_of_classes(dataset)

# train : 80% , validation : 100 - 80 = 20%
training = int(0.8 * len(dataset)) 
validation = len(dataset) - training
training_subset, validation_subset = random_split(dataset, [training, validation])

print(f"Training: {len(training_subset)}")
print(f"Validation: {len(validation_subset)}")

# Dataloaders
trainData_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
valData_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
best_val_loss = float('inf')


model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

# train the NN
def train(model, device, epoch, trainData_loader, optimizer):
    model.train() # train model
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainData_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) # predictions based on input data
        loss = criterion(output, target) # loss -> predictions vs real
        loss.backward() # gradients
        optimizer.step() # resfresh weights, from gradients 
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Training epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainData_loader.dataset),
                100. * batch_idx / len(trainData_loader), loss.item()))
    return train_loss / len(trainData_loader)

# check how it works after training 
def validate(model, device, valData_loader):
    model.eval() # no train
    val_loss = 0
    with torch.no_grad(): # no gradients cal
        for data, target in valData_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # predictions based on input data
            val_loss += criterion(output, target).item() # cal loss

    # val_loss / num of batches 
    val_loss /= len(valData_loader) # average loss
    print('\nAV loss: {:.4f}\n'.format(val_loss))
    return val_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# save train , val losses
training_losses = []
validate_losses = []

# initialize losses to compare
best_training = float('inf')
best_validate = float('inf')

# train model for each epoch
for epoch in range(1, num_epochs + 1): # 1 to 10
    train_loss = train(model, device, epoch, trainData_loader, optimizer) # av. loss during training in every epoch
    val_loss = validate(model, device, valData_loader) # av. loss during validation in every epoch 

    # compare current epoch with previous and save the epoch with < loss 
    if train_loss < best_training:
        print('Training loss decreased from {:.4f} to {:.4f}.'.format(best_training, train_loss))
        best_training = train_loss
        torch.save(model.state_dict(), 'best_training.pth')
    
    if val_loss < best_validate:
        print('Validation loss decreased from {:.4f} to {:.4f}.'.format(best_validate, val_loss))
        best_validate = val_loss
        torch.save(model.state_dict(), 'best_validate.pth')

    # adding losses in list
    training_losses.append(train_loss)
    validate_losses.append(val_loss)

test_dataset = datasets.MNIST(root='../Thedata', train=False, download=True, transform=transform)
testData_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # test data in batches

plt.figure(figsize=(10, 7))
plt.plot(range(1, num_epochs + 1), training_losses, label='training loss', color='purple')
plt.plot(range(1, num_epochs + 1), validate_losses, label='validate loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('training & validation losses')
plt.legend()
plt.show()



def accurancy_ConfMatrix(model, device, testData_loader):
    model.eval()
    total_number = 0  # total num of examples
    correct_predictions = 0  # correct predictions
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in testData_loader: # data , "real" labels --> save
            data, target = data.to(device), target.to(device)
            outputs = model(data)  # data through model 
            _, predicted = torch.max(outputs.data, 1)  # highest / predicted  
            total_number += target.size(0)  # ++
            correct_predictions += (predicted == target).sum().item()  # num of correct predictions --> add to correct_predictions
            all_targets.extend(target.cpu().numpy()) # real  "labels"
            all_predictions.extend(predicted.cpu().numpy()) # data

    accuracy = correct_predictions / total_number 
    print('Accuracy of the Neural Network: {:.2f}%'.format(accuracy * 100))


    ConfMatrix = confusion_matrix(all_targets, all_predictions)
    return accuracy, ConfMatrix

accuracy, ConfMatrix = accurancy_ConfMatrix(model, device, testData_loader)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{ConfMatrix}")



