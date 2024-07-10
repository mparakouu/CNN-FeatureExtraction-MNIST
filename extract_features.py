import numpy as np
from skimage.feature import hog
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

def extract_features(data):
    features = [] # save features 
    for image in data:
        image = image.numpy() # image --> numpy array 
        # 4 x 4 pixel , # 2 x 2 block (4 cells)
        hog_feature, hog_image = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
        features.append(hog_feature) # add hog features 
    return np.array(features)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1296, 128)  # input 1296 HOG features --> output 128
        self.fc2 = nn.Linear(128, 10)    # 10 classes output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train() # train model
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features) # predictions based on input data
            loss = criterion(outputs, targets) # loss -> predictions vs real
            loss.backward() # gradients
            optimizer.step() # resfresh weights, from gradients 
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        val_loss = validate(model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'hog_model.pth')

    print(f'Best Validation Loss: {best_val_loss:.4f}')


def validate(model, val_loader, criterion):
    model.eval() # no train
    val_loss = 0
    with torch.no_grad(): # no gradients cal
        for features, targets in val_loader:
            outputs = model(features) # predictions based on input data
            val_loss += criterion(outputs, targets).item() # cal loss
    # val_loss / num of batches 
    val_loss /= len(val_loader) # average loss
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss



if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()]) # --> tensor
    training_dataset = datasets.MNIST(root='../Thedata', train=True, download=True, transform=transform) # load
    testing_dataset = datasets.MNIST(root='../Thedata', train=False, download=True, transform=transform) # load

    train_data_image = training_dataset.data # image from the train dataset
    test_data_image = testing_dataset.data # image from the test dataset

    # extract features from the images (traing & test)
    train_features = extract_features(train_data_image)
    test_features = extract_features(test_data_image)

    np.save('train_features.npy', train_features)
    np.save('test_features.npy', test_features)

    print(f"Train Features Shape: {train_features.shape}")
    print(f"Test Features Shape: {test_features.shape}")

    train_features = np.load('train_features.npy')
    test_features = np.load('test_features.npy')

    # --> tensors 
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)

    # TensorDataset : features (inputs) , targets (outputs)
    train_dataset = TensorDataset(train_features_tensor, training_dataset.targets)
    test_dataset = TensorDataset(test_features_tensor, testing_dataset.targets)

    # 80% : training and 20% : validate
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Stochastic Gradient Descent algorithm

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

