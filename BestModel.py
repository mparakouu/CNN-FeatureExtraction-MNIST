import torch
import torch.nn as nn
import torch.nn.functional as F

# use the same NN
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

# new model
Best_model = NeuralNetwork()

# adjust the weights from the .pth files (the best performance / less loss)
state_dict_training = torch.load('best_training.pth')
state_dict_validate = torch.load('best_validate.pth')

Best_model.load_state_dict(state_dict_training)

print("Conv1.weight:", state_dict_training['Conv1.weight'])
print("Conv1.bias:", state_dict_training['Conv1.bias'])
print("Conv2.weight:", state_dict_training['Conv2.weight'])
print("Conv2.bias:", state_dict_training['Conv2.bias'])
print("fullyConnected1.weight:", state_dict_training['fullyConnected1.weight'])
print("fullyConnected1.bias:", state_dict_training['fullyConnected1.bias'])

