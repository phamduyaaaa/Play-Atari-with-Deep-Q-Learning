import torch.nn as nn
import torch.nn.functional as F
import torch

class DQNetwork(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=32*9*9, out_features=256)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(in_features=256, out_features=int(num_actions))
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = DQNetwork(num_actions=4)
    print(model)
    input_random = torch.rand(1, 4, 84, 84)
    output = model(input_random)
    print(output.shape)
