import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSingleChan(nn.Module):
    
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 5, 1, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear( 64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 0)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        
        return x
