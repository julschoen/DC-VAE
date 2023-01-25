import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim//2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
