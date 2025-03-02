import torch.nn as nn
from utils import CONTEXT_SIZE, alphabet

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(CONTEXT_SIZE * 2 * len(alphabet), 32)  # Input: ??, Hidden: 64
        self.fc2 = nn.Linear(32, 8)  # Input: ??, Hidden: 64
        self.fc3 = nn.Linear(8, 1)  # Input: ??, Hidden: 64
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x