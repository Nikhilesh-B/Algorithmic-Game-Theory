import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, input_dim=47, hidden_dim=128):
        """
        Input Features:
        - P0 Range (6)
        - P1 Range (6)
        - Board State (4: None, J, Q, K)
        - Pot Size (1)
        - History Embedding (30: 10 actions * 3 types)
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 12) # Value vector for P0 (6) and P1 (6)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v

