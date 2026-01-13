import torch
import torch.nn as nn
import torch.nn.functional as F

class NLHEValueNetwork(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=256):
        """
        Input Features:
        - P0 Range (1326)
        - P1 Range (1326)
        - Board (5 cards * 17 dim embedding? Or just One-Hot 52 * 5?)
        - Pot (1)
        - Stack (2) - Effective stack or both
        - History Embedding (Fixed size)
        
        Input Size:
        1326 * 2 = 2652 (Ranges)
        52 * 5 = 260 (Board - simplified, or just present cards)
        Actually, Board is 52-dim binary vector (cards on board).
        Pot + Stacks = 3 dims.
        History = 50 dims.
        Total ~ 3000.
        """
        super(NLHEValueNetwork, self).__init__()
        
        self.input_dim = 1326 + 1326 + 52 + 3 + 50
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output: Value vector for P0 (1326) and P1 (1326)
        # Predicting value per hand.
        self.output_head = nn.Linear(hidden_dim, 1326 * 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.output_head(x)
        return out

