import torch
import numpy as np
import random
import os
from .game import NLHERules, GameConstants
from .models import NLHEValueNetwork
from .search import NLHESearch
from .features import get_nlhe_features

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.data = []
    def push(self, features, targets):
        self.data.append((features, targets))
        if len(self.data) > 1000:
            self.data.pop(0)
    def sample(self, batch_size):
        batch = random.sample(self.data, min(len(self.data), batch_size))
        feats, targs = zip(*batch)
        return torch.stack(feats), torch.stack(targs)
    def __len__(self):
        return len(self.data)

def train(epochs=5, steps_per_epoch=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    model = NLHEValueNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = ReplayBuffer()
    search = NLHESearch(model, device=device)
    
    # Check if model exists
    if os.path.exists("rebel_nlhe.pt"):
        print("Loading existing model...")
        model.load_state_dict(torch.load("rebel_nlhe.pt", map_location=device))
    
    model.train()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 1. Self Play Data Gen
        # Generate random states
        for _ in range(steps_per_epoch):
            # Init random beliefs
            r0 = np.random.dirichlet(np.ones(1326), size=1)[0]
            r1 = np.random.dirichlet(np.ones(1326), size=1)[0]
            
            board = [] # Empty board for now
            # Random short history (actions 0-3)
            hist_len = random.randint(0, 6)
            history = [random.randint(0, 3) for _ in range(hist_len)]
            pot = float(random.randint(150, 2000))
            stacks = [20000.0 - pot/2, 20000.0 - pot/2]
            
            # Run Search
            strat, values = search.solve_subgame(r0, r1, board, pot, stacks, history)
            
            # Prepare Training Data
            ft = get_nlhe_features(r0, r1, board, pot, stacks, history).to(device)
            
            # Target: Concatenate P0 and P1 values
            # values[0] is (1326,), values[1] is (1326,)
            target = np.concatenate([values[0], values[1]])
            target_t = torch.tensor(target, dtype=torch.float32).to(device)
            
            buffer.push(ft, target_t)
            
        # 2. Train
        if len(buffer) > 10:
            for _ in range(5): # Gradient steps
                feats, targs = buffer.sample(32)
                feats = feats.to(device)
                targs = targs.to(device)
                
                preds = model(feats)
                loss = torch.mean((preds - targs) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), "rebel_nlhe.pt")
    print("Model saved to rebel_nlhe.pt")

if __name__ == "__main__":
    train()
