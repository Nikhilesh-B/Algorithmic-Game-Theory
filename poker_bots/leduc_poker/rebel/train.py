import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .game import GameConstants, LeducRules
from .models import ValueNetwork
from .search import CFRSolver
from .features import get_features

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, features, values):
        self.buffer.append((features, values))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        features, values = zip(*batch)
        return torch.stack(features), torch.stack(values)
        
    def __len__(self):
        return len(self.buffer)

class ReBeLTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.value_net = ValueNetwork().to(device)
        self.target_net = ValueNetwork().to(device) # For stability? Optional.
        self.target_net.load_state_dict(self.value_net.state_dict())
        
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()
        self.solver = CFRSolver(self.target_net, iterations=100, device=device) # Use target net for search
        
    def generate_data(self, num_games=1):
        self.value_net.eval()
        for _ in range(num_games):
            self._play_one_game()
            
    def _play_one_game(self):
        # Initialize
        deck = [(r, s) for r in range(3) for s in range(2)]
        random.shuffle(deck)
        hand_p0 = deck[0] # (rank, suit)
        hand_p1 = deck[1]
        board = deck[2]
        
        # Indices
        c0 = hand_p0[0] * 2 + hand_p0[1]
        c1 = hand_p1[0] * 2 + hand_p1[1]
        b_rank = board[0] # Just rank for now
        
        # Initial State
        history = []
        bets = {0: 1.0, 1: 1.0}
        board_state = None
        
        # Initial Beliefs (Uniform over 6 cards)
        # But wait, P1 knows P0 doesn't have c1? No, private info.
        # Public belief is Uniform.
        r0 = np.ones(6) / 6.0
        r1 = np.ones(6) / 6.0
        
        # Play until terminal
        while True:
            # Check if round end
            if LeducRules.is_terminal_round(history):
                # Transition or End Game
                if board_state is None:
                    # Transition to Round 2
                    board_state = b_rank
                    history = [] # Reset history for R2
                    # Raises reset
                    
                    # Update beliefs for board card
                    # Zero out the board card index
                    # Which specific card is the board?
                    # The 'board' variable holds it.
                    b_idx = board[0] * 2 + board[1]
                    r0[b_idx] = 0
                    r1[b_idx] = 0
                    # Renormalize
                    r0 /= r0.sum()
                    r1 /= r1.sum()
                    
                    # Continue loop (Round 2 start)
                    continue
                else:
                    # Game Over
                    break
            
            # Check Fold (should be caught by is_terminal_round actually)
            if len(history) > 0 and history[-1] == 0:
                break
                
            # Run Search
            # solve returns avg_strat (dict action->prob_vec) and value (dict player->val_vec)
            # We want to store (State, Value)
            # State = (r0, r1, board, history, pot)
            pot = bets[0] + bets[1]
            
            # Prepare tensors for search
            t_r0 = torch.tensor(r0, dtype=torch.float32).to(self.device)
            t_r1 = torch.tensor(r1, dtype=torch.float32).to(self.device)
            
            avg_strat, values = self.solver.solve(
                history, board_state, bets, t_r0, t_r1
            )
            
            # Store Data
            # Store features and VALUES.
            # Which values? The search returns EV for both players.
            # V0 is vector (6,), V1 is vector (6,).
            # Concatenate to (12,).
            val_vec = np.concatenate([values[0], values[1]])
            
            ft = get_features(t_r0, t_r1, board_state, history, pot)
            self.buffer.push(ft.cpu(), torch.tensor(val_vec, dtype=torch.float32))
            
            # Sample Action
            active = len(history) % 2
            my_card = c0 if active == 0 else c1
            my_strat = avg_strat # map action -> vector
            
            # Get probs for my card
            probs = []
            actions = list(my_strat.keys())
            for a in actions:
                probs.append(my_strat[a][my_card])
            
            # Sample
            if sum(probs) < 1e-9:
                a = random.choice(actions)
            else:
                a = random.choices(actions, weights=probs)[0]
            
            # Update Beliefs
            # r_active = r_active * strat[a]
            if active == 0:
                r0 = r0 * my_strat[a]
                if r0.sum() > 0: r0 /= r0.sum()
            else:
                r1 = r1 * my_strat[a]
                if r1.sum() > 0: r1 /= r1.sum()
                
            # Update History/Bets
            # Logic from search
            opponent = 1 - active
            if a == 1:
                diff = bets[opponent] - bets[active]
                if diff > 0: bets[active] += diff
            elif a == 2:
                diff = bets[opponent] - bets[active]
                amount = 4.0 if board_state is not None else 2.0
                bets[active] += diff + amount
                
            history.append(a)
            
    def train(self, batch_size=32, steps=100):
        self.value_net.train()
        losses = []
        for _ in range(steps):
            if len(self.buffer) < batch_size:
                continue
                
            features, targets = self.buffer.sample(batch_size)
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            preds = self.value_net(features)
            loss = torch.mean((preds - targets) ** 2)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        # Update target net periodically
        self.target_net.load_state_dict(self.value_net.state_dict())
        return np.mean(losses) if losses else 0.0


