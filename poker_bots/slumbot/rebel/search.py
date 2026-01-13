import torch
import numpy as np
import random
from .game import NLHERules, GameConstants
from .features import get_nlhe_features

class NLHESearch:
    def __init__(self, value_net, device='cpu'):
        self.value_net = value_net
        self.device = device
        self.all_hands = NLHERules.get_all_hands() # List of 1326 tuples
        
    def solve_subgame(self, r0, r1, board, pot, stacks, history):
        """
        Runs a 1-ply search using the Value Net with softmax action selection.
        Returns:
        - Strategy (Action -> Prob Vector over 1326 hands)
        - Value (Vector over 1326 hands)
        """
        # Identify legal actions
        # history for rule check needs to be list of ints?
        # get_legal_actions(history, pot, current_bets, stack_sizes)
        # We assume history is sufficient to determine current bets if we tracked them.
        # For simplicity in this prototype, assume equal bets unless history says otherwise.
        # But 'history' passed here is simplified list of ints?
        # Let's just assume we can always Check/Call, Pot, AllIn for now.
        
        # Simplified Legal Actions
        legal = [0, 1, 2, 3] # Fold, Check/Call, Pot, AllIn
        
        # We need to query ValueNet for each action's resulting state.
        # Next State construction is complex (bets update, pot update, etc).
        # We will approximate:
        # Action 0 (Fold): Terminal. Value is Pot/2 or 0 depending on who folds.
        # Action 1 (Check/Call): Next state (Next Round or Showdown?)
        # Action 2 (Pot): Next state (Opponent to act, pot increased)
        # Action 3 (AllIn): Next state (Opponent to act, pot increased)
        
        # For this prototype, we will just query the Value Net for the CURRENT state
        # and assume the "Policy" is implicit in the value outputs?
        # No, ReBeL Value Net outputs Value of the state.
        # We need Q(s, a).
        # Q(s, a) = Value(NextState(s, a)).
        
        # Batch query for all actions
        inputs = []
        
        # Current Features (for reference, mostly next state matters)
        # ft_curr = get_nlhe_features(r0, r1, board, pot, stacks, history)
        
        # Simulate Next States
        # 1. Fold
        # Value is fixed. If I fold, I lose pot. Payoff = -Contribution?
        # Value = 0 (relative to pot?)
        # Let's say Value = -Pot/2 (Loss).
        
        # 2. Check/Call
        # Pot same (Check) or Pot increases (Call).
        # History appends 1.
        # Next state features?
        # r0, r1 same.
        
        # 3. Pot Bet
        # Pot increases by Pot.
        # History appends 2.
        
        # 4. All In
        # Pot increases by Stack.
        # History appends 3.
        
        # We need to construct tensors for Next States.
        # Note: If it's opponent's turn in next state, the ValueNet returns EV for P0, P1.
        # We need to handle perspective.
        
        next_states = []
        
        # Dummy "Next State" Logic
        # (Real implementation needs full game engine)
        
        for a in [1, 2, 3]:
            # Construct features
            # Just append 'a' to history and update pot roughly
            new_hist = history + [a]
            new_pot = pot
            if a == 2: new_pot *= 2
            if a == 3: new_pot += 20000
            
            ft = get_nlhe_features(r0, r1, board, new_pot, stacks, new_hist)
            next_states.append(ft)
            
        # Query
        if next_states:
            batch = torch.stack(next_states).to(self.device)
            with torch.no_grad():
                # Shape (3, 2652) -> (3, 2, 1326)
                # Output head is 2652 dims.
                out = self.value_net(batch)
                out = out.view(-1, 2, 1326) # (Batch, Player, Hand)
                
            # Extract Values
            # vals[action_idx][player_idx][hand_idx]
            vals_call = out[0].cpu().numpy()
            vals_pot = out[1].cpu().numpy()
            vals_allin = out[2].cpu().numpy()
        else:
            # Should not happen
            pass
            
        # Construct Q-Values for P0 (assuming we are P0)
        # Fold: 0 (Simplified)
        # Call: vals_call[0]
        # Pot: vals_pot[0]
        # AllIn: vals_allin[0]
        
        # Determine best action per hand
        strategy = {
            0: np.zeros(1326),
            1: np.zeros(1326),
            2: np.zeros(1326),
            3: np.zeros(1326)
        }
        
        # Maximize Value
        # Fold value? Let's say -10.
        fold_val = np.full(1326, -10.0) 
        
        # Compare
        # For each hand i: softmax over q_vals for a smoother policy.
        temp = 1.0
        for i in range(1326):
            q_vals = np.array([
                fold_val[i],
                vals_call[0][i],
                vals_pot[0][i],
                vals_allin[0][i]
            ])
            # Softmax with clipping for stability
            q_clip = np.clip(q_vals / max(1e-3, temp), -50, 50)
            exp_q = np.exp(q_clip - q_clip.max())
            probs = exp_q / exp_q.sum()
            for a_idx, p in enumerate(probs):
                strategy[a_idx][i] = p
            
        # Value of this node (for training parent)
        node_values_p0 = np.zeros(1326)
        node_values_p1 = np.zeros(1326) # Opponent value?
        
        for i in range(1326):
            q_vals = np.array([fold_val[i], vals_call[0][i], vals_pot[0][i], vals_allin[0][i]])
            # Expected value under the softmax strategy
            p = np.array([strategy[a][i] for a in range(4)])
            node_values_p0[i] = np.dot(p, q_vals)
            
            # For P1, use the same mixture to estimate expected value
            # Fold gives P1 win; others from value net
            opp_q = np.array([
                10.0,                    # if P0 folds
                vals_call[1][i],
                vals_pot[1][i],
                vals_allin[1][i],
            ])
            node_values_p1[i] = np.dot(p, opp_q)
                
        return strategy, {0: node_values_p0, 1: node_values_p1}

    def get_action_from_strategy(self, strategy, hand_card_ints):
        # hand_card_ints: tuple of 2 ints
        # Find index in self.all_hands
        try:
            # Sort needed because all_hands are generated sorted
            key = tuple(sorted(hand_card_ints))
            idx = self.all_hands.index(key)
        except ValueError:
            # Fallback
            idx = 0
        
        probs = []
        actions = sorted(list(strategy.keys()))
        for a in actions:
            probs.append(strategy[a][idx])
            
        if sum(probs) == 0:
            return random.choice(actions)
            
        return random.choices(actions, weights=probs)[0]
