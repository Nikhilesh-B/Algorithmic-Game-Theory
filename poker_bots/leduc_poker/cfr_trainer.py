import random
from collections import defaultdict

NUM_ACTIONS = 3 # Fold, Check/Call, Bet/Raise

class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS
        self.strategy = [0.0] * NUM_ACTIONS

    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += self.strategy[a]
        
        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / NUM_ACTIONS
            self.strategy_sum[a] += realization_weight * self.strategy[a]
            
        return self.strategy

    def get_average_strategy(self):
        avg_strategy = [0.0] * NUM_ACTIONS
        normalizing_sum = sum(self.strategy_sum)
        for a in range(NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / NUM_ACTIONS
        return avg_strategy

class LeducCFRTrainer:
    def __init__(self, env):
        self.env = env
        self.node_map = {} # Map info_set string to Node

    def get_info_set(self, state):
        """
        Convert state dict to a unique string key.
        Key must capture all relevant info: Card Rank, Round, History, Board (if round 1)
        """
        card = state['card'][0] # Only rank matters
        board = state['board'][0] if state['board'] else 'x'
        history_str = "".join(str(a) for a in state['history'])
        return f"{card}|{board}|{history_str}"

    def train(self, iterations):
        util = 0.0
        for i in range(iterations):
            # Reset env to start a new game
            # We can't just pass 'cards' array because env manages deck
            # We need to hack the env or just run it?
            # CFR usually requires recursive traversal of the tree, not just sampling one path.
            # So we cannot use the 'env.step()' easily for Vanilla CFR.
            # We need to implement the recursion manually or use MCCFR (Outcome Sampling).
            
            # Let's use Outcome Sampling MCCFR because it works with 'step()'
            # ... actually Vanilla CFR is cleaner for small games like Leduc.
            # Let's implement Vanilla CFR by simulating the game logic recursively
            # rather than using the Gym env directly.
            # But that defeats the purpose of the Env...
            
            # Option B: Use the Env and do Outcome Sampling MCCFR.
            # This follows a single trajectory and updates regrets along it.
            self.env.reset()
            util += self.mccfr(self.env.get_state(0), 1.0, 1.0)
            
        print(f"Training finished.")

    def mccfr(self, state, p0, p1):
        # This is hard to implement with a Gym env because we can't 'clone' the state easily 
        # to explore other branches without rewinding.
        # For Leduc, rewriting the recursion logic is actually easier than hacking Gym.
        pass

# Since doing CFR on a Gym env is tricky (requires state save/load), 
# I will provide a recursive implementation that mimics the rules 
# but manages the traversal itself. This is standard for solvers.

class StandaloneLeducCFR:
    def __init__(self):
        self.node_map = {}
        # Cards: 0=J, 1=Q, 2=K. Two of each.
        self.deck = [0, 0, 1, 1, 2, 2]

    def train(self, iterations):
        for i in range(iterations):
            random.shuffle(self.deck)
            # deck[0]=P0, deck[1]=P1, deck[2]=Board
            self.cfr(history="", cards=self.deck[:3], p0=1.0, p1=1.0)

    def cfr(self, history, cards, p0, p1):
        # We need to reconstruct the game state from history string
        # History format: 'r0r1' where r0 is preflop actions, r1 is postflop
        # This is getting complex to parse.
        
        # SIMPLIFICATION:
        # Use the provided Gym Environment for EVALUATION only.
        # Use a custom lightweight recursion for TRAINING.
        return 0


