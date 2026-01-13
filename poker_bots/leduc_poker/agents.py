import random

class Agent:
    def act(self, state):
        """
        state: dict {
            'card': (rank, suit),
            'board': (rank, suit) or None,
            'history': [actions],
            'pot': float,
            'round': int
        }
        Returns: 0 (FOLD), 1 (CHECK/CALL), 2 (BET/RAISE)
        """
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, env_ref=None):
        self.env = env_ref # Reference to env to check legal actions if needed
        
    def act(self, state):
        # In a real scenario, we should ask the env for legal actions
        # But for this simple interface, we'll assume the agent knows or tries
        # For robustness, let's assume we can access legal actions from the env if provided
        # Or just return a random action and hope it's legal (bad practice but simple start)
        # Ideally, we pass legal_actions into act() or have state include it.
        
        # Let's assume the agent just picks 1 or 2 mostly, 0 sometimes.
        # But we need to know what is legal.
        # A proper Agent interface usually gets (state, legal_actions)
        # Let's update this later. For now, we will update Arena to pass legal actions?
        # Or simpler: RandomAgent just guesses.
        
        return random.choice([0, 1, 2])

class HeuristicAgent(Agent):
    """
    Simple rule-based Leduc agent:
    - If Pair (Card matches Board): Always Raise (2)
    - If High Card (King): Call (1) or Raise (2)
    - If Low Card (Jack): Check/Fold (0) or Call (1)
    """
    def act(self, state):
        card_rank = state['card'][0]
        board = state['board']
        
        # Round 2: We have a board card
        if board is not None:
            board_rank = board[0]
            if card_rank == board_rank:
                # We have a pair! Raise!
                return 2
        
        # Round 1 or no pair
        if card_rank == 2: # King
            return random.choice([1, 2]) # Aggressive
        elif card_rank == 1: # Queen
            return 1 # Passive
        else: # Jack
            # If facing a bet (how do we know? History), fold.
            # Simplified: Random Check or Fold
            return random.choice([0, 1])


