import random

class Agent:
    def act(self, state):
        """
        state: (card, history_string)
        Returns: 0 (PASS) or 1 (BET)
        """
        raise NotImplementedError

class RandomAgent(Agent):
    def act(self, state):
        return random.choice([0, 1])

class HeuristicAgent(Agent):
    """
    Simple rule-based agent:
    - King (2): Always Bet/Call (Action 1)
    - Jack (0): Always Check/Fold (Action 0)
    - Queen (1): Random
    """
    def act(self, state):
        card, history = state
        if card == 2: return 1
        if card == 0: return 0
        return random.choice([0, 1])

