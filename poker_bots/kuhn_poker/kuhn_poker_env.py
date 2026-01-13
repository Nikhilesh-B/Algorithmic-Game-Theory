import random
from enum import Enum

class Action(Enum):
    PASS = 0  # Check or Fold
    BET = 1   # Bet or Call

class KuhnPokerEnv:
    def __init__(self):
        self.deck = [0, 1, 2] # J, Q, K
        self.reset()
        
    def reset(self):
        random.shuffle(self.deck)
        self.hands = {0: self.deck[0], 1: self.deck[1]}
        self.history = [] # List of actions (0 or 1)
        self.player_turn = 0
        self.done = False
        self.pot = 2 # Ante 1 each
        return self.get_state(0)
        
    def get_state(self, player_id):
        """
        Returns a tuple representing the state for a specific player.
        Format: (MyCard, HistoryString)
        """
        hist_str = "".join(str(a) for a in self.history)
        return (self.hands[player_id], hist_str)
        
    def step(self, action):
        """
        Executes an action (0 for PASS, 1 for BET).
        Returns: (state_next_player, reward_dict, done)
        """
        self.history.append(action)
        
        # Check for terminal states
        if self._is_terminal():
            self.done = True
            payoffs = self._calculate_payoffs()
            return None, payoffs, True
            
        self.player_turn = 1 - self.player_turn
        return self.get_state(self.player_turn), {0:0, 1:0}, False
        
    def _is_terminal(self):
        h = self.history
        if len(h) == 2 and h[0] == 0 and h[1] == 0: return True # Check-Check
        if len(h) == 2 and h[0] == 1: return True # Bet-Fold or Bet-Call
        if len(h) == 3: return True # Check-Bet-Fold or Check-Bet-Call
        return False

    def _calculate_payoffs(self):
        # Returns dictionary {0: score, 1: score}
        h = self.history
        p0 = self.hands[0]
        p1 = self.hands[1]
        
        # Determine winner by card rank
        winner = 0 if p0 > p1 else 1
        
        # history is list of ints [0, 1, ...]
        
        # Case: Check-Check [0, 0] -> Showdown, win 1
        if h == [0, 0]:
            return {winner: 1, 1-winner: -1}
            
        # Case: Bet-Fold [1, 0] -> Player 0 bet, Player 1 folded. P0 wins 1.
        if h == [1, 0]:
            return {0: 1, 1: -1}
            
        # Case: Bet-Call [1, 1] -> Showdown, win 2
        if h == [1, 1]:
            return {winner: 2, 1-winner: -2}
            
        # Case: Check-Bet-Fold [0, 1, 0] -> P0 checked, P1 bet, P0 folded. P1 wins 1.
        if h == [0, 1, 0]:
            return {0: -1, 1: 1}
            
        # Case: Check-Bet-Call [0, 1, 1] -> Showdown, win 2
        if h == [0, 1, 1]:
            return {winner: 2, 1-winner: -2}
            
        return {0:0, 1:0}

