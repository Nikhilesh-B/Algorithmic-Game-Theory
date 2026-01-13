import copy

class GameConstants:
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2
    
    # Ranks
    JACK = 0
    QUEEN = 1
    KING = 2
    
    NUM_ACTIONS = 3
    NUM_RANKS = 3
    NUM_SUITS = 2
    NUM_CARDS = 6

class LeducRules:
    @staticmethod
    def get_legal_actions(history, raises_in_round):
        """
        Returns list of legal actions given the sequence of actions in the current round.
        history: list of integers (0, 1, 2)
        raises_in_round: int
        """
        acts = history
        if len(acts) == 0:
            return [1, 2] # Check, Bet
            
        last_action = acts[-1]
        
        if last_action == 2: # BET/RAISE
            # Facing a bet.
            can_raise = raises_in_round < 2
            if can_raise:
                return [0, 1, 2]
            else:
                return [0, 1]
                
        if last_action == 1: # CHECK
            return [1, 2]
            
        return []

    @staticmethod
    def is_terminal_round(history):
        if len(history) == 0:
            return False
        
        last = history[-1]
        if last == 0: # Fold
            return True
            
        # Check-Check
        if len(history) >= 2 and history[-1] == 1 and history[-2] == 1:
            return True
            
        # Bet-Call or Raise-Call
        if last == 1:
            if len(history) >= 2 and history[-2] == 2:
                return True
                
        return False
    
    @staticmethod
    def get_payoffs_from_bets(bets, winner, folded=False):
        """
        bets: {0: amount, 1: amount}
        winner: int or -1 (tie)
        folded: bool (if winner won because opponent folded)
        """
        pot = bets[0] + bets[1]
        rewards = {0: 0, 1: 0}
        
        if winner == -1:
            # Tie - return bets (net 0)
            return {0: 0, 1: 0}
            
        # Winner gets pot. Net = Pot - Contribution
        rewards[winner] = pot - bets[winner]
        rewards[1-winner] = -bets[1-winner]
        
        return rewards

    @staticmethod
    def get_winner(hand_p0, hand_p1, board_rank):
        """
        hand_p0, hand_p1: int 0-5
        board_rank: int 0-2 or None
        """
        # Rank: Pair > High Card
        r0 = hand_p0 // 2
        r1 = hand_p1 // 2
        
        if board_rank is not None:
            pair0 = (r0 == board_rank)
            pair1 = (r1 == board_rank)
            
            if pair0 and not pair1: return 0
            if pair1 and not pair0: return 1
            if pair0 and pair1: return -1 # Tie
            
        # High card
        if r0 > r1: return 0
        if r1 > r0: return 1
        return -1
