from treys import Card, Evaluator
import copy

class GameConstants:
    # 0=Preflop, 1=Flop, 2=Turn, 3=River
    STREET_PREFLOP = 0
    STREET_FLOP = 1
    STREET_TURN = 2
    STREET_RIVER = 3
    
    BIG_BLIND = 100
    SMALL_BLIND = 50
    STACK_SIZE = 20000

class NLHERules:
    evaluator = Evaluator()

    @staticmethod
    def get_legal_actions(history, pot, current_bets, stack_sizes):
        """
        Returns list of abstract actions:
        0: FOLD
        1: CHECK/CALL
        2: MIN_RAISE / POT_FRACTION (simplified)
        3: ALL_IN
        """
        # This is a simplification. Real NLHE has continuous action space.
        # We will abstract to: Fold, Check/Call, Pot-Sized Bet/Raise, All-In.
        
        # Determine if we can check
        # We can check if bets are equal.
        can_check = (current_bets[0] == current_bets[1])
        
        actions = []
        
        # Fold is always legal unless we can check?
        # If we can check, checking is strictly better than folding usually, but fold is legal.
        # But typically we map Fold->Check if Check is valid.
        if not can_check:
            actions.append(0) # Fold
        
        actions.append(1) # Check/Call
        
        # Bet/Raise
        # Simplified abstraction: 
        # - Pot Size Bet (or Raise to Pot)
        # - All In
        
        # Logic to calculate legal bet amounts is complex.
        # For this prototype, we just allow "Bet" (mapped to Pot Size) and "All-In".
        
        actions.append(2) # Pot Bet
        actions.append(3) # All In
        
        return actions

    @staticmethod
    def get_winner(hand_p0, hand_p1, board):
        """
        hand_p0, hand_p1: Lists of 2 ints (treys card ints)
        board: List of 3, 4, or 5 ints
        """
        # Evaluate
        # treys.evaluate returns a score (lower is better)
        s0 = NLHERules.evaluator.evaluate(board, hand_p0)
        s1 = NLHERules.evaluator.evaluate(board, hand_p1)
        
        if s0 < s1:
            return 0
        elif s1 < s0:
            return 1
        else:
            return -1 # Tie

    @staticmethod
    def card_str_to_int(card_str):
        # 'Ah' -> int
        # treys.Card.new('Ah')
        return Card.new(card_str)

    @staticmethod
    def int_to_card_str(card_int):
        return Card.int_to_str(card_int)

    @staticmethod
    def get_all_hands():
        # Returns list of all 1326 starting hand combinations (as tuples of 2 ints)
        deck = [Card.new(r+s) for r in '23456789TJQKA' for s in 'shdc']
        hands = []
        for i in range(len(deck)):
            for j in range(i+1, len(deck)):
                hands.append((deck[i], deck[j]))
        return hands

