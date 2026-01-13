import random
from .client import SlumbotClient, BIG_BLIND, STACK_SIZE, NUM_STREETS

class Agent:
    def get_action(self, state_dict, hole_cards, board):
        raise NotImplementedError

class RandomAgent(Agent):
    def get_action(self, state_dict, hole_cards, board):
        # Determine valid actions
        # Check (k), Call (c), Fold (f), Bet (b)
        
        last_bet_size = state_dict['last_bet_size']
        total_last_bet_to = state_dict['total_last_bet_to']
        street_last_bet_to = state_dict['street_last_bet_to']
        
        valid_actions = []
        
        # Can Check?
        if last_bet_size == 0:
            valid_actions.append('k')
        else:
            # Facing a bet
            valid_actions.append('f')
            valid_actions.append('c')
            
        # Can Bet/Raise?
        # Max bet is remaining stack
        remaining = STACK_SIZE - total_last_bet_to
        if remaining > 0:
            # Min raise?
            # Must raise by at least last_bet_size (unless all-in)
            # Min bet size (amount to ADD)
            min_raise = max(BIG_BLIND, last_bet_size)
            
            # But 'b' command takes "total amount on this street".
            # Example: 
            # Opp bets 200 (total 200 on street).
            # I call 200. Total 200.
            # I raise to 400 (add 200). Command b400.
            
            # The 'state_dict' logic says: 
            # new_last_bet_size = new_street_last_bet_to - street_last_bet_to
            # So input is "street_last_bet_to".
            
            # Min new street total:
            # street_last_bet_to + min_raise
            
            min_bet_total = street_last_bet_to + min_raise
            
            if min_bet_total <= remaining + street_last_bet_to: # Wait.
                # remaining is "stack size - total put in pot".
                # total_last_bet_to includes previous streets + current street so far.
                # So max I can put in additionally is 'remaining'.
                # My current contribution on this street? 
                # If I am to act, I might have put something in? (e.g. SB preflop)
                # But 'street_last_bet_to' is the HIGHEST bet on the street.
                # If I am acting, I have put in less or equal.
                pass
                
            valid_actions.append('b')
            
        action = random.choice(valid_actions)
        
        if action == 'b':
            # Pick a size
            # For random agent, let's just do Min Raise or All-In
            min_raise = max(BIG_BLIND, last_bet_size)
            if min_raise > remaining:
                # Can only go all in
                # All in amount on this street?
                # Total stack = 20000.
                # Total put in = total_last_bet_to (by opponent? max?)
                # If I go all in, my total contrib = 20000.
                # My contrib on this street = 20000 - (total_last_bet_to - street_last_bet_to) ?
                # No. 
                # Total Pot = X.
                # Max I can bet:
                # I can put my whole stack in.
                # My current stack = STACK_SIZE - (Amount I put in).
                # Wait, 'total_last_bet_to' tracks the MAX put in by anyone?
                # Yes. "total_last_bet_to counts all chips put into the pot" (per player? No, usually 'bet to').
                # "total_last_bet_to" in ParseAction seems to track the CURRENT bet level.
                
                # If I go all-in, I match the current level and add the rest of my stack.
                # Actually, Slumbot API says: "Bet sizes are the number of chips that the player has put into the pot *on that street* (only)."
                
                # So if I go all in:
                # My Total = STACK_SIZE.
                # My Street Total = STACK_SIZE - (My Previous Streets Total).
                # We don't track 'My Previous Streets Total' in ParseAction.
                # However, we know STACK_SIZE is 20000.
                # We can calculate 'Amount in Pot before this street'.
                # total_last_bet_to - street_last_bet_to = Amount put in by Leader on previous streets.
                # Assuming I matched previous streets (I must have to be here).
                prev_streets = total_last_bet_to - street_last_bet_to
                max_street_bet = STACK_SIZE - prev_streets
                
                amount = max_street_bet
            else:
                # Random amount between min and max?
                prev_streets = total_last_bet_to - street_last_bet_to
                max_street_bet = STACK_SIZE - prev_streets
                
                min_street_bet = street_last_bet_to + min_raise
                
                if min_street_bet >= max_street_bet:
                    amount = max_street_bet
                else:
                    # Random choice
                    amount = random.randint(min_street_bet, max_street_bet)
                    
            return f"b{amount}"
            
        return action

