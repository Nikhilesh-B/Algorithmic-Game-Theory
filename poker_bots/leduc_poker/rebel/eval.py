import torch
import numpy as np
import random
from .game import LeducRules, GameConstants
from .features import get_features

def evaluate(agent_model, num_games=100, device='cpu'):
    """
    Evaluates ReBeL agent vs Random Agent.
    ReBeL is P0 half time, P1 half time.
    Returns average payoff for ReBeL.
    """
    from .search import CFRSolver
    
    solver = CFRSolver(agent_model, iterations=50, device=device) # Fewer iterations for speed
    total_payoff = 0
    
    for g in range(num_games):
        rebel_p = g % 2 # Alternate
        
        # Init Game
        deck = [(r, s) for r in range(3) for s in range(2)]
        random.shuffle(deck)
        hand_p0 = deck[0]
        hand_p1 = deck[1]
        board = deck[2]
        
        c0 = hand_p0[0] * 2 + hand_p0[1]
        c1 = hand_p1[0] * 2 + hand_p1[1]
        b_rank = board[0]
        
        history = []
        bets = {0: 1.0, 1: 1.0}
        board_state = None
        
        r0 = np.ones(6) / 6.0
        r1 = np.ones(6) / 6.0
        
        while True:
            # Round Check
            if LeducRules.is_terminal_round(history):
                if board_state is None:
                    board_state = b_rank
                    history = []
                    b_idx = board[0] * 2 + board[1]
                    r0[b_idx] = 0
                    r1[b_idx] = 0
                    if r0.sum() > 0: r0 /= r0.sum()
                    if r1.sum() > 0: r1 /= r1.sum()
                    continue
                else:
                    # Showdown
                    # Calculate payoff
                    winner = LeducRules.get_winner(c0, c1, b_rank)
                    payoffs = LeducRules.get_payoffs_from_bets(bets, winner)
                    total_payoff += payoffs[rebel_p]
                    break
            
            # Fold Check
            if len(history) > 0 and history[-1] == 0:
                winner = 1 - (len(history)-1)%2
                payoffs = LeducRules.get_payoffs_from_bets(bets, winner, folded=True)
                total_payoff += payoffs[rebel_p]
                break
                
            active = len(history) % 2
            raises = history.count(2)
            valid = LeducRules.get_legal_actions(history, raises)
            
            if active == rebel_p:
                # ReBeL acts
                # Run search
                t_r0 = torch.tensor(r0, dtype=torch.float32).to(device)
                t_r1 = torch.tensor(r1, dtype=torch.float32).to(device)
                
                avg_strat, _ = solver.solve(history, board_state, bets, t_r0, t_r1)
                
                my_card = c0 if active == 0 else c1
                probs = []
                actions = list(avg_strat.keys())
                for a in actions:
                    probs.append(avg_strat[a][my_card])
                
                if sum(probs) < 1e-9:
                    action = random.choice(actions)
                else:
                    action = random.choices(actions, weights=probs)[0]
                    
                # Update beliefs (ReBeL updates its own belief about itself? 
                # No, we update the beliefs passed to the next search state)
                # We assume the opponent knows our strategy?
                # In self-play training, we update beliefs.
                # In Eval, do we update belief of opponent?
                # The search state requires *public beliefs*.
                # If ReBeL plays according to strat, public belief updates by strat.
                if active == 0:
                    r0 = r0 * avg_strat[action]
                    if r0.sum() > 0: r0 /= r0.sum()
                else:
                    r1 = r1 * avg_strat[action]
                    if r1.sum() > 0: r1 /= r1.sum()
                    
            else:
                # Random Agent
                action = random.choice(valid)
                
                # Update belief for Random Agent?
                # If Random agent plays uniformly, update logic:
                # P(action | card) = 1/len(valid) for all cards.
                # So belief doesn't change relative shape, just normalized?
                # r_opp = r_opp * (1/N).
                # Normalized -> No change.
                # Unless action was invalid for some cards? (Not in Leduc)
                pass 
                
            # Update Game State
            opponent = 1 - active
            if action == 1:
                diff = bets[opponent] - bets[active]
                if diff > 0: bets[active] += diff
            elif action == 2:
                diff = bets[opponent] - bets[active]
                amount = 4.0 if board_state is not None else 2.0
                bets[active] += diff + amount
            
            history.append(action)
            
    return total_payoff / num_games

