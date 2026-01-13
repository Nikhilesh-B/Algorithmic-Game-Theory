import torch
import numpy as np
from treys import Card

def get_card_index(card_int):
    """
    Maps treys card int to 0-51 index.
    """
    rank = Card.get_rank_int(card_int) # 0-12
    suit_int = Card.get_suit_int(card_int) # 1, 2, 4, 8
    
    # Map suit power of 2 to 0-3
    if suit_int == 1: suit_idx = 0
    elif suit_int == 2: suit_idx = 1
    elif suit_int == 4: suit_idx = 2
    elif suit_int == 8: suit_idx = 3
    else: suit_idx = 0 # Should not happen
    
    return rank * 4 + suit_idx

def get_nlhe_features(r0, r1, board_cards, pot, stacks, history):
    """
    Construct input tensor.
    r0, r1: (1326,) probabilities
    board_cards: list of ints
    pot: float
    stacks: list of 2 floats
    history: list of action ints (padded)
    """
    # 1. Ranges
    r0_t = torch.as_tensor(r0, dtype=torch.float32)
    r1_t = torch.as_tensor(r1, dtype=torch.float32)
    
    # 2. Board
    # 52-dim one-hot of board cards
    board_vec = torch.zeros(52)
    for c in board_cards:
        idx = get_card_index(c)
        if 0 <= idx < 52:
            board_vec[idx] = 1.0
        
    # 3. Pot/Stacks
    # Normalize by Big Blind (100)
    pot_norm = pot / 100.0
    stack_norm = [s / 100.0 for s in stacks]
    meta = torch.tensor([pot_norm] + stack_norm, dtype=torch.float32)
    
    # 4. History
    # Pad to 50
    hist_vec = torch.zeros(50)
    for i, a in enumerate(history[-50:]):
        hist_vec[i] = float(a)
        
    return torch.cat([r0_t, r1_t, board_vec, meta, hist_vec])
