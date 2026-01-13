import torch
import torch.nn.functional as F

def get_features(range_p0, range_p1, board_rank, history, pot):
    """
    Construct input tensor for Value Network.
    range_p0: tensor (6,)
    range_p1: tensor (6,)
    board_rank: int or None
    history: list of ints (actions)
    pot: float
    """
    # 1. Ranges (Probabilities)
    # Ensure they are normalized? The network expects probabilities.
    # Concatenate p0 and p1
    r0 = range_p0
    r1 = range_p1
    
    # 2. Board
    # One-hot encoding of board. 
    # If None (Round 1), maybe use [1, 0, 0, 0]
    # If J, Q, K (Round 2), use [0, 1, 0, 0], [0, 0, 1, 0], etc.
    board_vec = torch.zeros(4)
    if board_rank is None:
        board_vec[0] = 1.0
    else:
        # board_rank 0, 1, 2 maps to indices 1, 2, 3
        board_vec[board_rank + 1] = 1.0
        
    # 3. Pot
    # Normalize pot? Max pot in Leduc is roughly 2 + 2*4 + 2*8 = ~26?
    # Let's just pass raw or log pot.
    pot_vec = torch.tensor([pot / 20.0]) # Rough normalization
    
    # 4. History
    # Fixed size history embedding.
    # Max actions per round is small. Total actions ~10.
    # Let's map actions 0, 1, 2 to one-hot vectors.
    # Pad to length 10.
    MAX_LEN = 10
    hist_vec = torch.zeros(MAX_LEN * 3)
    for i, a in enumerate(history[-MAX_LEN:]):
        # Action a is 0, 1, 2
        # Index = i * 3 + a
        hist_vec[i*3 + a] = 1.0
        
    # Concatenate all
    # ranges need to be flat
    return torch.cat([r0, r1, board_vec, pot_vec, hist_vec])

