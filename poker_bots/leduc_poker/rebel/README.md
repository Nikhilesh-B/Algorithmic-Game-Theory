# ReBeL for Leduc Poker

This is a from-scratch implementation of the ReBeL algorithm (Recursive Belief-based Learning) for Leduc Hold'em Poker.

## Structure

- `game.py`: Implements Leduc Poker rules and payoff logic.
- `models.py`: PyTorch implementation of the Value Network.
- `features.py`: Feature extraction logic (Ranges, Board, Pot, History).
- `search.py`: CFR Solver modified to use the Value Network at leaf nodes (Subgame Solving).
- `train.py`: Self-play data generation and training loop using a Replay Buffer.
- `eval.py`: Evaluation against a Random Agent.
- `main.py`: Entry point for training and evaluation.

## How to Run

Ensure you have `torch` and `numpy` installed.

```bash
python -m poker_bots.leduc_poker.rebel.main
```

## Implementation Details

- **Value Network**: Predicts the expected value for each card (6 cards) for both players (Total 12 outputs) given the public belief state (ranges), board, pot, and history.
- **Search**: Uses CFR (Counterfactual Regret Minimization) for a limited depth (or until end of round) and uses the Value Network to estimate values at leaf nodes (end of round 1).
- **Training**: Generates data via self-play where the agent plays against itself using the search algorithm. The results (beliefs, state -> value) are stored in a replay buffer to train the Value Network.

## Notes

- The implementation assumes a simplified Leduc Poker (Fixed Limit).
- The "History Embedding" in the neural network input is a fixed-size one-hot encoding of the last 10 actions.
- The "Value" target for training is the EV computed by the CFR solver at the root of the subgame.

