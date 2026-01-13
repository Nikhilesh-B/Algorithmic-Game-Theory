import random
import torch
import numpy as np
from treys import Card
from .client import SlumbotClient
from .agent import Agent
from .rebel.models import NLHEValueNetwork
from .rebel.search import NLHESearch
from .rebel.game import NLHERules


class ReBeLAgent(Agent):
    def __init__(self, model_path="rebel_nlhe.pt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = NLHEValueNetwork().to(self.device)

        try:
            self.model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            print(f"Loaded ReBeL model from {model_path}")
        except:
            print("Could not load model, using random initialization")

        self.model.eval()
        self.search = NLHESearch(self.model, device=self.device)
        self.reset_hand()

    def reset_hand(self):
        self.r0 = np.ones(1326) / 1326.0
        self.r1 = np.ones(1326) / 1326.0
        self._all_hands = NLHERules.get_all_hands()
        self._hand_strength = np.array(
            [self._hand_strength_bucket(h) for h in self._all_hands])

    def _hand_strength_bucket(self, hand):
        """
        Very rough preflop bucket based only on ranks.
        Returns float in [0, 1]; higher = stronger.
        """
        r1 = Card.get_rank_int(hand[0])  # 0..12 (2..A)
        r2 = Card.get_rank_int(hand[1])
        suited = Card.get_suit_int(hand[0]) == Card.get_suit_int(hand[1])
        ranks = sorted([r1, r2], reverse=True)
        high, low = ranks
        # Pair
        if r1 == r2:
            return 0.8 + high / 20.0
        # Broadways
        if high >= 10 and low >= 8:
            return 0.65 + (high + low) / 30.0 + (0.05 if suited else 0)
        # Suited connectors
        if suited and high - low == 1 and high >= 7:
            return 0.55
        # Suited ace
        if suited and high == 12:
            return 0.5
        # Default
        return 0.3 + (high / 30.0)

    def _history_from_action_str(self, action_str):
        mapping = {'f': 0, 'c': 1, 'k': 1, 'b': 2}
        hist = []
        for ch in action_str:
            if ch in mapping:
                hist.append(mapping[ch])
        return hist

    def _update_opponent_belief(self, action_str):
        """
        Very rough belief shift based on opponent's last action.
        """
        if not action_str:
            return
        last = ''
        num_buf = ''
        for ch in action_str[::-1]:
            if ch == '/':
                if last:
                    break
                continue
            if ch.isdigit() and not last:
                num_buf = ch + num_buf
                continue
            if ch in ['b', 'c', 'k', 'f']:
                last = ch
                break
        bet_size = 0
        if last == 'b' and num_buf:
            try:
                bet_size = int(num_buf)
            except:
                bet_size = 0
        if last == 'b' and bet_size > 0:
            scale = min(1.0, bet_size / 5000.0)
            w = 0.5 + scale * 0.5
            weights = 0.5 * (1.0 - w) + w * self._hand_strength
        elif last in ['c', 'k']:
            weights = np.ones_like(self.r1)
        else:
            weights = np.ones_like(self.r1)
        self.r1 = self.r1 * weights
        s = self.r1.sum()
        if s > 1e-9:
            self.r1 /= s

    def get_action(self, state_dict, hole_cards, board):
        # hole_cards: ['Ac', '9d']
        c1 = Card.new(hole_cards[0])
        c2 = Card.new(hole_cards[1])
        my_hand = (c1, c2)

        board_ints = [Card.new(c) for c in board]

        # Estimate Pot more conservatively:
        # Pot ~ last street total * 2 (assume symmetrical) plus blinds (150).
        total_last = state_dict['total_last_bet_to']
        pot = max(150.0, float(total_last * 2))
        # Remaining stacks (very rough): starting stack - total_last
        stacks = [max(0.0, 20000.0 - total_last),
                  max(0.0, 20000.0 - total_last)]
        # Build crude history from action string
        action_str = state_dict.get('action_full', '') if isinstance(
            state_dict, dict) else ''
        history = self._history_from_action_str(action_str)
        # Update opponent belief based on last observed action
        self._update_opponent_belief(action_str)

        # Update Beliefs?
        # For prototype, we reset beliefs every move (Stateless ReBeL - treating every move as new subgame root)
        # To do better, we should carry over r0/r1 and update based on opponent action.

        # Run Search
        strategy, values = self.search.solve_subgame(
            self.r0, self.r1, board_ints, pot, stacks, history
        )

        action_idx = self.search.get_action_from_strategy(strategy, my_hand)

        # 0: Fold, 1: Check/Call, 2: Pot, 3: AllIn
        # Hand-strength-aware guardrails
        strength = self._hand_strength_bucket(my_hand)
        facing_bet = state_dict['last_bet_size'] > 0
        # Avoid punting all-in with trash preflop
        if action_idx == 3 and strength < 0.7:
            action_idx = 1 if facing_bet else 2  # downgrade to call/check or small raise

        if action_idx == 0:
            # If we can check, prefer check over fold
            if state_dict['last_bet_size'] == 0:
                return 'k'
            return 'f'
        elif action_idx == 1:
            if facing_bet:
                return 'c'
            return 'k'
        elif action_idx == 2:
            # Pot raise heuristic: raise to ~pot or 3x last bet, capped by stack
            call_amt = state_dict['last_bet_size']
            target = state_dict['street_last_bet_to'] + \
                max(300, max(call_amt * 3, pot))
            max_street = min(20000, target)
            return f"b{int(max_street)}"
        elif action_idx == 3:
            # All-in capped at stack
            return f"b{int(stacks[0])}"

        return 'k'
