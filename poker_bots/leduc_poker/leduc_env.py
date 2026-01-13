
import random
from enum import IntEnum


class Rank(IntEnum):
    JACK = 0
    QUEEN = 1
    KING = 2


class LeducHoldemEnv:
    def __init__(self):
        # Deck: 2 suits of J, Q, K (Total 6 cards)
        self.raw_deck = [(r, s) for r in range(3) for s in range(2)]
        self.reset()

    def reset(self):
        self.deck = self.raw_deck.copy()
        random.shuffle(self.deck)

        self.hands = {0: self.deck[0], 1: self.deck[1]}
        self.board_card = self.deck[2]  # The "Flop"

        # Game State
        self.history = []     # List of actions in current round
        self.round = 0        # 0 = Preflop, 1 = Postflop
        self.active_player = 0
        self.done = False

        # Money
        self.pot = 2.0        # Ante 1.0 each
        self.bets = {0: 1.0, 1: 1.0}  # Total wagered by each player

        # Round Management
        # In Leduc, limit is usually 2 bets/raises per round per player
        self.raises_in_round = 0

        return self.get_state(0)

    def get_state(self, player_id):
        """
        Returns: (MyCard, BoardCard, Round, HistoryList)
        BoardCard is None if Round 0
        """
        board = self.board_card if self.round == 1 else None
        # We stringify the history for easier lookup in CFR/Tables
        # But keep it as a list for the Agent
        return {
            'card': self.hands[player_id],
            'board': board,
            'round': self.round,
            'history': self.history.copy(),
            'pot': self.pot
        }

    def get_legal_actions(self):
        """
        Returns list of legal actions:
        0: FOLD
        1: CHECK / CALL
        2: BET / RAISE
        """
        actions = [0, 1, 2]

        # Logic to restrict actions
        current_bet_p0 = self.bets[0]
        current_bet_p1 = self.bets[1]
        opponent = 1 - self.active_player

        diff = self.bets[opponent] - self.bets[self.active_player]

        # Can we check? (Diff is 0)
        can_check = (diff == 0)

        # Can we raise? (Leduc usually caps at 2 raises per round)
        can_raise = (self.raises_in_round < 2)

        legal = []

        if diff > 0:
            legal.append(0)  # FOLD is allowed if facing a bet
            legal.append(1)  # CALL is allowed
        else:
            legal.append(1)  # CHECK (which is mapped to Call/1)

        if can_raise:
            legal.append(2)  # BET/RAISE

        return legal

    def step(self, action):
        """
        Execute action for active_player.
        Actions: 0=FOLD, 1=CHECK/CALL, 2=BET/RAISE
        """
        if action not in self.get_legal_actions():
            raise ValueError(
                f"Illegal action {action} for player {self.active_player} in state {self.history}")

        reward = {0: 0, 1: 0}

        # --- EXECUTE ACTION ---
        if action == 0:  # FOLD
            self.done = True
            opponent = 1 - self.active_player
            # Opponent wins the pot (minus their contribution, so they win what loser put in)
            # Standard: Winner takes all. Net profit = Pot - MyContr
            # But simpler: Loser loses what they bet. Winner wins that amount.
            # Let's return net profit.
            # If I fold, I lose my bets. Opponent wins my bets.
            # Actually easier: Winner gets Pot. Profit = Pot - Invested.
            total_pot = self.bets[0] + self.bets[1]
            reward[opponent] = total_pot - self.bets[opponent]
            reward[self.active_player] = - (self.bets[self.active_player])
            return None, reward, True

        elif action == 1:  # CHECK / CALL
            opponent = 1 - self.active_player
            diff = self.bets[opponent] - self.bets[self.active_player]

            if diff > 0:  # This is a CALL
                self.bets[self.active_player] += diff
                self.pot += diff
                # Calling ends the betting round usually, UNLESS it was a check-check
                # If we call a bet, round ends.
                self._end_round()
            else:  # This is a CHECK
                # If opponent also checked (and not start of round), round ends
                # Start of round: P1 checks. P2 acts.
                # Check-Check
                if len(self.history) > 0 and self.history[-1] == 1:
                    self._end_round()
                else:
                    # Just a check, pass turn
                    self.history.append(1)
                    self.active_player = opponent
                    return self.get_state(self.active_player), {0: 0, 1: 0}, False

        elif action == 2:  # BET / RAISE
            amount = 2.0 if self.round == 0 else 4.0  # Fixed Limit: 2 pre, 4 post

            # If raising, we match opponent first then add amount?
            # Fixed limit: Raises are always incremental.
            opponent = 1 - self.active_player
            call_amount = self.bets[opponent] - self.bets[self.active_player]

            # Total to put in = call_amount + raise_amount
            total_add = call_amount + amount

            self.bets[self.active_player] += total_add
            self.pot += total_add
            self.raises_in_round += 1

            self.history.append(2)
            self.active_player = opponent
            return self.get_state(self.active_player), {0: 0, 1: 0}, False

        # If round ended in the logic above
        if self.done:
            return None, reward, True

        return self.get_state(self.active_player), {0: 0, 1: 0}, False

    def _end_round(self):
        if self.round == 0:
            # Go to round 1
            self.round = 1
            self.history = []  # Reset history for new round? Or keep full?
            # Usually better to reset local round history for simplicity, but keep global?
            # Let's reset action list for the new round but state keeps context
            self.history = []
            self.raises_in_round = 0
            self.active_player = 0  # Preflop P0 starts? Or winner?
            # Standard: Dealer is P0. P0 acts first preflop?
            # Actually Leduc/HeadsUp: P1 (Small Blind) acts first Preflop?
            # Let's stick to: Player 0 always starts Round 1.

        else:
            # Game Over (Showdown)
            self.done = True
            payoffs = self._showdown()
            self.final_payoffs = payoffs
            # We need to return these.
            # This helper is called from step(), which will return them.
            pass

    def _showdown(self):
        # Rank: Pair > High Card
        r0 = self.hands[0][0]
        r1 = self.hands[1][0]
        board = self.board_card[0]

        # Check pairs
        pair0 = (r0 == board)
        pair1 = (r1 == board)

        winner = -1
        if pair0 and not pair1:
            winner = 0
        elif pair1 and not pair0:
            winner = 1
        elif pair0 and pair1:
            # Tie (both pair board? distinct cards? impossible in leduc with 1 board card if hands distinct)
            winner = -1
        # Actually with 1 board card, if both pair, they must have same rank.
        # But deck has 2 of each rank. So P0=Kh, P1=Ks, Board=Kd. Both have pair. Tie.
        else:
            # High card
            if r0 > r1:
                winner = 0
            elif r1 > r0:
                winner = 1
            else:
                winner = -1

        rewards = {0: 0, 1: 0}
        total_pot = self.bets[0] + self.bets[1]

        if winner == -1:
            # Split
            rewards[0] = 0  # Get money back (net 0)
            rewards[1] = 0
        else:
            rewards[winner] = total_pot - self.bets[winner]
            rewards[1-winner] = -self.bets[1-winner]

        return rewards
