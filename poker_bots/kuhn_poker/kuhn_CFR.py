import numpy as np
import random as rand
PASS, BET = 0, 1
NUM_ACTIONS = 2

infoset_map = dict()


class InfosetNode():
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = [0.0]*NUM_ACTIONS
        self.strategy = [0.0]*NUM_ACTIONS
        self.strategy_sum = [0.0]*NUM_ACTIONS

    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        for i in range(NUM_ACTIONS):
            self.strategy[i] = max(self.regret_sum[i], 0)
            normalizing_sum += self.strategy[i]

        for i in range(NUM_ACTIONS):
            if normalizing_sum:
                self.strategy[i] /= normalizing_sum
            else:
                self.strategy[i] = 1/NUM_ACTIONS
            self.strategy_sum[i] += self.strategy[i]*realization_weight

        return self.strategy

    def get_avg_strategy(self):
        avg_strategy = [0.0]*NUM_ACTIONS
        normalizing_sum = sum(self.strategy_sum)

        for i in range(NUM_ACTIONS):
            if normalizing_sum:
                avg_strategy[i] = self.strategy_sum[i]/normalizing_sum
            else:
                avg_strategy[i] = 1/NUM_ACTIONS

        return avg_strategy

    def __str__(self):
        return f"{self.info_set}: {self.get_avg_strategy()}"


class KuhnCFRTrainer():
    def __init__(self, iterations):
        self.iterations = iterations

    def train(self):
        cards = [1, 2, 3]
        util = 0.0
        for _ in range(self.iterations):
            rand.shuffle(cards)
            util += self.cfr(cards, "", 1, 1)

        print("=" * 80)
        print(
            f"Kuhn Poker CFR Training Results ({self.iterations:,} iterations)")
        print("=" * 80)
        print(f"\nAverage game value: {util/self.iterations:.6f}")
        print("\n" + "=" * 80)
        print("Final Strategy Distribution")
        print("=" * 80)

        # Sort information sets for better readability
        sorted_infosets = sorted(infoset_map.keys())

        for key in sorted_infosets:
            node = infoset_map[key]
            avg_strategy = node.get_avg_strategy()

            # Parse the information set
            card = key[0]
            history = key[1:] if len(key) > 1 else ""

            print(
                f"\nCard: {card}, History: '{history if history else 'START'}'")
            print(
                f"  Pass: {avg_strategy[0]:.4f} ({avg_strategy[0]*100:.2f}%)")
            print(
                f"  Bet:  {avg_strategy[1]:.4f} ({avg_strategy[1]*100:.2f}%)")

        print("\n" + "=" * 80)

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opps = 1-player

        if plays > 1:
            terminal_pass = history[-1] == 'p'
            double_bet = history[-2:] == 'bb'
            player_higher = cards[player] > cards[opps]
            if terminal_pass:
                if history == 'pp':
                    if player_higher:
                        return 1
                    else:
                        return -1
                else:
                    return 1
            elif double_bet:
                if player_higher:
                    return 2
                else:
                    return -2

        infoset = str(cards[player])+history

        if infoset in infoset_map:
            infoset_node = infoset_map[infoset]
        else:
            infoset_node = InfosetNode(infoset)
            infoset_map[infoset] = infoset_node

        if player == 0:
            strategy = infoset_node.get_strategy(p0)
        else:
            strategy = infoset_node.get_strategy(p1)

        utilities = [0.0]*NUM_ACTIONS
        node_util = 0.0

        for i in range(NUM_ACTIONS):
            next_history = history + ('p' if i == 0 else 'b')
            if player == 0:
                utilities[i] = -1 * \
                    self.cfr(cards, next_history, p0*strategy[i], p1)
            else:
                utilities[i] = -1 * \
                    self.cfr(cards, next_history, p0, p1*strategy[i])
            node_util += utilities[i]*strategy[i]

        for i in range(NUM_ACTIONS):
            regret = utilities[i]-node_util
            infoset_node.regret_sum[i] += regret*(p1 if player == 0 else p0)

        return node_util


if __name__ == "__main__":
    iterations = 100_000
    trainer = KuhnCFRTrainer(iterations)
    trainer.train()
