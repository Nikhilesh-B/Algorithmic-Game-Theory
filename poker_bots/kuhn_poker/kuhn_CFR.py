import numpy as np
import random as rand
PASS = 0, BET = 1
NUM_ACTIONS = 2

infoset_map = dict()


class InfosetNode():
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = [0.0]*NUM_ACTIONS
        self.strategy = [0.0]*NUM_ACTIONS
        self.strategy_sum = [0.0]*NUM_ACTIONS

    def get_startegy(self, realization_weight):
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
    def train(self, iterations):
        cards = [1, 2, 3]
        util = 0.0
        for _ in range(iterations):
            cards = rand.shuffle(cards)
            util += self.cfr(cards, "", 1, 1)
        print(f"Average game value: {util/iterations}")
        print("X"*100)
        for key in infoset_map:
            node = infoset_map[key]
            print(node)
        print("X"*100)

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
                elif double_bet:
                    if player_higher:
                        return 2
                    else:
                        return -2

        infoset = cards[player]+history

        if infoset in infoset_map:
            infoset_node = infoset_map[infoset]
        else:
            infoset_node = InfosetNode()
            InfosetNode.info_set = infoset
            infoset_map[infoset] = infoset_node

        if player == 0:
            strategy = infoset_node.get_startegy(p0)
        else:
            strategy = infoset_node.get_startegy(p1)

        utilities = [0.0]*NUM_ACTIONS
        node_util = 0.0

        for i in range(NUM_ACTIONS):
            next_history = history + 'p' if i == 0 else 'b'
            if player == 0:
                utilities[i] = - \
                    self.cfr(cards, next_history, p0*strategy[i], p1)
            else:
                utilities[i] = - \
                    self.cfr(cards, next_history, p0, p1*strategy[i])
            node_util += utilities[i]*strategy[i]

        for i in range(NUM_ACTIONS):
            regret = utilities[i]-node_util
            infoset_node.regret_sum[i] += regret*(p0 if player == 0 else p1)


if __name__ == "__main__":
    iterations = 100_000
    trainer = KuhnCFRTrainer(iterations)
